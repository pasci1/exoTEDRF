#!/usr/bin/env python3
"""
optimizer_V3.py: Coordinate-descent over Stage 1, 2 & 3 parameters for exoTEDRF.
Automatically picks only the parameters relevant to your instrument (NIRISS/SOSS, NIRSpec, or MIRI).
Usage:
    python3 optimizer_V3.py --config Optimize_WASP39b/run_WASP39b.yaml
"""
import os
import glob
import time
import argparse

import numpy as np
from astropy.stats import mad_std
from jwst import datamodels

from exotedrf.utils import parse_config
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3

# -----------------------------------------------------------------------------
# 1) cost function: robust white-light scatter
# -----------------------------------------------------------------------------
def compute_white_light(dm):
    """Sum over all pixels of a CubeModel to get a 1D light curve."""
    data = dm.data  # shape (nints, ny, nx)
    return data.reshape(data.shape[0], -1).sum(axis=1)

def cost_function(dm1, w1=1.0, w2=1.0):
    """
    Measure the fractional scatter of the white‐light curve coming out
    of Stage 1.  You can of course switch this to dm3 if you prefer.
    """
    wl = compute_white_light(dm1)
    frac = mad_std(wl) / np.abs(np.median(wl))
    return w1*frac + w2*frac  # or modify to be w1*frac + w2*(something else)


# -----------------------------------------------------------------------------
# 2) main & coordinate-descent
# -----------------------------------------------------------------------------
def main():
    # --- parse CLI ---
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="run_WASP39b.yaml",
                   help="path to your run_*.yaml")
    p.add_argument("--w1", type=float, default=1.0,
                   help="weight on Stage 1 fractional scatter")
    p.add_argument("--w2", type=float, default=1.0,
                   help="extra weight term (unused placeholder)")
    p.add_argument("--log", "-o", default="Cost_function_V3.txt",
                   help="output TSV log")
    args = p.parse_args()

    # --- load config & pick off everything ---
    cfg       = parse_config(args.config)
    cfg_dir   = os.path.dirname(args.config) or "."
    mode      = cfg["observing_mode"]            # e.g. "NIRSpec/G395H"
    instrument= mode.split("/")[0].upper()       # "NIRSPEC", "NIRISS" or "MIRI"
    baseline  = cfg.get("baseline_ints", list(range(60))) 

    # --- find a single seg001 slice in your input_dir ---
    indir = os.path.join(cfg_dir, cfg["input_dir"])
    pattern = os.path.join(indir, "*seg001*_uncal.fits")
    matches = sorted(glob.glob(pattern))
    if len(matches)==0:
        raise FileNotFoundError(f"No seg001 slice found with '{pattern}'")
    seg = matches[0]
    print("→ Using slice:", seg)

    # --- open + slice to K=60 for speed ---
    dm_full = datamodels.open(seg)
    K       = min(60, dm_full.data.shape[0])
    dm_slice= dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end   = K
    dm_slice.meta.exposure.nints            = K
    dm_full.close()

    # --- build a dict of only those parameters we want to sweep ---
    param_ranges = {}
    # Stage 1
    if instrument=="NIRISS":
        param_ranges["soss_inner_mask_width"] = [20, 40, 80]
        param_ranges["soss_outer_mask_width"] = [35, 70, 140]
    if instrument in ("NIRISS","NIRSPEC"):
        key = "nirspec_mask_width" if instrument=="NIRSPEC" else None
        if key:
            param_ranges[key] = [8, 16, 32]
    if instrument=="MIRI":
        param_ranges["miri_drop_groups"] = [6, 12, 24]

    # jumps (all instruments use the same jump thresholds)
    param_ranges["jump_threshold"]      = [5, 15, 30]
    param_ranges["time_jump_threshold"] = [3, 10, 20]

    # Stage 2 (2D calibrations)
    if instrument=="MIRI":
        param_ranges["miri_trace_width"]      = [10, 20, 40]
        param_ranges["miri_background_width"] = [7, 14, 28]
    param_ranges["space_outlier_threshold"] = [5, 15, 30]
    param_ranges["time_outlier_threshold"]  = [3, 10, 20]
    param_ranges["pca_components"]          = [5, 10, 20]

    # Stage 3 (1D extraction)
    param_ranges["extract_width"] = [15, 30, 60]

    # keep a stable sweep order
    param_order = list(param_ranges.keys())

    # --- initialize current best at the config defaults or the median of the sweep ---
    current = {}
    for p in param_order:
        if p in cfg:
            current[p] = cfg[p]
        else:
            current[p] = int(np.median(param_ranges[p]))
    # pass along your weights
    current["w1"] = args.w1
    current["w2"] = args.w2

    # --- prepare logfile ---
    with open(args.log, "w") as log:
        # header
        log.write("\t".join(param_order + ["duration_s","J"]) + "\n")

        best_J = None
        step   = 1
        total  = sum(len(param_ranges[p]) for p in param_order)

        # coordinate-descent
        for key in param_order:
            print(f"\n→ Optimizing {key} ({step}/{len(param_order)})")
            best_val = current[key]

            for trial in param_ranges[key]:
                trial_params = current.copy()
                trial_params[key] = trial

                # --- run all three stages & score ---
                t0 = time.perf_counter()
                # Stage 1
                results1 = run_stage1(
                    [dm_slice],
                    mode=mode,
                    baseline_ints=baseline,
                    save_results=False,
                    skip_steps=[],
                    # pass only the kwargs this step needs:
                    **{ k: trial_params[k]
                        for k in ("soss_inner_mask_width",
                                  "soss_outer_mask_width",
                                  "nirspec_mask_width",
                                  "miri_drop_groups",
                                  "jump_threshold",
                                  "time_jump_threshold")
                        if k in trial_params }
                )
                dm1 = results1[0]

                # Stage 2
                results2, centroids = run_stage2(
                    results1,
                    mode=mode,
                    baseline_ints=baseline,
                    save_results=False,
                    force_redo=False,
                    skip_steps=[],
                    **{ k: trial_params[k]
                        for k in ("space_outlier_threshold",
                                  "time_outlier_threshold",
                                  "pca_components",
                                  "miri_trace_width",
                                  "miri_background_width")
                        if k in trial_params }
                )

                # Stage 3
                dm3 = run_stage3(
                    results2,
                    save_results=False,
                    force_redo=False,
                    extract_method=cfg.get("extract_method","box"),
                    extract_width=trial_params["extract_width"],
                    centroids=centroids
                )

                dt = time.perf_counter() - t0
                J  = cost_function(dm1, trial_params["w1"], trial_params["w2"])

                # log & print
                row = [ str(trial_params[p]) for p in param_order ] \
                      + [ f"{dt:.1f}", f"{J:.6f}" ]
                log.write("\t".join(row)+"\n")
                print(f"   {key}={trial} → J={J:.6f}  ({dt:.1f}s)")

                # update best
                if best_J is None or J < best_J:
                    best_J, best_val = J, trial

            # commit best for this key
            current[key] = best_val
            print(f"✔→  Best {key} = {best_val}  (J={best_J:.6f})")
            step += 1

        # final summary in log
        log.write("\n# Final optimized parameters:\n")
        for p in param_order:
            log.write(f"# {p} = {current[p]}\n")
        log.write(f"# Final cost J = {best_J:.6f}\n")

    print("\n=== DONE ===")
    print("Results written to", args.log)


if __name__ == "__main__":
    main()
