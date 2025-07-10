#!/usr/bin/env python3

import os
import glob
import time
import argparse
import numpy as np
from astropy.stats import mad_std
from scipy.signal import detrend

from exotedrf.utils       import parse_config, unpack_input_dir, fancyprint
from exotedrf.stage1      import run_stage1
from exotedrf.stage2      import run_stage2
from exotedrf.stage3      import run_stage3

# ————————————————————————————————————————————————————————
# 1) Cost function: robust fractional scatter of Stage 3 white-light
# ————————————————————————————————————————————————————————
def compute_white_light(dm):
    """
    Extract the white-light curve (sum over all pixels) from a CubeModel-like dm.
    """
    # dm.data after RampFitStep -> CubeModel: shape (nints, dimy, dimx)
    wl = dm.data.reshape(dm.data.shape[0], -1).sum(axis=1)
    return wl

def compute_spectral(dm):
    """
    Extract a toy spectral lightcurve: sum down columns, averaging across wavelength.
    """
    # reshape to (nints, dimy, dimx) then average over x→ wavelength
    spec = dm.data.reshape(dm.data.shape[0], dm.data.shape[1], -1)
    return spec.mean(axis=2)  # shape (nints, dimy)

def cost_function(dm, w1=0.5, w2=0.5):
    """
    Weighted sum of:
      1) white-light fractional scatter
      2) mean fractional scatter across spectral rows
    """
    wl   = compute_white_light(dm)    # shape (nints,)
    spec = compute_spectral(dm)       # shape (nints, nrows)

    # fractional scatter of white-light curve
    frac_wl = mad_std(wl) / abs(np.median(wl))

    # fractional scatter for each spectral row, then mean
    frac_spec_rows = [
        mad_std(spec[:, i]) / abs(np.median(spec[:, i]))
        for i in range(spec.shape[1])
    ]
    frac_spec = np.mean(frac_spec_rows)

    return w1 * frac_wl + w2 * frac_spec


# ————————————————————————————————————————————————————————
# 2) Main optimizer
# ————————————————————————————————————————————————————————
def main():
    parser = argparse.ArgumentParser(
        description="Coordinate-descent optimizer for exoTEDRF Stages 1–3"
    )
    parser.add_argument(
        "--config", default="run_DMS.yaml",
        help="Path to your DMS config YAML"
    )
    parser.add_argument(
        "--instrument", required=True,
        choices=["NIRISS", "NIRSPEC", "MIRI"],
        help="Which instrument to optimize"
    )
    args = parser.parse_args()

    # — load config & discover segment files —
    cfg = parse_config(args.config)
    input_files = unpack_input_dir(
        cfg['input_dir'],
        mode            = cfg['observing_mode'],
        filetag         = cfg['input_filetag'],
        filter_detector = cfg['filter_detector']
    )
    if len(input_files) == 0:
        fancyprint(f"[WARN] `unpack_input_dir` found ZERO files in {cfg['input_dir']}.")
        fancyprint("       Falling back to glob on '*.fits' in that directory.")
        input_files = sorted(glob.glob(os.path.join(cfg['input_dir'], "*.fits")))

    if len(input_files) == 0:
        raise RuntimeError(f"No FITS files found in {cfg['input_dir']}!")

    fancyprint(f"Using {len(input_files)} segment(s) from {cfg['input_dir']}")

    # — build parameter grid —
    param_ranges = {}
    if args.instrument == "NIRISS":
        param_ranges.update({
            "soss_inner_mask_width": [20, 40, 80],
            "soss_outer_mask_width": [35, 70,140],
            "jump_threshold":        [5, 15, 30],
            "time_jump_threshold":   [3, 10, 20],
        })
    elif args.instrument == "NIRSPEC":
        param_ranges.update({
            "nirspec_mask_width":   [8, 16, 32],
            "jump_threshold":       [5, 15, 30],
            "time_jump_threshold":  [3, 10, 20],
        })
    else:  # MIRI
        param_ranges.update({
            "miri_drop_groups":      [6, 12, 24],
            "jump_threshold":        [5, 15, 30],
            "time_jump_threshold":   [3, 10, 20],
            "miri_trace_width":      [10, 20, 40],
            "miri_background_width": [7, 14, 28],
        })

    # always sweep these
    param_ranges.update({
        "space_outlier_threshold": [5, 15, 30],
        "time_outlier_threshold":  [3, 10, 20],
        "pca_components":          [5, 10, 20],
        "extract_width":           [15, 30, 60],
    })

    # sweep in a fixed order
    param_order = list(param_ranges.keys())

    # start at the median of each range
    current = {k: int(np.median(v)) for k, v in param_ranges.items()}

    # prepare the log
    total_steps = sum(len(v) for v in param_ranges.values())
    count = 1
    logf = open("Cost_function_V2.txt", "w")
    logf.write("\t".join(param_order) + "\tduration_s\tcost\n")

    # coordinate‐descent
    for key in param_order:
        best_cost, best_val = None, current[key]
        fancyprint(f"\n→ Optimizing {key} (others fixed at "
                   f"{ {k:current[k] for k in current if k!=key} })")

        for trial in param_ranges[key]:
            trial_params = current.copy()
            trial_params[key] = trial

            t0 = time.perf_counter()

            # Stage 1
            st1 = run_stage1(
                input_files,
                mode=cfg['observing_mode'],
                save_results=False, skip_steps=[],
                rejection_threshold      = trial_params.get("jump_threshold"),
                time_rejection_threshold = trial_params.get("time_jump_threshold"),
                soss_inner_mask_width    = trial_params.get("soss_inner_mask_width"),
                soss_outer_mask_width    = trial_params.get("soss_outer_mask_width"),
                nirspec_mask_width       = trial_params.get("nirspec_mask_width"),
                miri_drop_groups         = trial_params.get("miri_drop_groups"),
                **cfg.get('stage1_kwargs', {})
            )

            # Stage 2
            st2, centroids = run_stage2(
                st1,
                mode=cfg['observing_mode'],
                save_results=False, skip_steps=[],
                space_thresh             = trial_params["space_outlier_threshold"],
                time_thresh              = trial_params["time_outlier_threshold"],
                pca_components           = trial_params["pca_components"],
                soss_inner_mask_width    = trial_params.get("soss_inner_mask_width"),
                soss_outer_mask_width    = trial_params.get("soss_outer_mask_width"),
                nirspec_mask_width       = trial_params.get("nirspec_mask_width"),
                miri_trace_width         = trial_params.get("miri_trace_width"),
                miri_background_width    = trial_params.get("miri_background_width"),
                **cfg.get('stage2_kwargs', {})
            )

            # Stage 3
            st3 = run_stage3(
                st2,
                centroids=centroids,
                save_results=False, skip_steps=[],
                extract_width=trial_params["extract_width"],
                **cfg.get('stage3_kwargs', {})
            )

            dt   = time.perf_counter() - t0
            cost = cost_function(st3)

            # log this trial
            vals = [str(trial_params[k]) for k in param_order]
            logf.write("\t".join(vals + [f"{dt:.1f}", f"{cost:.6f}"]) + "\n")

            # update best
            if best_cost is None or cost < best_cost:
                best_cost, best_val = cost, trial

            print(f"[{count}/{total_steps}] {key}={trial} → "
                  f"cost={cost:.6f} ({dt:.1f}s)")
            count += 1

        current[key] = best_val
        fancyprint(f"✔ Best {key} = {best_val} (cost={best_cost:.6f})")

    logf.close()
    fancyprint("\n=== FINAL OPTIMUM ===")
    fancyprint(current)
    fancyprint("Log saved to Cost_function_V2.txt")


if __name__ == "__main__":
    main()
