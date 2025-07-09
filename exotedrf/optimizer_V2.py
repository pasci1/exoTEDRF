#!/usr/bin/env python3
"""
optimizer_V2.py: Coordinate‐descent optimizer for exoTEDRF pipeline parameters across
Stage 1, 2, and 3.  Sweeps only the parameters relevant to your instrument.
"""
import time
import argparse
import os
import numpy as np
from jwst import datamodels
from exotedrf.utils import parse_config
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3
from astropy.stats import mad_std

def compute_white_light_from_dm(dm):
    """Sum over all pixels to extract white‐light curve."""
    wl = dm.data.reshape(dm.data.shape[0], -1).sum(axis=1)
    return wl

def cost_function(dm, w1=1.0, w2=1.0):
    """Robust fractional scatter of the white‐light curve."""
    wl = compute_white_light_from_dm(dm)
    frac = mad_std(wl) / np.abs(np.median(wl))
    return w1 * frac + w2 * frac

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="run_WASP39b.yaml",
                   help="Path to your run_*.yaml config")
    p.add_argument("--log", default="Cost_function_V2.txt",
                   help="Output TSV log")
    p.add_argument("--w1", type=float, default=1.0,
                   help="Cost weight for Stage 1 scatter")
    p.add_argument("--w2", type=float, default=1.0,
                   help="Cost weight for Stage 2+3 (placeholder)")
    args = p.parse_args()

    # ─── Parse & locate your slice ─────────────────────────────────────────
    cfg = parse_config(args.config)
    cfg_path = os.path.abspath(args.config)
    cfg_dir  = os.path.dirname(cfg_path)

    mode       = cfg["observing_mode"]            # e.g. "NIRISS/SOSS"
    instrument = mode.split("/")[0].upper()       # "NIRISS" / "NIRSPEC" / "MIRI"

    # input_dir in the YAML is usually relative to the config file:
    in_dir = cfg["input_dir"]
    data_dir = in_dir if os.path.isabs(in_dir) else os.path.join(cfg_dir, in_dir)

    slice_name = "jw01366003001_04101_00001-seg001_nrs1_uncal.fits"
    seg = os.path.join(data_dir, slice_name)
    if not os.path.isfile(seg):
        raise FileNotFoundError(f"Slice file not found: {seg!r}")

    dm_full = datamodels.open(seg)
    K       = min(60, dm_full.data.shape[0])
    dm_slice       = dm_full.copy()
    dm_slice.data  = dm_full.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end   = K
    dm_slice.meta.exposure.nints            = K
    dm_full.close()

    # ─── Build your parameter grid ──────────────────────────────────────────
    param_ranges = {}
    if instrument == "NIRISS":
        param_ranges.update({
            "soss_inner_mask_width": [20, 40, 80],
            "soss_outer_mask_width": [35, 70, 140],
            "jump_threshold":         [5, 15, 30],
            "time_jump_threshold":    [3, 10, 20],
        })
    elif instrument == "NIRSPEC":
        param_ranges.update({
            "nirspec_mask_width":  [8, 16, 32],
            "jump_threshold":      [5, 15, 30],
            "time_jump_threshold": [3, 10, 20],
        })
    else:  # MIRI
        param_ranges.update({
            "miri_drop_groups":      [6, 12, 24],
            "jump_threshold":        [5, 15, 30],
            "time_jump_threshold":   [3, 10, 20],
            "miri_trace_width":      [10, 20, 40],
            "miri_background_width": [7, 14, 28],
        })

    # Stage 2 + Stage 3 always:
    param_ranges["space_outlier_threshold"]  = [5, 15, 30]
    param_ranges["time_outlier_threshold"]   = [3, 10, 20]
    param_ranges["pca_components"]           = [5, 10, 20]
    param_ranges["extract_width"]            = [15, 30, 60]

    param_order = list(param_ranges.keys())

    # Initialize with defaults from the config (or the median of each range):
    current = {
        k: cfg.get(k, int(np.median(v)))
        for k, v in param_ranges.items()
    }
    current.update(w1=args.w1, w2=args.w2)

    # ─── Helper: evaluate one trial ────────────────────────────────────────
    def evaluate_one(params):
        t0 = time.perf_counter()

        # ---- Stage 1 ----
        results1 = run_stage1(
            [dm_slice],
            mode=mode,
            baseline_ints=cfg["baseline_ints"],
            save_results=False,
            skip_steps=[],
            # pass only the keys for your instrument
            soss_inner_mask_width = params.get("soss_inner_mask_width",
                                              cfg.get("soss_inner_mask_width", 40)),
            soss_outer_mask_width = params.get("soss_outer_mask_width",
                                              cfg.get("soss_outer_mask_width", 70)),
            nirspec_mask_width     = params.get("nirspec_mask_width",
                                              cfg.get("nirspec_mask_width", 16)),
            miri_drop_groups       = params.get("miri_drop_groups",
                                              cfg.get("miri_drop_groups", 12)),
            jump_threshold         = params.get("jump_threshold",
                                              cfg.get("jump_threshold", 15)),
            time_jump_threshold    = params.get("time_jump_threshold",
                                              cfg.get("time_jump_threshold", 10)),
            flag_up_ramp           = cfg.get("flag_up_ramp", False),
            flag_in_time           = cfg.get("flag_in_time", True),
        )
        dm1 = results1[0]

        # ---- Stage 2 ----
        results2, centroids = run_stage2(
            results1,
            mode=mode,
            baseline_ints=cfg["baseline_ints"],
            save_results=False,
            force_redo=False,
            skip_steps=[],
            space_thresh         = params.get("space_outlier_threshold",
                                              cfg.get("space_outlier_threshold", 15)),
            time_thresh          = params.get("time_outlier_threshold",
                                              cfg.get("time_outlier_threshold", 10)),
            pca_components       = params.get("pca_components",
                                              cfg.get("pca_components", 10)),
            miri_trace_width     = params.get("miri_trace_width",
                                              cfg.get("miri_trace_width", 20)),
            miri_background_width= params.get("miri_background_width",
                                              cfg.get("miri_background_width", 14)),
        )

        # ---- Stage 3 ----
        dm3 = run_stage3(
            results2,
            save_results=False,
            force_redo=False,
            extract_method=cfg.get("extract_method", "box"),
            extract_width =params.get("extract_width",
                                      cfg.get("extract_width", 30)),
            centroids=centroids
        )

        dt = time.perf_counter() - t0
        J  = cost_function(dm1, params["w1"], params["w2"])
        return J, dt

    # ─── Sweep with coordinate descent ────────────────────────────────────
    with open(args.log, "w") as logfile:
        logfile.write("\t".join(param_order + ["duration_s", "J"]) + "\n")
        best_J = None

        for i, key in enumerate(param_order, start=1):
            print(f"\n→ Optimizing {key} ({i}/{len(param_order)}) …")
            best_val = current[key]

            for trial in param_ranges[key]:
                trial_params         = current.copy()
                trial_params[key]    = trial
                trial_params["w1"]   = args.w1
                trial_params["w2"]   = args.w2

                J, dt = evaluate_one(trial_params)
                print(f"   {key}={trial:<3} → J={J:.6f}  ({dt:.1f}s) ")

                logfile.write(
                    "\t".join(str(trial_params.get(k, "")) for k in param_order)
                    + f"\t{dt:.1f}\t{J:.6f}\n"
                )

                if best_J is None or J < best_J:
                    best_J  = J
                    best_val= trial

            current[key] = best_val
            print(f"✔ Best {key} = {best_val}  (J={best_J:.6f})")

        logfile.write("\n# Final optimized parameters:\n")
        for k in param_order:
            logfile.write(f"# {k} = {current[k]}\n")
        logfile.write(f"# Final cost J = {best_J:.6f}\n")

if __name__ == "__main__":
    main()
