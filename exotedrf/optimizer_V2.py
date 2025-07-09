#!/usr/bin/env python3

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
def compute_white_light_scatter_from_stage3(spectra):
    """
    Sum over all spectral orders and pixels to make a 1D white-light curve,
    detrend it, and return MAD-based fractional scatter.
    """
    flux_keys = [k for k in spectra.keys() if k.startswith('flux')]
    if not flux_keys:
        raise RuntimeError("No 'flux_*' arrays found in Stage 3 output!")

    # sum over wavelength → WL per order
    wl_orders = [np.nansum(spectra[k], axis=1) for k in flux_keys]
    # sum across orders → single WL curve
    wl = np.sum(wl_orders, axis=0)

    # detrend & compute robust scatter / median
    return mad_std(detrend(wl)) / np.abs(np.median(wl))

def cost_function(stage3_res):
    """Single cost: the fractional scatter of the extracted white-light curve."""
    return compute_white_light_scatter_from_stage3(stage3_res)


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

    # load config & discover segment files
    cfg = parse_config(args.config)
    input_files = unpack_input_dir(
        cfg['input_dir'],
        mode            = cfg['observing_mode'],
        filetag         = cfg['input_filetag'],
        filter_detector = cfg['filter_detector']
    )
    fancyprint(f"Found {len(input_files)} segments for "
               f"{cfg['filter_detector']} / {cfg['observing_mode']}")

    # build parameter grid
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

    # fixed sweep order
    param_order = list(param_ranges.keys())

    # start from median of each range
    current = {k: int(np.median(v)) for k, v in param_ranges.items()}

    # prepare logfile
    total_steps = sum(len(v) for v in param_ranges.values())
    count = 1
    logf = open("Cost_function_V2.txt", "w")
    logf.write("\t".join(param_order) + "\tduration_s\tcost\n")

    # coordinate-descent
    for key in param_order:
        best_cost, best_val = None, current[key]
        fancyprint(f"→ Optimizing {key} (others fixed at "
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
                space_thresh           = trial_params["space_outlier_threshold"],
                time_thresh            = trial_params["time_outlier_threshold"],
                pca_components         = trial_params["pca_components"],
                soss_inner_mask_width  = trial_params.get("soss_inner_mask_width"),
                soss_outer_mask_width  = trial_params.get("soss_outer_mask_width"),
                nirspec_mask_width     = trial_params.get("nirspec_mask_width"),
                miri_trace_width       = trial_params.get("miri_trace_width"),
                miri_background_width  = trial_params.get("miri_background_width"),
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

            # log
            vals = [str(trial_params[k]) for k in param_order]
            logf.write("\t".join(vals + [f"{dt:.1f}", f"{cost:.6f}"]) + "\n")

            # update best
            if best_cost is None or cost < best_cost:
                best_cost, best_val = cost, trial

            print(f"[{count}/{total_steps}] {key}={trial} → "
                  f"cost={cost:.6f} ({dt:.1f}s)")
            count += 1

        # lock in best for this key
        current[key] = best_val
        fancyprint(f"✔ Best {key} = {best_val} (cost={best_cost:.6f})")

    logf.close()
    fancyprint("=== FINAL OPTIMUM ===")
    fancyprint(current)
    fancyprint("Log saved to Cost_function_V2.txt")


if __name__ == "__main__":
    main()
