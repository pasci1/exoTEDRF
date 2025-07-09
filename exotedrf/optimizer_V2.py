#!/usr/bin/env python3

import time
import numpy as np
import argparse
from astropy.stats import mad_std
from scipy.signal import detrend

from exotedrf.utils    import parse_config
from exotedrf.stage1   import run_stage1
from exotedrf.stage2   import run_stage2
from exotedrf.stage3   import run_stage3

# ———————— 1) Cost function ——————————
def compute_white_light_from_stage3(spectra):
    """
    spectra: dict returned by run_stage3, containing at least
             'flux_o1' and/or 'flux_o2' arrays of shape (nints, nwave).
    """
    # sum over wavelength for each order
    wl_orders = []
    for key in ('flux_o1', 'flux_o2'):
        if key in spectra:
            # sum over columns → white light per integration
            wl_orders.append(np.nansum(spectra[key], axis=1))
    wl = np.sum(wl_orders, axis=0)

    # detrend and robust scatter
    scat = mad_std(detrend(wl)) / np.abs(np.median(wl))
    return scat

def cost_function(stage1_res, stage2_res, stage3_res):
    # we only use the Stage 3 white-light scatter
    return compute_white_light_from_stage3(stage3_res)


# ———————— 2) Main optimizer ——————————
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="run_DMS.yaml",
                   help="Path to your DMS config")
    p.add_argument("--instrument", required=True,
                   choices=["NIRISS","NIRSPEC","MIRI"],
                   help="Which JWST instrument")
    args = p.parse_args()

    # load config
    cfg = parse_config(args.config)

    # set up parameter grid
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

    # initialize to medians
    current = {k: int(np.median(v)) for k, v in param_ranges.items()}

    # prepare log
    total = sum(len(v) for v in param_ranges.values())
    count = 1
    logfile = open("Cost_function_V2.txt", "w")
    logfile.write("\t".join(param_order) + "\tduration_s\tcost\n")

    # coordinate‐descent
    for key in param_order:
        best = (None, None)  # (best_cost, best_val)
        for trial in param_ranges[key]:
            # build kwargs for each stage
            trial_params = current.copy()
            trial_params[key] = trial

            # Stage 1
            t0 = time.perf_counter()
            st1 = run_stage1(
                cfg['input_files'],
                mode=cfg['observing_mode'],
                rejection_threshold=trial_params.get("jump_threshold"),
                time_rejection_threshold=trial_params.get("time_jump_threshold"),
                nirspec_mask_width=trial_params.get("nirspec_mask_width", None),
                soss_inner_mask_width=trial_params.get("soss_inner_mask_width", None),
                soss_outer_mask_width=trial_params.get("soss_outer_mask_width", None),
                miri_drop_groups=trial_params.get("miri_drop_groups", None),
                save_results=False,
                skip_steps=[],
                **cfg.get('stage1_kwargs',{})
            )  # :contentReference[oaicite:0]{index=0}

            # Stage 2
            st2, centroids = run_stage2(
                st1, mode=cfg['observing_mode'],
                space_thresh=trial_params["space_outlier_threshold"],
                time_thresh=trial_params["time_outlier_threshold"],
                pca_components=trial_params["pca_components"],
                soss_inner_mask_width=trial_params.get("soss_inner_mask_width", None),
                soss_outer_mask_width=trial_params.get("soss_outer_mask_width", None),
                nirspec_mask_width=trial_params.get("nirspec_mask_width", None),
                miri_trace_width=trial_params.get("miri_trace_width", None),
                miri_background_width=trial_params.get("miri_background_width", None),
                save_results=False,
                skip_steps=[],
                **cfg.get('stage2_kwargs',{})
            )  # :contentReference[oaicite:1]{index=1}

            # Stage 3
            st3 = run_stage3(
                st2, centroids=centroids,
                extract_width=trial_params["extract_width"],
                save_results=False,
                skip_steps=[],
                **cfg.get('stage3_kwargs',{})
            )  # :contentReference[oaicite:2]{index=2}

            dt = time.perf_counter() - t0
            cost = cost_function(st1, st2, st3)

            # log
            vals = [str(trial_params[k]) for k in param_order]
            logfile.write("\t".join(vals + [f"{dt:.1f}", f"{cost:.6f}"]) + "\n")

            # check best
            if best[0] is None or cost < best[0]:
                best = (cost, trial)

            print(f"[{count}/{total}] {key}={trial} → cost={cost:.6f} ({dt:.1f}s)")
            count += 1

        # lock in best
        current[key] = best[1]
        print(f"✔ Best {key} = {best[1]} (cost={best[0]:.6f})\n")

    logfile.close()
    print("FINAL OPTIMUM:", current)
    print("Log saved to Cost_function_V2.txt")

if __name__ == "__main__":
    main()
