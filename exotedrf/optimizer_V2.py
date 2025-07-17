#!/usr/bin/env python3

import os

os.environ.setdefault('CRDS_PATH', os.path.join(os.getcwd(),'Optimize_WASP39b','crds_cache'))
os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
os.environ.setdefault('CRDS_CONTEXT',    'jwst_1322.pmap')


import glob
import time
import argparse
import numpy as np
import pandas as pd
from jwst import datamodels
import matplotlib.pyplot as plt

from exotedrf.utils import parse_config, unpack_input_dir, fancyprint
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3

# ----------------------------------------
# cost (MAD-based)
# ----------------------------------------

def cost_function(st3):
    """
    Combined cost = w1 * norm_MAD_white + w2 * norm_MAD_spec
      - norm_MAD_white = MAD_white / |median_white|
      - norm_MAD_spec  = MAD_spec  / |median_spectral|
    """
    w1 = 0.5
    w2 = 0.5
    flux = np.asarray(st3['Flux'], dtype=float)  # shape (n_int, n_wave)

    # 1) White-light MAD
    white = np.nansum(flux, axis=1)               # shape (n_int,)
    white = white[~np.isnan(white)]               # drop NaNs
    med_white = np.median(white)
    mad_white = np.median(np.abs(white - med_white))
    norm_mad_white = mad_white / np.abs(med_white)

    # 2) Spectral MAD (per-integration)
    med_spec = np.nanmedian(flux, axis=1, keepdims=True)  # (n_int, 1)
    dev_spec = np.abs(flux - med_spec)
    mad_spec_per_int = np.nanmedian(dev_spec, axis=1)     # (n_int,)
    med_spec_vals = np.nanmedian(flux, axis=1)            # (n_int,)
    norm_mad_spec_per_int = mad_spec_per_int / np.abs(med_spec_vals)
    norm_mad_spec = np.nanmedian(norm_mad_spec_per_int)   # scalar

    # Combined cost
    return w1 * norm_mad_white + w2 * norm_mad_spec



# ----------------------------------------
# main
# ----------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Coordinate‐descent optimizer for exoTEDRF Stages 1–3"
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


    t0_total = time.perf_counter()
    cfg = parse_config(args.config)


    # get input data
    input_files = unpack_input_dir(
        cfg["input_dir"],
        mode=cfg["observing_mode"],
        filetag=cfg["input_filetag"],
        filter_detector=cfg["filter_detector"],
    )
    if not input_files:
        fancyprint(f"[WARN] No files in {cfg['input_dir']}, globbing *.fits")
        input_files = sorted(glob.glob(os.path.join(cfg["input_dir"], "*.fits")))
    if not input_files:
        raise RuntimeError(f"No FITS found in {cfg['input_dir']}")
    fancyprint(f"Using {len(input_files)} segment(s) from {cfg['input_dir']}")

    # Slice First segment
    #seg0 = os.path.join(cfg["input_dir"], input_files[0].split(os.sep)[-1])

    seg0 = os.path.join(
        cfg['input_dir'],
        "jw01366003001_04101_00001-seg001_nrs1_uncal.fits"
    )

    # determine integration number (slice) K=
    dm_full = datamodels.open(seg0)
    K = min(60, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end = K
    dm_slice.meta.exposure.nints = K
    dm_full.close()


    # Parameter to SWEEP
    param_ranges = {}
    if args.instrument == "NIRISS":
        param_ranges.update({
            "soss_inner_mask_width": [20, 40, 80],
            "soss_outer_mask_width": [35, 70, 140],
            "jump_threshold": [5, 15, 30],
            "time_jump_threshold": [3, 10, 20],
        })
    elif args.instrument == "NIRSPEC":
        param_ranges.update({
            'time_window':              list(range(1,7,2)), # works
            #'rejection_threshold':     list(range(4,9,2)), # works for Flag_up_ramp = True
            'time_rejection_threshold': list(range(14,17,1)), # works           
            "nirspec_mask_width":       list(range(16,19,1)), # works
        })
    else:  # MIRI
        param_ranges.update({
            "miri_drop_groups": [6, 12, 24],
            "jump_threshold": [5, 15, 30],
            "time_jump_threshold": [3, 10, 20],
            "miri_trace_width": [10, 20, 40], 
            "miri_background_width": [7, 14, 28],
        })
    # always sweep 
    param_ranges.update({

        "space_thresh": list(range(6,9,1)),
        "time_thresh":  list(range(1,7,3)),
        "box_size":     list(range(1,7,3)),  # semi
        "window_size":  list(range(1,7,3)),  # semi
        "extract_width": list(range(1,6,3 )),

    })


    param_order = list(param_ranges.keys())
    current = {k: int(np.median(v)) for k, v in param_ranges.items()}
    total_steps = sum(len(v) for v in param_ranges.values())

    # Logging
    logf = open("Cost_function_V2.txt", "w")
    logf.write("\t".join(param_order) + "\tduration_s\tcost\n")

    stage1_keys = [
        "rejection_threshold", "time_rejection_threshold",
        "nirspec_mask_width", "soss_inner_mask_width",
        "soss_outer_mask_width", "miri_drop_groups",
    ]
    
    stage2_keys = [
        "space_thresh", "time_thresh",      # ← your new BadPixStep args
        #"box_size", "window_size",          # ← if you end up sweeping these too
        "time_window",                      # ← stays for stage1/JumpStep
        "miri_trace_width", "miri_background_width",
    ]


    stage3_keys = ["extract_width"]

    count = 1
    # coordinate descent
    for key in param_order:
        fancyprint(
            f"→ Optimizing {key} "
            f"(fixed-other={{{', '.join([f'{k}:{current[k]}' for k in current if k != key])}}})"
        )
        best_cost, best_val = None, current[key]
        for trial in param_ranges[key]:
            fancyprint(f"Step {count}/{total_steps}: {key}={trial}")
            trial_params = {**current, key: trial}

            nints = dm_slice.data.shape[0]
            print("\033[1;91mnints is:\033[0m", nints)
            baseline_ints = [10,-10]

            t0 = time.perf_counter()
            s1_args = {k: trial_params[k] for k in stage1_keys if k in trial_params}
            s2_args = {k: trial_params[k] for k in stage2_keys if k in trial_params}
            s3_args = {k: trial_params[k] for k in stage3_keys if k in trial_params}

            # Inherit Parameters to sweep
            if "time_window" in trial_params:
                s1_args["JumpStep"] = {"time_window": trial_params["time_window"]}
            
            badpix = {}
            if "box_size"   in trial_params:
                badpix["box_size"]   = trial_params["box_size"]
            if "window_size" in trial_params:
                badpix["window_size"] = trial_params["window_size"]
            if badpix:
                s2_args["BadPixStep"] = badpix



            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} starting {key}={trial}",
                "\n############################################\n",
                flush=True
            )

            print("\033[1;91ms1_args is:\033[0m", s1_args)

            st1 = run_stage1( 
                [dm_slice],
                mode=cfg["observing_mode"],
                baseline_ints=baseline_ints,
                flag_up_ramp=False, # Problem
                save_results=False,
                skip_steps=[],
                **s1_args
            )

            print("\033[1;91ms2_args is:\033[0m", s2_args)
        


            st2, centroids = run_stage2(
                st1,
                mode=cfg["observing_mode"],
                baseline_ints=baseline_ints,
                save_results=False,
                skip_steps=[],
                **s2_args,
                **cfg.get("stage2_kwargs", {})
            ) 
            if isinstance(centroids, np.ndarray):
                centroids = pd.DataFrame(centroids.T, columns=["xpos", "ypos"])
            st3 = run_stage3(
                st2,
                centroids=centroids,
                save_results=False,
                skip_steps=[],
                **s3_args,
                **cfg.get("stage3_kwargs", {})
            )

            # Flux‐Array 
            model = st3
            if isinstance(model, dict):
                model = next(iter(model.values()))
            if hasattr(model, "data"):
                model = model.data
            arr = np.asarray(model)

            dt = time.perf_counter() - t0
            cost = cost_function(st3)
            fancyprint(f"→ cost = {cost:.12f} in {dt:.1f}s")

            logf.write(
                "\t".join(str(trial_params[k]) for k in param_order)
                + f"\t{dt:.1f}\t{cost:.12f}\n"
            )
            if best_cost is None or cost < best_cost:
                best_cost, best_val = cost, trial
            

            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} completed (dt={dt:.1f}s)",
                "\n############################################\n",
                flush=True
            )            

            print("\033[1m\033[94m========== Cost ==========\033[0m",'\n', cost,'\n')

            count += 1

        current[key] = best_val
        fancyprint(f"Best {key} = {best_val} (cost={best_cost:.12f})")



    # total runtime
    t1 = time.perf_counter() - t0_total
    h, m = divmod(int(t1), 3600)
    m, s = divmod(m, 60)
    runtime_str = f"TOTAL runtime: {h}h {m:02d}min {s:04.1f}s\n"
    fancyprint(runtime_str)
    logf.write(runtime_str)

    logf.close()
    fancyprint("=== FINAL OPTIMUM ===")
    fancyprint(current)
    fancyprint("Log saved to Cost_function_V2.txt")

if __name__ == "__main__":
    main()


