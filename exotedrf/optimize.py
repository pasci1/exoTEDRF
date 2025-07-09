#!/usr/bin/env python

import os, time, itertools, yaml, argparse
import numpy as np
from astropy.stats import mad_std
from jwst import datamodels
from scipy.signal import detrend
from astropy.stats import mad_std

from exotedrf.utils    import parse_config
from exotedrf.stage1   import run_stage1

# ----------------------------------------
# 1) Cost‐function definitions
# ----------------------------------------
def compute_white_light(dm):
    """Extract the white‐light curve (sum over all pixels)."""
    # dm.data after RampFitStep is a CubeModel: shape (nints, dimy, dimx)
    wl = dm.data.reshape(dm.data.shape[0], -1).sum(axis=1)
    return wl

def compute_spectral(dm):
    """Extract a toy spectral lightcurve: e.g. sum down columns."""
    spec = dm.data.reshape(dm.data.shape[0], dm.data.shape[1], -1)
    return spec.mean(axis=2)  # shape (nints, dimy)

def reduced_chi2(residuals, sigma=1.0):
    """Compute reduced chi-squared assuming errors = sigma."""
    Chi_2 = np.sum((residuals / sigma)**2)
    nu = residuals.size - 1  # degrees of freedom
    return Chi_2 / nu

"""
def cost_function(dm, w1, w2, w3):
    # 1) White-light stddev after detrending
    wl = compute_white_light(dm)
    std_wl = np.std(detrend(wl))

    # 2) Spectral stddev after detrending, averaged over all rows
    spec = compute_spectral(dm)
    std_spec = np.mean([np.std(detrend(spec[:, i])) for i in range(spec.shape[1])])

    # 3) Reduced chi-squared around a flat (mean) model
    Chi2 = reduced_chi2(wl - np.mean(wl))

    return w1 * std_wl + w2 * std_spec + w3 * Chi2

"""
    


def cost_function(dm, w1=1.0, w2=1.0, w3=1.0):
    # — Extract white-light & spectral curves —
    wl   = compute_white_light(dm)             # shape (nints,)
    spec = compute_spectral(dm)                # shape (nints, nrows)
 
    # — 1) Robust fractional scatter of white-light curve —
    #    MAD is ~1.4826*median(|x − median(x)|), but mad_std wraps that.
    frac_wl = mad_std(wl) / np.abs(np.median(wl))

    # — 2) Robust fractional scatter averaged over spectral rows —
    frac_spec_rows = [
        mad_std(spec[:, i]) / np.abs(np.median(spec[:, i]))
        for i in range(spec.shape[1])
    ]
    frac_spec = np.mean(frac_spec_rows)

    # — Combine with weights —
    return w1 * frac_wl + w2 * frac_spec


# ----------------------------------------
# 2) Main & coordinate‐descent
# ----------------------------------------
def main():
    import argparse
    import time
    import numpy as np
    from itertools import product
    from jwst import datamodels
    from exotedrf.utils import parse_config
    from exotedrf.stage1 import run_stage1



    # 1) parse args & config
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="run_WASP39b.yaml")
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=1.0)
    p.add_argument("--w3", type=float, default=1.0)
    args = p.parse_args()
    cfg = parse_config(args.config)

    # ─── START GLOBAL TIMER ─────────────────────────────────────────────
    t0_total = time.perf_counter()
    # ────────────────────────────────────────────────────────────────────

    # 2) load K‐int slice K= K =
    seg1 = cfg['input_dir'] + "/jw01366003001_04101_00001-seg001_nrs1_uncal.fits"
    dm_full = datamodels.open(seg1)
    K = min(60, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    #
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end   = K
    #
    dm_slice.meta.exposure.nints = K
    dm_full.close()

    # 3) parameter ranges & order SWEEP OVER THESE PARAMETERS
    param_ranges = {
        'time_window':              [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,71,131],
        'box_size':                 [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,70,150,200],
        'thresh':                   [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,70,150,200],
        'rejection_threshold':      [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,70,150,200],
        'time_rejection_threshold': [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,70,150,200],
        'nirspec_mask_width':       [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,70,150,200],
    }

    
    # fast Check Params
    
    param_ranges = {
        'time_window':              [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43],
        'box_size':                 [10],
        'thresh':                   [10],
        'rejection_threshold':      [10], 
        'time_rejection_threshold': [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100],
        'nirspec_mask_width':       [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100],
    }


    

    param_order = [
        'time_window',
        'box_size',
        'thresh',
        'rejection_threshold',
        'time_rejection_threshold',
        'nirspec_mask_width',
    ]




    # counter for status update
    count = 1
    total_steps = sum(len(v) for v in param_ranges.values())

    # 4) initialize current best at medians
    # this just makes sure to start with a median for the rest parameters 
    current = {p: int(np.median(param_ranges[p])) for p in param_order}
    current.update(w1=args.w1, w2=args.w2, w3=args.w3)

    skip_steps = []  
 
    def evaluate_one(params):
        print("Running with params:", params)

        # ─── Build the kwargs that run_stage1 actually knows ────────────────────
        run_kwargs = {
            # Up-the-ramp sigma threshold
            'rejection_threshold':      params['thresh'],
            # Time-domain sigma threshold
            'time_rejection_threshold': params['time_rejection_threshold'],
            # NIRSpec mask width for the OneOverFStep
            'nirspec_mask_width':       params['nirspec_mask_width'],
            # Everything else (time_window) goes under the JumpStep sub-dict:
            'JumpStep': {
                'time_window': params['time_window']
            }
        }

        # ─── Time it & run Stage 1 ─────────────────────────────────────────────
        t0 = time.perf_counter()
        baseline_ints = list(range(dm_slice.data.shape[0]))  # [0…K-1]

        results = run_stage1(
            [dm_slice],
            mode=cfg['observing_mode'],
            baseline_ints=baseline_ints,
            save_results=False,
            skip_steps=skip_steps,
            **run_kwargs
        )
        dt = time.perf_counter() - t0

        # ─── Score it ─────────────────────────────────────────────────────────
        dm_out = results[0]
        J = cost_function(dm_out, params['w1'], params['w2'], params['w3'])
        return J, dt


    # 5) open log file (TSV) and write header
    logfile = open("Cost_function.txt", "w")
    logfile.write(
        "time_window\t"
        "box_size\t"
        "thresh\t"
        "rejection_threshold\t"
        "time_rejection_threshold\t"
        "nirspec_mask_width\t"
        "duration_s\t"
        "J\n"
    )

    # 6) coordinate‐descent
    for key in param_order:
        print(f"\n→ Optimizing {key} (others fixed = "
              f"{ {k:current[k] for k in current if k!=key} })")
        best_J = None
        best_val = current[key]
        best_dt = None

        for trial in param_ranges[key]:
            trial_params = current.copy()
            trial_params[key] = trial
            J,  dt = evaluate_one(trial_params)

            print(f"\n\n\n###########################################################\n Step: {count}/{total_steps} completed\n###########################################################\n\n\n")
            count +=1 
            # log this trial
            logfile.write(
                f"{trial_params['time_window']}\t"
                f"{trial_params['box_size']}\t"
                f"{trial_params['thresh']}\t"
                f"{trial_params['rejection_threshold']}\t"
                f"{trial_params['time_rejection_threshold']}\t"
                f"{trial_params['nirspec_mask_width']}\t"
                f"{dt:.1f}\t"
                f"{J:.6f}\n"
            )

            print(f"   {key}={trial} → J={J:.6f} ({dt:.1f}s)")

            if best_J is None or J < best_J:
                best_J, best_val, best_dt = J, trial, dt

        current[key] = best_val
        print(f"✔→ Best {key} = {best_val} (J={best_J:.6f}, dt={best_dt:.1f}s)")

    logfile.close()

    # 7) final report
    print("\n=== FINAL OPTIMUM ===")
    print("params =", {k: current[k] for k in param_order})
    print("J =", best_J)
    print("last dt =", best_dt)

    # ─── STOP GLOBAL TIMER & PRINT TOTAL ────────────────────────────────
    t1_total = time.perf_counter()
    total = t1_total - t0_total
    h = int(total) // 3600
    m = (int(total) % 3600) // 60
    s = total % 60
    print(f"TOTAL optimization runtime: {h}h {m:02d}min {s:04.1f}s")
    # ────────────────────────────────────────────────────────────────────

    # 8) write final optimum to logfile
    logfile = open("Cost_function.txt", "a")  # reopen in append mode
    logfile.write("\n# Final optimized parameters:\n")
    for k in param_order:
        logfile.write(f"# {k} = {current[k]}\n")
    logfile.write(f"# Final cost J = {best_J:.6f}\n")
    logfile.close()




if __name__ == "__main__":
    main()
