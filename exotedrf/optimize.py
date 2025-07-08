#!/usr/bin/env python

import os, time, itertools, yaml, argparse
import numpy as np
from astropy.stats import mad_std
from jwst import datamodels
from scipy.signal import detrend

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

# ----------------------------------------
# 2) Stage1 evaluation
# ----------------------------------------
def evaluate(cfg, dm_slice, params, skip_steps):
    kwargs = dict(
        rejection_threshold = params['thresh'],
        time_rejection_threshold = params['time_rejection_threshold'],
        nirspec_mask_width = params['nirspec_mask_width'],
        **cfg['stage1_kwargs']
    )
    t0 = time.perf_counter()
    result = run_stage1(
        [dm_slice],
        mode=cfg['observing_mode'],
        baseline_ints=dm_slice.data.shape[0],
        save_results=False,
        skip_steps=skip_steps,
        **kwargs
    )
    dt = time.perf_counter() - t0

    # unwrap the output model
    dm_out = result if isinstance(result, CubeModel) else result[0]
    J = cost_function(dm_out, params['w1'], params['w2'], params['w3'])

    # free memory
    dm_out.close()
    return J, dt

# ----------------------------------------
# 3) Main & coordinate‐descent
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

    # 2) load K‐int slice
    seg1 = cfg['input_dir'] + "/jw01366003001_04101_00001-seg001_nrs1_uncal.fits"
    dm_full = datamodels.open(seg1)
    K = min(10, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.nints = K
    dm_full.close()

    # 3) parameter ranges & order SWEEP OVER THESE PARAMETERS
    param_ranges = {
        'time_window':              [5,7,9],
        'box_size':                 [5,10,15],
        'thresh':                   [12,15,18],
        'rejection_threshold':      [6,8,10,12],
        'time_rejection_threshold': [5,10,15],
        'nirspec_mask_width':       [10,15,20],
    }
    param_order = [
        'time_window',
        'box_size',
        'thresh',
        'rejection_threshold',
        'time_rejection_threshold',
        'nirspec_mask_width',
    ]

    # 4) initialize current best at medians
    current = {p: int(np.median(param_ranges[p])) for p in param_order}
    current.update(w1=args.w1, w2=args.w2, w3=args.w3)

    skip_steps = ['OneOverFStep'] 

    def evaluate_one(params):

        # DEBUG ???
        print("Running with params:", params)



        # build kwargs
        stage1_kwargs = dict(
            rejection_threshold        = params['rejection_threshold'],
            time_rejection_threshold   = params['time_rejection_threshold'],
            nirspec_mask_width         = params['nirspec_mask_width'],
            **cfg['stage1_kwargs']
        )
        jump_kwargs = {
            'window':   params['time_window'],
            'box_size': params['box_size'],
        }
        t0 = time.perf_counter()
        result = run_stage1(
            [dm_slice],
            mode=cfg['observing_mode'],
            baseline_ints=dm_slice.data.shape[0],
            save_results=False,
            skip_steps=skip_steps,
            **stage1_kwargs,
            **jump_kwargs
        )
        dt = time.perf_counter() - t0
        dm_out = result[0]
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

            # log this trial
            logfile.write(
                f"{trial_params['time_window']}\t"
                f"{trial_params['box_size']}\t"
                f"{trial_params['thresh']}\t"
                f"{trial_params['rejection_threshold']}\t"
                f"{trial_params['time_rejection_threshold']}\t"
                f"{trial_params['nirspec_mask_width']}\t"
                f"{dt:.1f}\t"
                f"{J:.0f}\n"
            )

            print(f"   {key}={trial} → J={J:.0f} ({dt:.1f}s)")

            if best_J is None or J < best_J:
                best_J, best_val, best_dt = J, trial, dt

        current[key] = best_val
        print(f"✔→ Best {key} = {best_val} (J={best_J:.0f}, dt={best_dt:.1f}s)")

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

if __name__ == "__main__":
    main()
