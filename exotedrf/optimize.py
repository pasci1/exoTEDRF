#!/usr/bin/env python

import os, time, itertools, yaml, argparse
import numpy as np
from astropy.stats import mad_std
from jwst import datamodels

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

def reduced_chi2(residuals, sigma=1.):
    """Assume unit‐variance errors or pass in per‐point sigma."""
    χ2 = np.sum((residuals / sigma)**2)
    ν  = residuals.size - 1
    return χ2 / ν

def cost_function(dm, w1, w2, w3):
    # 1) White light MAD
    wl = compute_white_light(dm)
    mad_wl = mad_std(wl)  # robust scatter
    
    # 2) Spectral MAD (average over channels)
    spec = compute_spectral(dm)  # shape (nints, nchannels)
    mad_spec = np.mean([mad_std(spec[:,i]) for i in range(spec.shape[1])])
    
    # 3) χ²ν around a flat model (mean)
    χν2 = reduced_chi2(wl - np.mean(wl))
    
    return w1*mad_wl + w2*mad_spec + w3*χν2

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
# 3) Main & coordinate‐descent optimization with logging
# ----------------------------------------
def main():
    import argparse, time, numpy as np
    from jwst import datamodels
    from exotedrf.utils import parse_config
    from exotedrf.stage1 import run_stage1

    # 1) parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="run_WASP39b.yaml")
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=1.0)
    p.add_argument("--w3", type=float, default=1.0)
    args = p.parse_args()

    # 2) load config + 50‐int slice
    cfg = parse_config(args.config)
    seg1    = cfg['input_dir'] + "/jw01366003001_04101_00001-seg001_nrs1_uncal.fits"
    dm_full = datamodels.open(seg1)
    K       = min(50, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.nints = K
    dm_full.close()

    # 3) define your parameter ranges
    param_ranges = {
        'time_window':              [5, 7],
        'box_size':                 [10, 12],
        'thresh':                   [14,15],
        'rejection_threshold':      [10,12],
        'time_rejection_threshold': [10],
        'nirspec_mask_width':       [16],
    }
    param_order = [
        'time_window',
        'box_size',
        'thresh',
        'rejection_threshold',
        'time_rejection_threshold',
        'nirspec_mask_width',
    ]

    # 4) initialize with median/default values
    current = {p: int(np.median(param_ranges[p])) for p in param_order}
    current.update(w1=args.w1, w2=args.w2, w3=args.w3)

    skip_steps = ['OneOverFStep', 'JumpStep']

    def evaluate_one(params):
        # run_stage1 kwargs
        kwargs = dict(
            rejection_threshold      = params['rejection_threshold'],
            time_rejection_threshold = params['time_rejection_threshold'],
            nirspec_mask_width       = params['nirspec_mask_width'],
            **cfg['stage1_kwargs']
        )
        step_kwargs = {
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
            **kwargs,
            **step_kwargs
        )
        dt = time.perf_counter() - t0
        dm_out = result[0]
        J = cost_function(dm_out, params['w1'], params['w2'], params['w3'])
        return J, dt

    # 5) open log file and write header
    logfile = open("Cost_function.txt", "w")
    logfile.write("time_window,box_size,thresh,rejection_threshold,"
                  "time_rejection_threshold,nirspec_mask_width,duration_s,J\n")

    # 6) coordinate‐descent: optimize each parameter in turn
    best = (np.inf, None, None)  # (J, params, dt)
    for key in param_order:
        print(f"\n→ Optimizing {key} (others fixed at "
              f"{ {k:current[k] for k in current if k != key} })")
        best_val = current[key]
        best_J, best_dt = None, None

        for trial in param_ranges[key]:
            trial_params = current.copy()
            trial_params[key] = trial
            J, dt = evaluate_one(trial_params)

            # log to file
            logfile.write(
                f"{trial_params['time_window']},"
                f"{trial_params['box_size']},"
                f"{trial_params['thresh']},"
                f"{trial_params['rejection_threshold']},"
                f"{trial_params['time_rejection_threshold']},"
                f"{trial_params['nirspec_mask_width']},"
                f"{dt:.3f},{J:.6g}\n"
            )

            print(f"   {key}={trial} → J={J:.3g}, dt={dt:.1f}s")
            if best_J is None or J < best_J:
                best_J, best_dt, best_val = J, dt, trial

        current[key] = best_val
        best = (best_J, dict(current), best_dt)
        print(f"✔→ Best {key} = {best_val} (J={best_J:.3g}, dt={best_dt:.1f}s)")

    # 7) final summary
    print("\n=== FINAL OPTIMUM ===")
    print("params =", {k: current[k] for k in param_order})
    print("J =", best[0])
    print("last dt =", best[2])

    # 8) close log file
    logfile.close()

if __name__ == "__main__":
    main()