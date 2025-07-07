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
# 3) Main & grid‐search
# ----------------------------------------
def main():
    import argparse, yaml, time, numpy as np
    from itertools import product
    from jwst import datamodels
    from exotedrf.utils import parse_config
    from exotedrf.stage1 import run_stage1

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="run_WASP39b.yaml")
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=1.0)
    p.add_argument("--w3", type=float, default=1.0)
    args = p.parse_args()

    # 1) load config + 50‐int slice exactly as before
    cfg = parse_config(args.config)
    seg1    = cfg['input_dir'] + "/jw01366003001_04101_00001-seg001_nrs1_uncal.fits"
    dm_full = datamodels.open(seg1)
    K       = min(50, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.nints = K
    dm_full.close()

    # 2) define your parameter ranges
    param_ranges = {
        'time_window':              [3, 5, 7],
        'box_size':                 [5, 10, 15],
        'thresh':                   [10, 15, 20],
        'rejection_threshold':      [5, 10, 15],
        'time_rejection_threshold': [5, 10, 15],
        'nirspec_mask_width':       [14, 16, 18],
    }
    # the order you want to optimize in:
    param_order = [
        'time_window',
        'box_size',
        'thresh',
        'rejection_threshold',
        'time_rejection_threshold',
        'nirspec_mask_width',
    ]

    # 3) initialize a “current best” dict with the default (middle) values
    current = {p: np.median(param_ranges[p]).astype(int) for p in param_order}
    current.update(w1=args.w1, w2=args.w2, w3=args.w3)

    skip_steps = ['OneOverFStep', 'JumpStep']

    def evaluate_one(params):
        # run stage1 with just these params
        kwargs = dict(
            rejection_threshold = params['rejection_threshold'],
            time_rejection_threshold = params['time_rejection_threshold'],
            nirspec_mask_width = params['nirspec_mask_width'],
            **cfg['stage1_kwargs']
        )
        # pass time_window and box_size into jumpstep if you wired them in:
        step_kwargs = {'window': params['time_window'],
                       'box_size': params['box_size']}
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
        t1 = time.perf_counter()
        dm_out = result[0]
        J = cost_function(dm_out, params['w1'], params['w2'], params['w3'])
        return J, t1-t0

    # 4) coordinate‐descent: optimize each parameter in turn, holding the others fixed
    best_time = None
    for key in param_order:
        best_val = current[key]
        best_J, best_dt = None, None

        print(f"\n→ Optimizing {key} (fixing others at { {k:current[k] for k in current if k!=key} })")
        for trial in param_ranges[key]:
            trial_params = current.copy()
            trial_params[key] = trial

            J, dt = evaluate_one(trial_params)
            print(f"   {key}={trial} → J={J:.3g} ({dt:.1f}s)")

            if best_J is None or J < best_J:
                best_J, best_dt, best_val = J, dt, trial

        # lock in the winner
        current[key] = best_val
        print(f"✔→ Best {key} = {best_val} (J={best_J:.3g}, dt={best_dt:.1f}s)")

    # 5) final result
    print("\n=== FINAL OPTIMUM ===")
    print("params =", {k:current[k] for k in param_order})
    print("J =", best_J)
    print("last dt =", best_dt)


if __name__ == "__main__":
    main()
