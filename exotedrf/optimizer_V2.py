#!/usr/bin/env python3
"""
optimizer_V2.py: Coordinate‐descent optimizer for exoTEDRF pipeline parameters across Stage 1, 2, and 3.
Sweeps only parameters relevant to the instrument (NIRISS/SOSS, NIRSpec, or MIRI).
"""
import time
import argparse
import numpy as np
from jwst import datamodels
from exotedrf.utils import parse_config
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3
from astropy.stats import mad_std
from scipy.signal import detrend

# ----------------------------------------
# Cost function (white-light robust scatter)
# ----------------------------------------
def compute_white_light_from_dm(dm):
    """Sum over all pixels to extract white-light curve."""
    wl = dm.data.reshape(dm.data.shape[0], -1).sum(axis=1)
    return wl

def cost_function(dm, w1=1.0, w2=1.0):
    """Robust scatter of detrended white-light curve."""
    wl = compute_white_light_from_dm(dm)
    frac = mad_std(wl) / np.abs(np.median(wl))
    return w1 * frac + w2 * frac  # placeholder weights

# ----------------------------------------
# Main optimizer
# ----------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="run_DMS.yaml")
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=1.0)
    args = p.parse_args()

    # Parse pipeline config
    cfg = parse_config(args.config)
    mode = cfg['observing_mode']  # e.g. 'NIRISS/SOSS', 'NIRSpec/G395H', 'MIRI/LRS'
    instrument = mode.split('/')[0].upper()

    # Prepare a short slice for speed
    seg = cfg['input_dir'] + "/" + cfg['input_dir'].split('/')[-1] + "_seg001_uncal.fits"
    dm_full = datamodels.open(seg)
    K = min(60, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end = K
    dm_slice.meta.exposure.nints = K
    dm_full.close()

    # Define parameter ranges based on instrument
    param_ranges = {}
    # Stage 1
    if instrument == 'NIRISS':
        param_ranges['soss_inner_mask_width'] = [20, 40, 80]
        param_ranges['soss_outer_mask_width'] = [35, 70, 140]
        param_ranges['jump_threshold']         = [5, 15, 30]
        param_ranges['time_jump_threshold']    = [3, 10, 20]
    elif instrument == 'NIRSPEC':
        param_ranges['nirspec_mask_width']     = [8, 16, 32]
        param_ranges['jump_threshold']         = [5, 15, 30]
        param_ranges['time_jump_threshold']    = [3, 10, 20]
    elif instrument == 'MIRI':
        param_ranges['miri_drop_groups']       = [6, 12, 24]
        param_ranges['jump_threshold']         = [5, 15, 30]
        param_ranges['time_jump_threshold']    = [3, 10, 20]

    # Stage 2
    if instrument == 'MIRI':
        param_ranges['miri_trace_width']       = [10, 20, 40]
        param_ranges['miri_background_width']  = [7, 14, 28]
    param_ranges['space_outlier_threshold']   = [5, 15, 30]
    param_ranges['time_outlier_threshold']    = [3, 10, 20]
    param_ranges['pca_components']            = [5, 10, 20]

    # Stage 3
    param_ranges['extract_width']             = [15, 30, 60]

    param_order = list(param_ranges.keys())
    total_steps = sum(len(v) for v in param_ranges.values())

    # Initialize current best at defaults from config
    current = {}
    for p in param_order:
        current[p] = cfg.get(p, int(np.median(param_ranges[p])))
    current.update(w1=args.w1, w2=args.w2)

    def evaluate_one(params):
        print("Running with params:", params)
        t0 = time.perf_counter()
        # --- Stage 1 ---
        stage1_kwargs = {
            'mode': mode,
            'baseline_ints': cfg['baseline_ints'],
            'save_results': False,
            'skip_steps': [],
            'soss_inner_mask_width': params.get('soss_inner_mask_width', cfg['soss_inner_mask_width']),
            'soss_outer_mask_width': params.get('soss_outer_mask_width', cfg['soss_outer_mask_width']),
            'nirspec_mask_width':       params.get('nirspec_mask_width', cfg['nirspec_mask_width']),
            'miri_drop_groups':         params.get('miri_drop_groups', cfg['miri_drop_groups']),
            'rejection_threshold':      cfg['jump_threshold'],
            'time_rejection_threshold': cfg['time_jump_threshold'],
            'flag_up_ramp': cfg['flag_up_ramp'],
            'flag_in_time': cfg['flag_in_time'],
            'jump_threshold': params.get('jump_threshold', cfg['jump_threshold']),
            'time_jump_threshold': params.get('time_jump_threshold', cfg['time_jump_threshold']),
        }
        results1 = run_stage1([dm_slice], **stage1_kwargs)
        dm1 = results1[0]

        # --- Stage 2 ---
        stage2_kwargs = {
            'mode': mode,
            'baseline_ints': cfg['baseline_ints'],
            'save_results': False,
            'force_redo': False,
            'space_thresh': params.get('space_outlier_threshold', cfg['space_outlier_threshold']),
            'time_thresh':  params.get('time_outlier_threshold', cfg['time_outlier_threshold']),
            'pca_components': params.get('pca_components', cfg['pca_components']),
            'miri_trace_width':      params.get('miri_trace_width', cfg['miri_trace_width']),
            'miri_background_width': params.get('miri_background_width', cfg['miri_background_width']),
            'skip_steps': []
        }
        results2, centroids = run_stage2(results1, **stage2_kwargs)

        # --- Stage 3 ---
        stage3_kwargs = {
            'save_results': False,
            'force_redo': False,
            'extract_method': cfg['extract_method'],
            'extract_width': params.get('extract_width', cfg['extract_width']),
            'centroids': centroids,
        }
        dm3 = run_stage3(results2, **stage3_kwargs)

        # Compute cost based on Stage 1 output (dm1)
        J = cost_function(dm1, params['w1'], params['w2'])
        dt = time.perf_counter() - t0
        return J, dt

    # Coordinate descent
    count = 1
    best_J = None
    for key in param_order:
        print(f"\n→ Optimizing {key} ({count}/{total_steps})")
        best_val = current[key]
        for trial in param_ranges[key]:
            trial_params = current.copy()
            trial_params[key] = trial
            J, dt = evaluate_one(trial_params)
            print(f"   {key}={trial} → J={J:.6f} ({dt:.1f}s)")
            if best_J is None or J < best_J:
                best_J, best_val = J, trial
        current[key] = best_val
        print(f"✔→ Best {key} = {best_val} (J={best_J:.6f})")
        count += 1

    print("\n=== FINAL OPTIMUM PARAMETERS ===")
    for k in param_order:
        print(f"{k} = {current[k]}")
    print(f"Final cost J = {best_J:.6f}")

if __name__ == '__main__':
    main()
