#!/usr/bin/env python3
"""
optimizer_V2.py: Coordinate‐descent optimizer for exoTEDRF parameters (Stage 1–3),
with automatic slice detection, absolute paths, and PRISM skip logic.
"""
import os
import sys
import time
import glob
import argparse
import numpy as np
from jwst import datamodels
from astropy.stats import mad_std

from exotedrf.utils import parse_config
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3


def compute_white_light(dm):
    """Sum over all pixels to get the white-light curve."""
    wl = dm.data.reshape(dm.data.shape[0], -1).sum(axis=1)
    return wl

def cost_function(dm, w1=1.0, w2=1.0):
    """Robust fractional scatter of the white-light curve."""
    wl = compute_white_light(dm)
    frac = mad_std(wl) / np.abs(np.median(wl))
    # you could mix in a spectral term here if you like, but for now:
    return w1 * frac + w2 * frac


def main():
    p = argparse.ArgumentParser(
        description="Optimize exoTEDRF pipeline parameters via coordinate descent"
    )
    p.add_argument(
        "--config", "-c",
        default="run_WASP39b.yaml",
        help="Path to your DMS YAML (e.g. Optimize_WASP39b/run_WASP39b.yaml)"
    )
    p.add_argument(
        "--log", "-l",
        default="Cost_function_V2.txt",
        help="TSV logfile for J and runtimes"
    )
    p.add_argument("--w1", type=float, default=1.0, help="Weight on term 1")
    p.add_argument("--w2", type=float, default=1.0, help="Weight on term 2")
    args = p.parse_args()

    # 1) absolute config + parse
    config_file = os.path.abspath(args.config)
    cfg_dir     = os.path.dirname(config_file)
    cfg         = parse_config(config_file)
    mode        = cfg['observing_mode']            # e.g. "NIRISS/SOSS"
    instrument  = mode.split('/')[0].upper()       # "NIRISS", "NIRSPEC", or "MIRI"

    # 2) find your 1-segment slice automatically
    input_dir = os.path.join(cfg_dir, cfg['input_dir'])
    pattern   = os.path.join(input_dir, "*seg001*_uncal.fits")
    matches   = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(f"No slice found with pattern {pattern!r}")
    seg = matches[0]
    print("→ Using slice:", seg)

    # 3) load & trim to K=60 ints for speed
    dm_full   = datamodels.open(seg)
    K         = min(60, dm_full.data.shape[0])
    dm_slice  = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end   = K
    dm_slice.meta.exposure.nints            = K
    dm_full.close()

    # 4) build the list of parameters you wanted to sweep
    param_ranges = {}
    # ── Stage 1 (detector level) ───────────
    if instrument == 'NIRISS':
        param_ranges['soss_inner_mask_width'] = [20, 40, 80]
        param_ranges['soss_outer_mask_width'] = [35, 70, 140]
    elif instrument == 'NIRSPEC':
        param_ranges['nirspec_mask_width']    = [8, 16, 32]
    elif instrument == 'MIRI':
        param_ranges['miri_drop_groups']      = [6, 12, 24]

    # common 1/f detection thresholds
    param_ranges['jump_threshold']         = [5, 15, 30]
    param_ranges['time_jump_threshold']    = [3, 10, 20]

    # ── Stage 2 (2D calibration) ───────────
    if instrument == 'MIRI':
        param_ranges['miri_trace_width']      = [10, 20, 40]
        param_ranges['miri_background_width'] = [7, 14, 28]
    param_ranges['space_outlier_threshold'] = [5, 15, 30]
    param_ranges['time_outlier_threshold']  = [3, 10, 20]
    param_ranges['pca_components']          = [5, 10, 20]

    # ── Stage 3 (1D extraction) ────────────
    param_ranges['extract_width']           = [15, 30, 60]

    param_order = list(param_ranges.keys())

    # 5) initialize at your config defaults (or median of range if missing)
    current = {}
    for key, vals in param_ranges.items():
        current[key] = cfg.get(key, int(np.median(vals)))
    # bring in weights so cost_function can see them
    current.update(w1=args.w1, w2=args.w2)

    # 6) decide which Stage 1 steps to skip (so PRISM doesn’t blow up)
    stage1_skip = []
    # follow DMS logic for skip flags:
    for step in ['DQInitStep','EmiCorrStep','SaturationStep','ResetStep',
                 'SuperBiasStep','RefPixStep','DarkCurrentStep',
                 'OneOverFStep_grp','LinearityStep','JumpStep',
                 'RampFitStep','GainScaleStep']:
        if cfg.get(step,'run') == 'skip':
            if step == 'OneOverFStep_grp':
                stage1_skip.append('OneOverFStep')
            else:
                stage1_skip.append(step)
    # **force** skip the NIRSpec 1/f step for PRISM (unknown grating)
    if instrument == 'NIRSPEC' and 'PRISM' in mode.upper():
        stage1_skip.append('OneOverFStep')

    # 7) helper to run one trial
    def evaluate_one(params):
        t0 = time.perf_counter()
        # — Stage 1 —
        res1 = run_stage1([dm_slice],
            mode=mode,
            soss_background_model=cfg.get('soss_background_file', None),
            baseline_ints=cfg['baseline_ints'],
            oof_method=cfg.get('oof_method', None),
            superbias_method=cfg.get('superbias_method', None),
            soss_timeseries=cfg.get('soss_timeseries', None),
            soss_timeseries_o2=cfg.get('soss_timeseries_o2', None),
            save_results=False,
            force_redo=cfg.get('force_redo', False),
            flag_up_ramp=cfg.get('flag_up_ramp', False),
            rejection_threshold=params.get('jump_threshold', cfg['jump_threshold']),
            flag_in_time=cfg.get('flag_in_time', True),
            time_rejection_threshold=params.get('time_jump_threshold', cfg['time_jump_threshold']),
            skip_steps=stage1_skip,
            soss_inner_mask_width=params.get('soss_inner_mask_width', cfg.get('soss_inner_mask_width')),
            soss_outer_mask_width=params.get('soss_outer_mask_width', cfg.get('soss_outer_mask_width')),
            nirspec_mask_width=params.get('nirspec_mask_width', cfg.get('nirspec_mask_width')),
            miri_drop_groups=params.get('miri_drop_groups', cfg.get('miri_drop_groups')),
        )
        dm1 = res1[0]

        # — Stage 2 —
        res2, cents = run_stage2(res1,
            mode=mode,
            baseline_ints=cfg['baseline_ints'],
            save_results=False,
            force_redo=False,
            skip_steps=[],
            space_thresh = params.get('space_outlier_threshold', cfg['space_outlier_threshold']),
            time_thresh  = params.get('time_outlier_threshold', cfg['time_outlier_threshold']),
            pca_components=params.get('pca_components', cfg['pca_components']),
            miri_trace_width     = params.get('miri_trace_width', cfg.get('miri_trace_width')),
            miri_background_width= params.get('miri_background_width', cfg.get('miri_background_width')),
        )

        # — Stage 3 —
        dm3 = run_stage3(res2,
            save_results=False,
            force_redo=False,
            extract_method=cfg['extract_method'],
            extract_width=params.get('extract_width', cfg['extract_width']),
            centroids=cents
        )

        dt = time.perf_counter() - t0
        J  = cost_function(dm1, args.w1, args.w2)
        return J, dt

    # 8) run coordinate descent & log
    total_steps = sum(len(v) for v in param_ranges.values())
    with open(args.log, 'w') as f:
        f.write("\t".join(param_order + ['duration_s','J']) + "\n")
        best_J = None
        step = 1
        for key in param_order:
            print(f"\n→ Optimizing {key} ({step}/{len(param_order)})")
            best_val = current[key]
            for trial in param_ranges[key]:
                current[key] = trial
                J, dt = evaluate_one(current)
                f.write("\t".join(str(current[k]) for k in param_order)
                        + f"\t{dt:.1f}\t{J:.6f}\n")
                print(f"   {key}={trial} → J={J:.6f}  ({dt:.1f}s)")
                if best_J is None or J < best_J:
                    best_J, best_val = J, trial
            current[key] = best_val
            print(f"✔  → Best {key} = {best_val} (J={best_J:.6f})")
            step += 1

        f.write("\n# Final optimized parameters:\n")
        for k in param_order:
            f.write(f"# {k} = {current[k]}\n")
        f.write(f"# Final cost J = {best_J:.6f}\n")

    print("\n=== DONE! results in", args.log)

if __name__ == "__main__":
    main()
