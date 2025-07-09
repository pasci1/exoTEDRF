#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
from astropy.stats import mad_std
from jwst import datamodels
from exotedrf.utils import parse_config
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3

# ----------------------------------------
# 1) Cost‐function definitions
# ----------------------------------------
def compute_white_light(dm):
    """Sum over all pixels to extract the white‐light curve."""
    wl = dm.data.reshape(dm.data.shape[0], -1).sum(axis=1)
    return wl

def cost_function(dm, w1=1.0, w2=1.0):
    """Robust fractional scatter of the white-light curve."""
    wl = compute_white_light(dm)
    frac = mad_std(wl) / np.abs(np.median(wl))
    return w1 * frac + w2 * frac

# ----------------------------------------
# 2) Main & coordinate‐descent
# ----------------------------------------
def main():
    # 1) parse args & config
    p = argparse.ArgumentParser(description="Optimize exoTEDRF Stages 1–3")
    p.add_argument("--config", default="run_WASP39b.yaml",
                   help="Path to your run_*.yaml")
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=1.0)
    args = p.parse_args()
    cfg = parse_config(args.config)

    # 2) load a short slice exactly as in optimize_stage1.py
    seg = os.path.join(cfg['input_dir'],
                       "jw01366003001_04101_00001-seg001_nrs1_uncal.fits")
    dm_full = datamodels.open(seg)
    K = min(60, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end   = K
    dm_slice.meta.exposure.nints            = K
    dm_full.close()

    # 3) Parameter ranges: only those you listed
    param_ranges = {}
    mode       = cfg['observing_mode'].upper()
    instr      = mode.split('/')[0]

    # ── Stage 1 detector‐level params ───────────
    if instr == 'NIRISS':
        param_ranges['soss_inner_mask_width'] = [20, 40, 80]
        param_ranges['soss_outer_mask_width'] = [35, 70,140]
    if instr == 'NIRSPEC':
        param_ranges['nirspec_mask_width']    = [8, 16, 32]
    if instr == 'MIRI':
        param_ranges['miri_drop_groups']      = [6, 12, 24]

    # common jump detection thresholds
    param_ranges['jump_threshold']         = [5, 15, 30]
    param_ranges['time_jump_threshold']    = [3, 10, 20]

    # ── Stage 2 2D‐calib params ───────────
    if instr == 'MIRI':
        param_ranges['miri_trace_width']      = [10, 20, 40]
        param_ranges['miri_background_width'] = [7, 14, 28]
    param_ranges['space_outlier_threshold'] = [5, 15, 30]
    param_ranges['time_outlier_threshold']  = [3, 10, 20]
    param_ranges['pca_components']          = [5, 10, 20]

    # ── Stage 3 1D‐extract params ──────────
    param_ranges['extract_width']           = [15, 30, 60]

    param_order = list(param_ranges.keys())
    total_steps = sum(len(v) for v in param_ranges.values())

    # 4) initialize current best at yaml defaults (or median if missing)
    current = {}
    for key, vals in param_ranges.items():
        current[key] = cfg.get(key, int(np.median(vals)))
    current.update(w1=args.w1, w2=args.w2)

    # 5) open log file
    logfile = open("Cost_function_allstages.txt", "w")
    logfile.write("\t".join(param_order + ['duration_s', 'J']) + "\n")

    count = 1
    best_J = None

    # 6) coordinate‐descent
    for key in param_order:
        print(f"\n→ Optimizing {key} ({count}/{len(param_order)})")
        best_val = current[key]

        for trial in param_ranges[key]:
            trial_params = current.copy()
            trial_params[key] = trial

            # --- run one trial across all three stages ---
            t0 = time.perf_counter()

            # Stage 1
            res1 = run_stage1([dm_slice],
                mode=cfg['observing_mode'],
                baseline_ints=cfg['baseline_ints'],
                save_results=False,
                force_redo=cfg.get('force_redo', False),
                skip_steps=[],  # use DMS defaults
                flag_up_ramp=cfg.get('flag_up_ramp', False),
                jump_threshold=trial_params.get('jump_threshold', cfg['jump_threshold']),
                flag_in_time=cfg.get('flag_in_time', True),
                time_rejection_threshold=trial_params.get('time_jump_threshold', cfg['time_jump_threshold']),
                soss_inner_mask_width=trial_params.get('soss_inner_mask_width', cfg.get('soss_inner_mask_width')),
                soss_outer_mask_width=trial_params.get('soss_outer_mask_width', cfg.get('soss_outer_mask_width')),
                nirspec_mask_width=trial_params.get('nirspec_mask_width', cfg.get('nirspec_mask_width')),
                miri_drop_groups=trial_params.get('miri_drop_groups', cfg.get('miri_drop_groups')),
            )
            dm1 = res1[0]

            # Stage 2
            res2, cents = run_stage2(res1,
                mode=cfg['observing_mode'],
                baseline_ints=cfg['baseline_ints'],
                save_results=False,
                force_redo=False,
                skip_steps=[],
                space_thresh = trial_params.get('space_outlier_threshold', cfg['space_outlier_threshold']),
                time_thresh  = trial_params.get('time_outlier_threshold', cfg['time_outlier_threshold']),
                pca_components=trial_params.get('pca_components', cfg['pca_components']),
                miri_trace_width     = trial_params.get('miri_trace_width', cfg.get('miri_trace_width')),
                miri_background_width= trial_params.get('miri_background_width', cfg.get('miri_background_width')),
            )

            # Stage 3
            dm3 = run_stage3(res2,
                save_results=False,
                force_redo=False,
                extract_method=cfg['extract_method'],
                extract_width=trial_params.get('extract_width', cfg['extract_width']),
                centroids=cents
            )

            dt = time.perf_counter() - t0
            J  = cost_function(dm1, args.w1, args.w2)

            # log & print
            row = [ str(trial_params[k]) for k in param_order ]
            logfile.write("\t".join(row + [f"{dt:.1f}", f"{J:.6f}"]) + "\n")
            print(f"   {key}={trial} → J={J:.6f} ({dt:.1f}s)")

            if best_J is None or J < best_J:
                best_J, best_val = J, trial

        current[key] = best_val
        print(f"✔  → Best {key} = {best_val} (J={best_J:.6f})")
        count += 1

    logfile.write("\n# Final optimized parameters:\n")
    for k in param_order:
        logfile.write(f"# {k} = {current[k]}\n")
    logfile.write(f"# Final cost J = {best_J:.6f}\n")
    logfile.close()

    print("\n=== FINAL OPTIMUM ===")
    print("params =", {k: current[k] for k in param_order})
    print("J =", best_J)

if __name__ == "__main__":
    main()
