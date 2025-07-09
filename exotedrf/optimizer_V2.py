# optimizer_V2.py (updated version with logging + optional plotting)

import time
import argparse
import os
import numpy as np
from jwst import datamodels
from exotedrf.utils import parse_config
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3
from astropy.stats import mad_std


def compute_white_light_from_dm(dm):
    wl = dm.data.reshape(dm.data.shape[0], -1).sum(axis=1)
    return wl

def cost_function(dm, w1=1.0, w2=1.0):
    wl = compute_white_light_from_dm(dm)
    frac = mad_std(wl) / np.abs(np.median(wl))
    return w1 * frac + w2 * frac

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="run_WASP39b.yaml")
    p.add_argument("--log", default="Cost_function_V2.txt")
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=1.0)
    args = p.parse_args()

    cfg = parse_config(args.config)
    cfg_dir = os.path.dirname(args.config) or '.'
    mode = cfg['observing_mode']
    instrument = mode.split('/')[0].upper()

    seg = os.path.join(cfg_dir, cfg['input_dir'], "jw01366003001_04101_00001-seg001_nrs1_uncal.fits")
    dm_full = datamodels.open(seg)
    K = min(60, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end   = K
    dm_slice.meta.exposure.nints            = K
    dm_full.close()

    # --- Define sweep parameters ---
    param_ranges = {}
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
        param_ranges['miri_trace_width']       = [10, 20, 40]
        param_ranges['miri_background_width']  = [7, 14, 28]

    param_ranges['space_outlier_threshold'] = [5, 15, 30]
    param_ranges['time_outlier_threshold']  = [3, 10, 20]
    param_ranges['pca_components']          = [5, 10, 20]
    param_ranges['extract_width']           = [15, 30, 60]

    param_order = list(param_ranges.keys())
    current = {k: cfg.get(k, int(np.median(v))) for k, v in param_ranges.items()}
    current.update(w1=args.w1, w2=args.w2)

    # --- Log file ---
    with open(args.log, 'w') as logfile:
        logfile.write("\t".join(param_order + ['duration_s', 'J']) + "\n")

        count = 1
        best_J = None
        for key in param_order:
            print(f"\n→ Optimizing {key} ({count}/{len(param_order)})")
            best_val = current[key]
            for trial in param_ranges[key]:
                trial_params = current.copy()
                trial_params[key] = trial

                def evaluate_one(params):
                    t0 = time.perf_counter()
                    results1 = run_stage1([dm_slice],
                        mode=mode,
                        baseline_ints=cfg['baseline_ints'],
                        save_results=False,
                        skip_steps=[],
                        soss_inner_mask_width=params.get('soss_inner_mask_width', cfg.get('soss_inner_mask_width', 40)),
                        soss_outer_mask_width=params.get('soss_outer_mask_width', cfg.get('soss_outer_mask_width', 70)),
                        nirspec_mask_width=params.get('nirspec_mask_width', cfg.get('nirspec_mask_width', 16)),
                        miri_drop_groups=params.get('miri_drop_groups', cfg.get('miri_drop_groups', 12)),
                        jump_threshold=params.get('jump_threshold', cfg.get('jump_threshold', 15)),
                        time_jump_threshold=params.get('time_jump_threshold', cfg.get('time_jump_threshold', 10)),
                        flag_up_ramp=cfg.get('flag_up_ramp', False),
                        flag_in_time=cfg.get('flag_in_time', True),
                    )
                    dm1 = results1[0]

                    results2, centroids = run_stage2(results1,
                        mode=mode,
                        baseline_ints=cfg['baseline_ints'],
                        save_results=False,
                        force_redo=False,
                        skip_steps=[],
                        space_thresh=params.get('space_outlier_threshold', cfg.get('space_outlier_threshold', 15)),
                        time_thresh=params.get('time_outlier_threshold', cfg.get('time_outlier_threshold', 10)),
                        pca_components=params.get('pca_components', cfg.get('pca_components', 10)),
                        miri_trace_width=params.get('miri_trace_width', cfg.get('miri_trace_width', 20)),
                        miri_background_width=params.get('miri_background_width', cfg.get('miri_background_width', 14))
                    )

                    dm3 = run_stage3(results2,
                        save_results=False,
                        force_redo=False,
                        extract_method=cfg.get('extract_method', 'box'),
                        extract_width=params.get('extract_width', cfg.get('extract_width', 30)),
                        centroids=centroids
                    )

                    dt = time.perf_counter() - t0
                    J = cost_function(dm1, params['w1'], params['w2'])
                    return J, dt

                J, dt = evaluate_one(trial_params)
                row = [str(trial_params.get(p, '')) for p in param_order] + [f"{dt:.1f}", f"{J:.6f}"]
                logfile.write("\t".join(row) + "\n")
                print(f"   {key}={trial} → J={J:.6f} ({dt:.1f}s)")

                if best_J is None or J < best_J:
                    best_J = J
                    best_val = trial

            current[key] = best_val
            print(f"✔→ Best {key} = {best_val} (J={best_J:.6f})")
            count += 1

        logfile.write("\n# Final optimized parameters:\n")
        for k in param_order:
            logfile.write(f"# {k} = {current[k]}\n")
        logfile.write(f"# Final cost J = {best_J:.6f}\n")

if __name__ == "__main__":
    main()
