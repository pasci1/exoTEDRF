#!/usr/bin/env python3

import os, time, itertools
import numpy as np
import pandas as pd
from jwst import datamodels
from astropy.stats import mad_std
from exotedrf.utils    import parse_config
from exotedrf.stage1   import run_stage1
from exotedrf.stage2   import run_stage2
from exotedrf.stage3   import run_stage3

# ----------------------------------------
# 1) Cost-function definitions
# ----------------------------------------
def compute_white_light(lc):
    """White-light curve from lightcurve array."""
    wl = np.asarray(lc)
    if wl.ndim == 2:
        wl = np.nansum(wl, axis=1)
    return wl

def compute_spectral(lc):
    """Mean spectral lightcurve (for each channel/row/col)."""
    arr = np.asarray(lc)
    if arr.ndim == 2:
        return arr
    elif arr.ndim == 1:
        return arr[:, None]
    else:
        raise ValueError("Lightcurve array shape not supported.")

def cost_function(lc, w1=1.0, w2=1.0):
    """Weighted scatter in white-light and spectral channels."""
    wl = compute_white_light(lc)
    spec = compute_spectral(lc)
    frac_wl = mad_std(wl) / np.abs(np.median(wl))
    frac_spec = np.mean([mad_std(spec[:, i]) / np.abs(np.median(spec[:, i])) for i in range(spec.shape[1])])
    return w1 * frac_wl + w2 * frac_spec

# ----------------------------------------
# 2) Main & coordinate-descent
# ----------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="run_WASP39b.yaml")
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    args = parser.parse_args()
    cfg = parse_config(args.config)

    t0_total = time.perf_counter()

    seg1 = os.path.join(cfg['input_dir'], "jw01366003001_04101_00001-seg001_nrs1_uncal.fits")
    dm_full = datamodels.open(seg1)
    K = min(60, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end   = K
    dm_slice.meta.exposure.nints = K
    dm_full.close()

    # FAST-params for test
    param_ranges = {
        'time_window':              [5,7],
        'box_size':                 [10,20],
        'thresh':                   [10,20],
        'rejection_threshold':      [10,20],
        'time_rejection_threshold': [10,20],
        'nirspec_mask_width':       [8,20],
    }
    param_order = [
        'time_window',
        'box_size',
        'thresh',
        'rejection_threshold',
        'time_rejection_threshold',
        'nirspec_mask_width',
    ]
    count = 1
    total_steps = sum(len(v) for v in param_ranges.values())
    current = {p: int(np.median(param_ranges[p])) for p in param_order}
    current.update(w1=args.w1, w2=args.w2)

    skip_steps = []

    def evaluate_one(params):
        print("Running with params:", params)
        # Stage 1
        s1_kwargs = {
            'rejection_threshold':      params['rejection_threshold'],
            'time_rejection_threshold': params['time_rejection_threshold'],
            'nirspec_mask_width':       params['nirspec_mask_width'],
            'JumpStep': {
                'time_window': params['time_window']
            }
        }
        baseline_ints = list(range(dm_slice.data.shape[0]))

        t0 = time.perf_counter()
        st1 = run_stage1(
            [dm_slice],
            mode=cfg['observing_mode'],
            baseline_ints=baseline_ints,
            save_results=False,
            skip_steps=skip_steps,
            **s1_kwargs
        )
        # Stage 2
        st2, centroids = run_stage2(
            st1,
            mode=cfg['observing_mode'],
            baseline_ints=baseline_ints,
            save_results=False,
            skip_steps=['BadPixStep','PCAReconstructStep'],
            space_thresh=params['thresh'],
            time_thresh=params['thresh'],
            pca_components=5,
            box_size=params['box_size'],
            time_window=params['time_window'],
        )
        if isinstance(centroids, np.ndarray):
            import pandas as pd
            centroids = pd.DataFrame(centroids.T, columns=['xpos','ypos'])
        # Stage 3
        st3 = run_stage3(
            st2,
            centroids=centroids,
            save_results=False,
            skip_steps=[],
            extract_width=5,
        )
        # ------- SUPER ROBUST TYP-UNWRAP -------
        # Stage 3 Output (kann dict, Model, ndarray, etc sein)
        print("DEBUG st3 type:", type(st3))
        if isinstance(st3, dict):
            print("DEBUG st3 keys:", st3.keys())
            st3_model = next(iter(st3.values()))
        else:
            st3_model = st3
        print("DEBUG st3_model type:", type(st3_model))
        if hasattr(st3_model, 'data'):
            lc = np.asarray(st3_model.data)
            print("DEBUG st3_model.data shape:", lc.shape)
        elif isinstance(st3_model, np.ndarray):
            lc = st3_model
            print("DEBUG st3_model ndarray shape:", lc.shape)
        else:
            print("!! UNKNOWN LIGHTCURVE TYPE:", type(st3_model))
            print("Dir geholfen hat: evaluate_one-debugging.")
            raise RuntimeError("Stage 3 output unknown format: %s" % str(type(st3_model)))

        if lc.ndim == 0:
            raise ValueError("Lightcurve output is scalar!? Something is wrong.")

        dt = time.perf_counter() - t0
        J = cost_function(lc, params['w1'], params['w2'])
        print(f"DEBUG J={J:.6f}, dt={dt:.2f}")
        return J, dt


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

    for key in param_order:
        print(f"\n→ Optimizing {key} (others fixed = "
              f"{ {k:current[k] for k in current if k!=key} })")
        best_J = None
        best_val = current[key]
        best_dt = None

        for trial in param_ranges[key]:
            trial_params = current.copy()
            trial_params[key] = trial
            J, dt = evaluate_one(trial_params)

            print(f"\n\n\n###########################################################\n Step: {count}/{total_steps} completed\n###########################################################\n\n\n")
            count += 1
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
    print("\n=== FINAL OPTIMUM ===")
    print("params =", {k: current[k] for k in param_order})
    print("J =", best_J)
    print("last dt =", best_dt)

    t1_total = time.perf_counter()
    total = t1_total - t0_total
    h = int(total) // 3600
    m = (int(total) % 3600) // 60
    s = total % 60 
    print(f"TOTAL optimization runtime: {h}h {m:02d}min {s:04.1f}s")

    logfile = open("Cost_function.txt", "a")
    logfile.write("\n# Final optimized parameters:\n")
    for k in param_order:
        logfile.write(f"# {k} = {current[k]}\n")
    logfile.write(f"# Final cost J = {best_J:.6f}\n")
    logfile.close()

if __name__ == "__main__":
    main()
 