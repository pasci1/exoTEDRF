#!/usr/bin/env python3

import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
from jwst import datamodels
import matplotlib.pyplot as plt
import numpy as np

from exotedrf.utils       import parse_config, unpack_input_dir, fancyprint
from exotedrf.stage1      import run_stage1
from exotedrf.stage2      import run_stage2
from exotedrf.stage3      import run_stage3

### for debuging
def plot_stage3_output_simple(data):
    plt.figure()
    if hasattr(data, 'shape') and len(data.shape) == 1:
        plt.scatter(range(len(data)), data)
    elif hasattr(data, 'shape') and len(data.shape) == 2:
        # Plot the first column vs the second column if possible
        plt.scatter(data[:,0], data[:,1])
    else:
        # Fallback: just plot the data directly
        plt.plot(data)
    plt.show()


# ————————————————————————————————————————————————————————
# 1) Cost function: weighted sum of white-light and spectral scatter
# ————————————————————————————————————————————————————————

def compute_white_light(dm):
    """
    Extract the white-light curve (sum over all pixels) from a CubeModel-like dm,
    ignoring any NaNs.
    """
    print("DEBUG [compute_white_light] input type:", type(dm))
    data = np.asarray(dm.data, dtype=float)
    print("DEBUG [compute_white_light] data shape:", data.shape)
    wl = np.nansum(data.reshape(data.shape[0], -1), axis=1)
    print("DEBUG [compute_white_light] wl shape:", wl.shape, "min/max:", np.nanmin(wl), np.nanmax(wl))
    return wl

def compute_spectral(dm):
    """
    Extract a toy spectral lightcurve: mean over spatial axis for each integration,
    ignoring NaNs. Supports 3D cube (time, rows, cols), 2D output (time, channels),
    and 1D spectra (time,).
    """
    print("DEBUG [compute_spectral] input type:", type(dm))
    raw = dm.data if hasattr(dm, 'data') else dm
    data = np.asarray(raw, dtype=float)
    print("DEBUG [compute_spectral] data shape:", data.shape)
    if data.ndim == 3:
        reshaped = data.reshape(data.shape[0], data.shape[1], -1)
        spec = np.nanmean(reshaped, axis=2)
        print("DEBUG [compute_spectral] 3D -> spec shape:", spec.shape)
    elif data.ndim == 2:
        spec = np.asarray(data, dtype=float)
        print("DEBUG [compute_spectral] 2D -> spec shape:", spec.shape)
    elif data.ndim == 1:
        spec = data.reshape(-1, 1)
        print("DEBUG [compute_spectral] 1D -> spec shape:", spec.shape)
    else:
        raise ValueError(f"Unexpected data dimensions {data.ndim} in compute_spectral")
    print("DEBUG [compute_spectral] spec min/max:", np.nanmin(spec), np.nanmax(spec))
    return spec

def cost_function(dm, w1=0.5, w2=0.5):
    """
    Weighted sum of:
      1) white-light fractional scatter (std/median)
      2) mean fractional scatter across spectral channels
    Uses nan-aware statistics to avoid NaNs propagating.
    """
    print("DEBUG [cost_function] input type:", type(dm))
    wl = compute_white_light(dm)
    spec = compute_spectral(dm)
    median_wl = np.nanmedian(wl)
    frac_wl = np.nanstd(wl) / abs(median_wl)
    print("DEBUG [cost_function] frac_wl:", frac_wl)

    frac_specs = []
    for i in range(spec.shape[1]):
        channel = spec[:, i]
        median_ch = np.nanmedian(channel)
        channel_scatter = np.nanstd(channel) / abs(median_ch)
        frac_specs.append(channel_scatter)
        print(f"DEBUG [cost_function] channel {i}: median={median_ch}, frac={channel_scatter}")
    frac_spec = np.nanmean(frac_specs)
    print("DEBUG [cost_function] frac_spec:", frac_spec)

    total = w1 * frac_wl + w2 * frac_spec
    print(f"DEBUG [cost_function] total cost: {total}")
    return total

def cost_function(st3_model):
    """
    Compute cost on Stage 3 output.
    Expects st3_model as numpy array of flux values (1D or 2D).
    If 2D, sums across wavelength to get the white-light curve.
    Returns relative scatter = std/median.
    """
    print("DEBUG [cost_function st3_model] input type:", type(st3_model))
    flux = np.asarray(st3_model)
    print("DEBUG [cost_function st3_model] flux shape:", flux.shape, "min/max:", np.nanmin(flux), np.nanmax(flux))
    if flux.ndim == 1:
        wl = flux
    elif flux.ndim == 2:
        wl = np.nansum(flux, axis=1)
    else:
        raise ValueError(f"Unexpected flux ndim = {flux.ndim}")
    std = np.nanstd(wl)
    med = np.nanmedian(wl)
    print("DEBUG [cost_function st3_model] wl shape:", wl.shape, "std:", std, "med:", med)
    return std / abs(med)

# ————————————————————————————————————————————————————————
# 2) Main optimizer
# ————————————————————————————————————————————————————————

def main():
    parser = argparse.ArgumentParser(
        description="Coordinate-descent optimizer for exoTEDRF Stages 1–3"
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
    print("DEBUG [main] args:", args)

    # start global timer
    t0_total = time.perf_counter()

    # load config & discover segment files
    cfg = parse_config(args.config)
    print("DEBUG [main] loaded cfg:", cfg)
    input_files = unpack_input_dir(
        cfg['input_dir'],
        mode=cfg['observing_mode'],
        filetag=cfg['input_filetag'],
        filter_detector=cfg['filter_detector']
    )
    print("DEBUG [main] input_files:", input_files)
    if not input_files:
        fancyprint(f"[WARN] No files in {cfg['input_dir']}, globbing *.fits")
        input_files = sorted(glob.glob(os.path.join(cfg['input_dir'], "*.fits")))
    if not input_files:
        raise RuntimeError(f"No FITS found in {cfg['input_dir']}")
    fancyprint(f"Using {len(input_files)} segment(s) from {cfg['input_dir']}")

    # load a short slice of the first segment
    seg0 = os.path.join(
        cfg['input_dir'],
        "jw01366003001_04101_00001-seg001_nrs1_uncal.fits"
    )
    print("DEBUG [main] opening segment file:", seg0)
    dm_full = datamodels.open(seg0)
    K       = min(60, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end   = K
    dm_slice.meta.exposure.nints = K
    dm_full.close()
    print("DEBUG [main] dm_slice shape:", dm_slice.data.shape)

    # build parameter grid
    param_ranges = {}
    if args.instrument == "NIRISS":
        param_ranges.update({
            "soss_inner_mask_width": [20,40,80],
            "soss_outer_mask_width": [35,70,140],
            "jump_threshold":         [5,15,30],
            "time_jump_threshold":    [3,10,20],
        })
    elif args.instrument == "NIRSPEC":
        param_ranges.update({
            'time_window':              [301,9,17],
            'thresh':                   list(range(5,15,5)),
            'rejection_threshold':      list(range(5,15,5)), 
            'time_rejection_threshold': list(range(5,15,5)),            
            "nirspec_mask_width":       list(range(5,15,5)),
        })
    else:
        param_ranges.update({
            "miri_drop_groups":      [6,12,24],
            "jump_threshold":        [5,15,30],
            "time_jump_threshold":   [3,10,20],
            "miri_trace_width":      [10,20,40],
            "miri_background_width": [7,14,28],
        })
    # always sweep these
    param_ranges.update({
        "space_outlier_threshold": list(range(5,10,5)),
        "time_outlier_threshold":  list(range(5,10,5)),
        "pca_components":          list(range(5,10,5)),
        "extract_width":           list(range(5,10,5)),
    })

    print("DEBUG [main] param_ranges:", param_ranges)
    param_order = list(param_ranges.keys())
    print("DEBUG [main] param_order:", param_order)
    current     = {k: int(np.median(v)) for k,v in param_ranges.items()}
    print("DEBUG [main] initial current param values:", current)
    total_steps = sum(len(v) for v in param_ranges.values())
    print("DEBUG [main] total_steps:", total_steps)
    count       = 1

    logf = open("Cost_function_V2.txt","w")
    logf.write("\t".join(param_order) + "\tduration_s\tcost\n")

    # --------- Stage Parameter Mapping ---------
    # Pass only relevant params to each Stage
    stage1_keys = [
        'rejection_threshold', 'time_rejection_threshold',
        'nirspec_mask_width', 'soss_inner_mask_width',
        'soss_outer_mask_width', 'miri_drop_groups', 'jump_threshold', 'time_jump_threshold',
        # Add other stage1-specific parameters here if needed
    ]
    stage2_keys = [
        'space_outlier_threshold', 'time_outlier_threshold', 'pca_components',
        'thresh', 'box_size', 'time_window',
        'miri_trace_width', 'miri_background_width'
        # Add other stage2-specific parameters here if needed
    ]
    stage3_keys = [
        'extract_width'
        # Add other stage3-specific parameters here if needed
    ]

    # coordinate‐descent
    for key in param_order:
        fancyprint(f"\n→ Optimizing {key} (others fixed = { {k:current[k] for k in current if k!=key} })")
        print("DEBUG [main loop] optimizing key:", key)
        best_cost, best_val = None, current[key]

        for trial in param_ranges[key]:
            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} starting {key}={trial}",
                "\n############################################\n",
                flush=True
            )            

            trial_params = current.copy()
            trial_params[key] = trial
            print("DEBUG [trial] trial_params:", trial_params)
            baseline_ints = list(range(dm_slice.data.shape[0]))
            print("DEBUG [trial] baseline_ints:", baseline_ints)

            t0 = time.perf_counter()

            # ----------- Build Stage kwargs dynamisch -----------
            s1_args = {k: trial_params[k] for k in stage1_keys if k in trial_params}
            s2_args = {k: trial_params[k] for k in stage2_keys if k in trial_params}
            s3_args = {k: trial_params[k] for k in stage3_keys if k in trial_params}
            print("DEBUG [trial] s1_args:", s1_args)
            print("DEBUG [trial] s2_args:", s2_args)
            print("DEBUG [trial] s3_args:", s3_args)

            # Speziell für JumpStep (nur wenn time_window übergeben wird)
            if "time_window" in trial_params:
                s1_args["JumpStep"] = {"time_window": trial_params["time_window"]}
                print("DEBUG [trial] s1_args after JumpStep inject:", s1_args)

            print("DEBUG [trial] calling run_stage1 with args:")
            print("    [dm_slice] shape:", dm_slice.data.shape)
            print("    mode:", cfg['observing_mode'])
            print("    baseline_ints:", baseline_ints)
            print("    save_results:", False)
            print("    skip_steps:", [])
            print("    kwargs:", s1_args)
            # Stage 1 run
            st1 = run_stage1(
                [dm_slice],
                mode=cfg['observing_mode'],
                baseline_ints=baseline_ints,
                save_results=False,
                skip_steps=[],
                **s1_args
            )
            print("DEBUG [trial] finished run_stage1, st1 type:", type(st1))

            # Stage 2 run
            print("DEBUG [trial] calling run_stage2 with args:")
            print("    st1 type:", type(st1))
            print("    s2_args:", s2_args)
            print("    stage2_kwargs:", cfg.get('stage2_kwargs', {}))
            st2, centroids = run_stage2(
                st1,
                mode=cfg['observing_mode'],
                baseline_ints=baseline_ints,
                save_results=False,
                skip_steps=['BadPixStep','PCAReconstructStep'],
                **s2_args,
                **cfg.get('stage2_kwargs', {})
            )
            print("DEBUG [trial] finished run_stage2, st2 type:", type(st2), "centroids type:", type(centroids))
            if isinstance(centroids, np.ndarray):
                print("DEBUG [trial] centroids.shape:", centroids.shape)
                centroids = pd.DataFrame(centroids.T, columns=['xpos','ypos'])

            # Stage 3 run
            print("DEBUG [trial] calling run_stage3 with args:")
            print("    st2 type:", type(st2))
            print("    centroids type:", type(centroids))
            print("    s3_args:", s3_args)
            print("    stage3_kwargs:", cfg.get('stage3_kwargs', {}))
            st3 = run_stage3(
                st2,
                centroids=centroids,
                save_results=False,
                skip_steps=[],
                **s3_args,
                **cfg.get('stage3_kwargs', {})
            )
            print("DEBUG [trial] finished run_stage3, st3 type:", type(st3))

            # ==== Änderung beginnt hier: Robust unwrap für Stage 3 Output ====
            st3_model = st3
            # Falls dict, nimm den ersten Value (z.B. NIRSpec: {'NRS1': Model, ...})
            if isinstance(st3_model, dict):
                print("DEBUG [trial] st3_model is dict, keys:", st3_model.keys())
                st3_model = next(iter(st3_model.values()))
            # Falls JWST-Model: versuche auf die .data zuzugreifen (wichtig!)
            if hasattr(st3_model, "data"):
                print("DEBUG [trial] st3_model has .data, extracting data, shape:", st3_model.data.shape)
                st3_model = st3_model.data
            # Falls immer noch nicht np.array: versuch es zu casten
            st3_model = np.asarray(st3_model)
            print("DEBUG [trial] final st3_model shape:", st3_model.shape, "min/max:", np.nanmin(st3_model), np.nanmax(st3_model))
            # ==== Änderung endet hier ====
 
            dt   = time.perf_counter() - t0

            plot_stage3_output_simple(st3_model)
            cost = cost_function(st3_model)

            print("DEBUG [trial] cost:", cost)
            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} completed (dt={dt:.1f}s)",
                "\n############################################\n",
                flush=True
            )
            count += 1

            logf.write(
                "\t".join(str(trial_params[k]) for k in param_order)
                + f"\t{dt:.1f}\t{cost:.6f}\n"
            )

            if best_cost is None or cost < best_cost:
                print(f"DEBUG [trial] updating best: old={best_cost}, new={cost}, val={trial}")
                best_cost, best_val = cost, trial

        current[key] = best_val
        fancyprint(f"Best {key} = {best_val} (cost={best_cost:.6f})")
        print("DEBUG [main loop] after param sweep, current:", current)

    logf.close()
    fancyprint("\n=== FINAL OPTIMUM ===\n")
    fancyprint(current)
    fancyprint("Log saved to Cost_function_V2.txt")

    # ─── STOP GLOBAL TIMER & PRINT TOTAL ────────────────────────────────
    t1_total = time.perf_counter()
    total = t1_total - t0_total
    h = int(total) // 3600
    m = (int(total) % 3600) // 60
    s = total % 60
    print(f"TOTAL optimization runtime: {h}h {m:02d}min {s:04.1f}s")
    print("DEBUG [main] finished all, FINAL current:", current)
    # ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
