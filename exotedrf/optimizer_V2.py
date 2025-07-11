#!/usr/bin/env python3

import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
from jwst import datamodels

from exotedrf.utils       import parse_config, unpack_input_dir, fancyprint
from exotedrf.stage1      import run_stage1
from exotedrf.stage2      import run_stage2
from exotedrf.stage3      import run_stage3

# ————————————————————————————————————————————————————————
# 1) Cost function: weighted sum of white-light and spectral scatter
# ————————————————————————————————————————————————————————

def compute_white_light(dm):
    """
    Extract the white-light curve (sum over all pixels) from a CubeModel-like dm,
    ignoring any NaNs.
    """
    data = np.asarray(dm.data, dtype=float)
    # sum over pixels, skipping NaNs
    wl = np.nansum(data.reshape(data.shape[0], -1), axis=1)
    return wl


def compute_spectral(dm):
    """
    Extract a toy spectral lightcurve: mean over spatial axis for each integration,
    ignoring NaNs. Supports 3D cube (time, rows, cols), 2D output (time, channels),
    and 1D spectra (time,).
    """
    raw = dm.data if hasattr(dm, 'data') else dm
    data = np.asarray(raw, dtype=float)

    if data.ndim == 3:
        # time x spatial x spatial -> time x spectral_channels
        reshaped = data.reshape(data.shape[0], data.shape[1], -1)
        spec = np.nanmean(reshaped, axis=2)
    elif data.ndim == 2:
        # already time x spectral
        spec = np.asarray(data, dtype=float)
    elif data.ndim == 1:
        # single channel per time
        spec = data.reshape(-1, 1)
    else:
        raise ValueError(f"Unexpected data dimensions {data.ndim} in compute_spectral")
    return spec


def cost_function(dm, w1=0.5, w2=0.5):
    """
    Weighted sum of:
      1) white-light fractional scatter (std/median)
      2) mean fractional scatter across spectral channels
    Uses nan-aware statistics to avoid NaNs propagating.
    """
    wl = compute_white_light(dm)
    spec = compute_spectral(dm)

    # compute fractional scatter, ignoring NaNs
    median_wl = np.nanmedian(wl)
    frac_wl = np.nanstd(wl) / abs(median_wl)

    frac_specs = []
    for i in range(spec.shape[1]):
        channel = spec[:, i]
        median_ch = np.nanmedian(channel)
        frac_specs.append(np.nanstd(channel) / abs(median_ch))
    frac_spec = np.nanmean(frac_specs)

    return w1 * frac_wl + w2 * frac_spec

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

    # start global timer
    t0_total = time.perf_counter()

    # load config & discover segment files
    cfg = parse_config(args.config)
    input_files = unpack_input_dir(
        cfg['input_dir'],
        mode=cfg['observing_mode'],
        filetag=cfg['input_filetag'],
        filter_detector=cfg['filter_detector']
    )
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
    dm_full = datamodels.open(seg0)
    K       = min(60, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end   = K
    dm_slice.meta.exposure.nints = K
    dm_full.close()

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
            "nirspec_mask_width":     list(range(5,15,5)),
            "jump_threshold":         list(range(5,15,5)),
            "time_jump_threshold":    list(range(5,15,5)),
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
        "space_outlier_threshold": list(range(5,15,5)),
        "time_outlier_threshold":  list(range(5,15,5)),
        "pca_components":          list(range(5,15,5)),
        "extract_width":           list(range(5,15,5)),
    })

    param_order = list(param_ranges.keys())
    current     = {k: int(np.median(v)) for k,v in param_ranges.items()}
    total_steps = sum(len(v) for v in param_ranges.values())
    count       = 1

    logf = open("Cost_function_V2.txt","w")
    logf.write("\t".join(param_order) + "\tduration_s\tcost\n")

    # coordinate‐descent
    for key in param_order:
        fancyprint(f"\n→ Optimizing {key} (others fixed = { {k:current[k] for k in current if k!=key} })")
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
            baseline_ints = [0, K]

            t0 = time.perf_counter()

            # Stage 1: use correct kw names rejection_threshold/time_rejection_threshold :contentReference[oaicite:2]{index=2}
            st1 = run_stage1(
                [dm_slice],
                mode=cfg['observing_mode'],
                baseline_ints=baseline_ints,
                save_results=False,
                skip_steps=[],
                rejection_threshold     = trial_params["jump_threshold"],
                time_rejection_threshold= trial_params["time_jump_threshold"],
                soss_inner_mask_width   = trial_params.get("soss_inner_mask_width"),
                soss_outer_mask_width   = trial_params.get("soss_outer_mask_width"),
                nirspec_mask_width      = trial_params.get("nirspec_mask_width"),
                miri_drop_groups        = trial_params.get("miri_drop_groups"),
                **cfg.get('stage1_kwargs', {})
            )

            # Stage 2: use space_thresh/time_thresh not space_outlier_threshold/... :contentReference[oaicite:3]{index=3}
            st2, centroids = run_stage2(
                st1,
                mode=cfg['observing_mode'],
                baseline_ints=baseline_ints,
                save_results=False,
                skip_steps=['BadPixStep','PCAReconstructStep'],
                space_thresh     = trial_params["space_outlier_threshold"],
                time_thresh      = trial_params["time_outlier_threshold"],
                pca_components   = trial_params["pca_components"],
                soss_inner_mask_width   = trial_params.get("soss_inner_mask_width"),
                soss_outer_mask_width   = trial_params.get("soss_outer_mask_width"),
                nirspec_mask_width      = trial_params.get("nirspec_mask_width"),
                miri_trace_width        = trial_params.get("miri_trace_width"),
                miri_background_width   = trial_params.get("miri_background_width"),
                **cfg.get('stage2_kwargs', {})
            )
            if isinstance(centroids, np.ndarray):
                centroids = pd.DataFrame(centroids.T, columns=['xpos','ypos'])

            st3 = run_stage3(
                st2,
                centroids=centroids,
                save_results=False,
                skip_steps=[],
                extract_width=trial_params["extract_width"],
                **cfg.get('stage3_kwargs', {})
            )

            # unwrap dict returned by run_stage3 if necessary
            if isinstance(st3, dict):
                st3_model = next(iter(st3.values()))
            else:
                st3_model = st3

            dt   = time.perf_counter() - t0
            cost = cost_function(st3_model)

        
            print(cost)
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
                best_cost, best_val = cost, trial

        current[key] = best_val
        fancyprint(f"✔ Best {key} = {best_val} (cost={best_cost:.6f})")

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
    # ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
