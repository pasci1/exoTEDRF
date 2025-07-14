#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizer script converted from Jupyter Notebook for exoTEDRF pipeline.
Runs parameter sweeps for Stage1-3 and evaluates cost function.
"""
import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
from jwst import datamodels

from exotedrf.utils import parse_config, fancyprint
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3


def cost_function(st3):
    """
    Compute a robust cost on the Stage 3 output dict using MAD:
      1) grab the Flux array
      2) collapse to a white-light series if 2D
      3) drop NaNs
      4) compute median absolute deviation (MAD)
      5) return MAD
    """
    flux = np.asarray(st3["Flux"])
    if flux.ndim == 2:
        series = np.nansum(flux, axis=1)
    elif flux.ndim == 1:
        series = flux.copy()
    else:
        raise ValueError(f"Unexpected flux ndim = {flux.ndim}")
    series = series[~np.isnan(series)]
    if series.size == 0:
        raise ValueError("No valid flux values after dropping NaNs")
    med = np.median(series)
    mad = np.median(np.abs(series - med))
    return mad


def main():
    parser = argparse.ArgumentParser(description="Run exoTEDRF parameter optimizer V2.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--instrument", required=True,
                        help="Instrument name, e.g. NIRSPEC or NIRISS/SOSS.")
    args = parser.parse_args()

    cfg = parse_config(args.config)
    input_dir = cfg['input_dir']

    # Locate first uncalibrated segment file
    fits_files = sorted(glob.glob(os.path.join(input_dir, "*_uncal.fits")))
    if not fits_files:
        raise FileNotFoundError(f"No *_uncal.fits files found in {input_dir}")
    seg0 = fits_files[0]

    # Load and slice integrations
    dm_full = datamodels.open(seg0)
    K = min(60, dm_full.data.shape[0])
    dm_slice = dm_full.copy()
    dm_slice.data = dm_full.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end = K
    dm_slice.meta.exposure.nints = K
    dm_full.close()

    # Monkey-patch utils functions to handle edge cases
    import exotedrf.utils as utils
    # 1) Unknown gratings fallback
    _orig_trace_start = utils.get_nrs_trace_start
    def _safe_get_nrs_trace_start(detector, subarray, grating):
        base = grating.split('+')[0]
        try:
            return _orig_trace_start(detector, subarray, base)
        except ValueError:
            return 0
    utils.get_nrs_trace_start = _safe_get_nrs_trace_start

    # 2) Catch centroiding failures in NIRSpec and fallback to straight trace
    _orig_centroid_nrspec = utils.get_centroids_nirspec
    def _safe_get_centroids_nirspec(deepframe, xstart=0, xend=None,
                                    save_results=True, save_filename=''):
        try:
            return _orig_centroid_nirspec(deepframe, xstart=xstart,
                                           xend=xend, save_results=save_results,
                                           save_filename=save_filename)
        except Exception as e:
            fancyprint(f"Warning in centroiding NIRSpec: {{e}}, using fallback", msg_type="WARNING")
            dimy, dimx = deepframe.shape
            if xend is None:
                xend = dimx
            xx = np.arange(xstart, xend)
            yy = np.full_like(xx, dimy//2, dtype=float)
            return np.array([xx, yy])
    utils.get_centroids_nirspec = _safe_get_centroids_nirspec

    # Define parameter ranges
    param_ranges = {
        'rejection_threshold': list(range(5, 16, 5)),
        'extract_width': list(range(10, 11, 5)),
    }
    param_order = list(param_ranges.keys())
    current = {k: int(np.median(v)) for k, v in param_ranges.items()}
    total_steps = sum(len(v) for v in param_ranges.values())

    # Define stage-specific keys
    stage1_keys = ['rejection_threshold', 'time_rejection_threshold',
                   'nirspec_mask_width', 'soss_inner_mask_width',
                   'soss_outer_mask_width', 'miri_drop_groups']
    stage2_keys = ['space_outlier_threshold', 'time_outlier_threshold', 'pca_components',
                   'thresh', 'box_size', 'time_window',
                   'miri_trace_width', 'miri_background_width']
    stage3_keys = ['extract_width']

    # Open log file
    log_filename = "Cost_function_V2.txt"
    logf = open(log_filename, "w")
    logf.write("\t".join(param_order) + "\tduration_s\tcost\n")

    count = 1
    for key in param_order:
        fixed = {k: current[k] for k in current if k != key}
        fancyprint(f"→ Optimizing {key} (others fixed = {fixed})")
        best_cost = None
        best_val = current[key]

        for trial in param_ranges[key]:
            fancyprint(f"Step {count}/{total_steps}: Testing {key}={trial}")
            trial_params = current.copy()
            trial_params[key] = trial

            nints = dm_slice.data.shape[0]
            baseline_ints = [0, nints - 1]

            t0 = time.perf_counter()
            # Build kwargs for each stage
            s1_args = {k: trial_params[k] for k in stage1_keys if k in trial_params}
            if 'time_window' in trial_params:
                s1_args['JumpStep'] = {'time_window': trial_params['time_window']}
            s2_args = {k: trial_params[k] for k in stage2_keys if k in trial_params}
            s3_args = {k: trial_params[k] for k in stage3_keys if k in trial_params}

            # Run Stage 1
            st1 = run_stage1(
                [dm_slice],
                mode=cfg['observing_mode'],
                baseline_ints=baseline_ints,
                flag_up_ramp=True,
                save_results=False,
                skip_steps=[],
                **s1_args
            )

            # Run Stage 2
            st2, centroids = run_stage2(
                st1,
                mode=cfg['observing_mode'],
                baseline_ints=baseline_ints,
                save_results=False,
                skip_steps=['BadPixStep', 'PCAReconstructStep'],
                **s2_args,
                **cfg.get('stage2_kwargs', {})
            )
            if isinstance(centroids, np.ndarray):
                centroids = pd.DataFrame(centroids.T, columns=['xpos', 'ypos'])

            # Run Stage 3
            st3 = run_stage3(
                st2,
                centroids=centroids,
                save_results=False,
                skip_steps=[],
                **s3_args,
                **cfg.get('stage3_kwargs', {})
            )

            # Evaluate cost
            cost = cost_function(st3)
            dt = time.perf_counter() - t0

            # Log results
            vals = "\t".join(str(trial_params[k]) for k in param_order)
            logf.write(f"{vals}\t{dt:.1f}\t{cost:.6f}\n")

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_val = trial
            count += 1

        current[key] = best_val
        fancyprint(f"Best {key} = {best_val} (cost={best_cost:.6f})")

    logf.close()
    fancyprint(f"=== FINAL OPTIMUM === {current}")
    fancyprint(f"Log saved to {log_filename}")


if __name__ == "__main__":
    main()
