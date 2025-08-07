#!/usr/bin/env python3

import os
import glob
import time
import argparse
from exotedrf import utils
import yaml

cfg = yaml.safe_load(open('run_optimize.yaml'))
os.environ.setdefault('CRDS_PATH',    cfg.get('crds_cache_path', './crds_cache'))
os.environ.setdefault('CRDS_SERVER_URL','https://jwst-crds.stsci.edu')
os.environ.setdefault('CRDS_CONTEXT',  cfg.get('crds_context',   'jwst_1322.pmap'))

import numpy as np
import pandas as pd
from jwst import datamodels
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits
from scipy.ndimage import median_filter
from exotedrf.utils import format_out_frames


from exotedrf.utils import parse_config, unpack_input_dir, fancyprint
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3

#####################################

#uncal_indir = 'Optimize_WASP39b/DMS_uncal/'  # Where our uncalibrated files are found.
outdir = 'pipeline_outputs_directory'
outdir_f = 'pipeline_outputs_directory/Files'
outdir_s1 = 'pipeline_outputs_directory/Stage1/'
outdir_s2 = 'pipeline_outputs_directory/Stage2/'
outdir_s3 = 'pipeline_outputs_directory/Stage3/'
outdir_s4 = 'pipeline_outputs_directory/Stage4/'

utils.verify_path('pipeline_outputs_directory')
utils.verify_path('pipeline_outputs_directory/Files')
utils.verify_path('pipeline_outputs_directory/Stage1')
utils.verify_path('pipeline_outputs_directory/Stage2')
utils.verify_path('pipeline_outputs_directory/Stage3')
utils.verify_path('pipeline_outputs_directory/Stage4')


# ----------------------------------------
# plot cost
# ----------------------------------------

def plot_cost(name_str, table_height=0.4):
    # Load data
    df = pd.read_csv(f"pipeline_outputs_directory/Files/Cost_{name_str}.txt", delimiter="\t")

    # Drop rows with non-numeric cost values
    df = df[pd.to_numeric(df["cost"], errors="coerce").notna()].reset_index(drop=True)

    # Get parameter columns (excluding duration_s and cost columns)
    param_cols = df.columns[:-2]

    # Track changed parameter and labels
    labels = []
    sweep_lines = []
    prev_row = None
    last_changed_param = None

    for idx, row in df.iterrows():
        if prev_row is None:
            value = int(row["nirspec_mask_width"]) if float(row["nirspec_mask_width"]).is_integer() else row["nirspec_mask_width"]
            label = f"nirspec_mask_width={value}"
            changed_param = "nirspec_mask_width"
        else:
            diffs = [col for col in param_cols if row[col] != prev_row[col]]
            if len(diffs) == 1:
                changed_param = diffs[0]
            elif len(diffs) >= 2:
                changed_param = diffs[1]
            else:
                changed_param = last_changed_param

            if changed_param != last_changed_param:
                sweep_lines.append(idx)

            value = row[changed_param]
            if isinstance(value, (int, float)) and float(value).is_integer():
                value = int(value)
            label = f"{changed_param}={value}"

        last_changed_param = changed_param
        labels.append(label)
        prev_row = row

    df["changed_label"] = labels

    # Highlight min-cost in each sweep
    sweep_boundaries = [0] + sweep_lines + [len(df)]
    colors = ['gray'] * len(df)
    for i in range(len(sweep_boundaries) - 1):
        start = sweep_boundaries[i]
        end = sweep_boundaries[i+1]
        min_idx = df.iloc[start:end]["cost"].idxmin()
        colors[min_idx] = 'green'

    # Best overall row
    best_row = df.loc[df["cost"].idxmin(), param_cols.tolist() + ["cost"]].copy()
    for col in best_row.index:
        val = best_row[col]
        if isinstance(val, (int, float)) and float(val).is_integer():
            best_row[col] = int(val)
    best_df = pd.DataFrame([best_row]).reset_index(drop=True)

    # Create layout with two vertical rows
    fig = plt.figure(figsize=(max(14, len(df) * 0.25), 10))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1 - table_height, table_height])
    ax_plot = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # Main plot
    ax_plot.scatter(df["changed_label"], df["cost"], color=colors)
    for x in sweep_lines:
        ax_plot.axvline(x=x - 0.5, color='gray', linestyle='--', linewidth=1)

    # simplify tick labels to just the value, not rotated
    values = [lbl.split('=')[1] for lbl in df["changed_label"]]
    ax_plot.set_xticks(range(len(df)))
    ax_plot.set_xticklabels(values, rotation=0, fontsize=8)

    # drop parameter names down by alternating offsets to avoid overlap
    ymin, ymax = ax_plot.get_ylim()
    base_y = ymin - 0.08 * (ymax - ymin)
    alt_y  = ymin - 0.15 * (ymax - ymin)
    for i, (start, end) in enumerate(zip(sweep_boundaries[:-1], sweep_boundaries[1:])):
        param_name = df.loc[start, "changed_label"].split("=")[0]
        center = (start + end - 1) / 2
        y_pos = base_y if i % 2 == 0 else alt_y
        ax_plot.text(center, y_pos, param_name, ha="center", va="top", fontsize=10)

    # expand bottom margin so parameter names stay visible
    fig.subplots_adjust(bottom=0.30)

    ax_plot.set_ylabel("Cost, (ppm)")
    ax_plot.set_title(f"Cost by Single Parameter Sweep: {name_str}")

    # prepare the table
    ax_table.axis("off")
    ax_table.text(0.5, 0.65, "Best Parameters", ha="center", va="bottom", fontsize=12)

    table = ax_table.table(
        cellText=best_df.values,
        colLabels=best_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.scale(1.0, 1.8)

    # turn off auto‐sizing globally
    table.auto_set_font_size(False)

    # let the header row still auto‐size
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(7)
        else:
            cell.set_fontsize(10)
    fig.savefig(f"pipeline_outputs_directory/Files/Cost_{name_str}.png", dpi=300, bbox_inches='tight')




# ----------------------------------------
# create filenames
# ----------------------------------------

def make_step_filenames(input_files, output_dir, step_tag):
    """
    Given a list of JWST‐style filenames, replace everything after the
    last '_' (the “step” part) with your new step_tag, and stick them
    into output_dir.
    """
    out = []
    for f in input_files:
        base = os.path.basename(f)
        # chop off the last “_something.fits” bit
        base_root = base[: base.rfind("_") ]
        new_name = f"{base_root}_{step_tag}.fits"
        out.append(os.path.join(output_dir, new_name))
    return out


# ----------------------------------------
# cost (P2P-based)
# ----------------------------------------

def cost_function(st3, baseline_ints=None, wave_range=None, tol=0.001):
    """
    Compute a combined white-light + spectral metric.

    Parameters
    ----------
    st3 : dict-like
        Must contain 'Flux' → 2D array of shape (n_int, n_wave)
        and 'Wave' → 1D array of length n_wave
    baseline_ints : list of 1 or 2 ints
        Which integrations define your baseline(s)
    wave_range : None or [min,max] or (min,max)
        If None: use all wavelengths.
        If list/tuple: pick only wavelengths within ±tol of the ends.
    tol : float
        Maximum allowed distance when matching wave_range endpoints.

    Returns
    -------
    cost : float
        w1*ptp2_white + w2*ptp2_spec
    ptp2_spec_wave : 1D np.ndarray
        The per-wavelength ptp2 values.
    """

    # unpack & weights
    w1, w2 = 0.0, 1.0
    flux = np.asarray(st3['Flux'], float)
    wave = np.asarray(st3['Wave'], float)

    # --- white-light term ---
    white      = np.nansum(flux, axis=1)
    white      = white[~np.isnan(white)]
    norm_white = white / np.median(white)
    d2_white   = 0.5*(norm_white[:-2] + norm_white[2:]) - norm_white[1:-1]
    ptp2_white = np.nanmedian(np.abs(d2_white))

    # --- spectral term (per-wavelength ptp2) ---
    wave_meds     = np.nanmedian(flux, axis=0, keepdims=True)
    norm_spec     = flux / wave_meds
    d2_spec       = 0.5*(norm_spec[:-2] + norm_spec[2:]) - norm_spec[1:-1]

    # select baseline integrations
    if baseline_ints == None:
        ptp2_spec_wave = np.nanmedian(np.abs(d2_spec), axis=0)
    
    elif len(baseline_ints) == 1:
        N = int(baseline_ints[0])
        ptp2_spec_wave = np.nanmedian(np.abs(d2_spec[:N]), axis=0)

    elif len(baseline_ints) == 2:
        Nlow, Nhigh = map(int, baseline_ints)
        low_term  = np.nanmedian(np.abs(d2_spec[:Nlow]), axis=0)
        high_term = np.nanmedian(np.abs(d2_spec[Nhigh:]), axis=0)
        ptp2_spec_wave = 0.5 * (low_term + high_term)

    else:
        raise ValueError(f"baseline_ints must be length 1 or 2, got {len(baseline_ints)}")

        # handle wave_range
    if wave_range is None:
        ptp2_spec = np.nanmedian(ptp2_spec_wave)

    elif isinstance(wave_range, (list, tuple)) and len(wave_range) == 2:
        lo, hi = wave_range

        # build a mask of finite wavelengths
        finite = np.isfinite(wave)
        if not finite.any():
            raise ValueError("All entries in wave are NaN!")

        # compute distances, forcing NaN entries to +inf
        dist_lo = np.abs(wave - lo)
        dist_lo[~finite] = np.inf
        dist_hi = np.abs(wave - hi)
        dist_hi[~finite] = np.inf

        idx_lo = int(np.argmin(dist_lo))
        idx_hi = int(np.argmin(dist_hi))

        # tolerance check
        if dist_lo[idx_lo] > tol or dist_hi[idx_hi] > tol:
            raise ValueError(f"wave_range {wave_range} not found within ±{tol}")

        # ensure idx_lo ≤ idx_hi
        i0, i1 = sorted((idx_lo, idx_hi))

        # slice and take the median of the subrange
        sub = ptp2_spec_wave[i0:i1+1]
        if np.all(np.isnan(sub)):
            raise ValueError(f"No valid ptp2_spec values in wave range {wave_range}")
        ptp2_spec = np.nanmedian(sub)

    else:
        raise ValueError("wave_range must be None or a length-2 list/tuple")

    # final cost
    cost = w1 * ptp2_white + w2 * ptp2_spec

    return cost, ptp2_spec_wave



# ----------------------------------------
# diagnostic_plot
# ----------------------------------------

def diagnostic_plot(st3, name_str, baseline_ints, outdir=outdir_f):
    """
    Generate three diagnostic plots from the stage-3 flux array:
      1. Normalized white-light curve
      2. Normalized white-light curve with error bars
      3. Normalized flux image (integrations vs spectral pixels)

    Parameters
    ----------
    st3 : dict-like
        Must contain a key 'Flux' giving a 2D array (n_int x n_pix).
    name_str : str
        Identifier used in output filenames.

    Output
    ------
    pipeline_outputs_directory/Files/
        norm_white_{name_str}.png
        flux_img_{name_str}.png
    """
    # ensure output dir exists
    os.makedirs(outdir, exist_ok=True)

    # grab the flux array
    flux = np.asarray(st3['Flux'], dtype=float)
    wave = np.asarray(st3['Wave'], dtype=float)

    # ---  white light curve ---
    white = np.nansum(flux, axis=1)
    white = white[~np.isnan(white)]
    norm_white = white / np.median(white[:100])

    if len(baseline_ints) == 1:
        N = int(baseline_ints[0])
        norm_white = white / np.median(white[:N])

    elif len(baseline_ints) == 2:
        Nlow, Nhigh = map(int, baseline_ints)
        low_term  = np.median(white[:Nlow])
        high_term = np.median(white[Nhigh:])
        median_base = 0.5 * (low_term + high_term)
        norm_white = white / median_base

    else:
        raise ValueError(f"baseline_ints must be length 1 or 2, got {len(baseline_ints)}")
    

    # 1) simple normalized white curve
    plt.figure()
    plt.plot(norm_white, marker='.')
    plt.xlabel("Integration Number")
    plt.ylabel("Normalized White Flux")
    plt.title("Normalized White-light Curve")
    plt.grid(True)
    plt.savefig(f"{outdir}/norm_white_{name_str}.png", dpi=300)
    plt.close()

    # 2) normalized flux image (integration vs. wavelength)
    plt.figure()
    img = flux / np.nanmedian(flux, axis=0, keepdims=True)
    n_int, n_pix = flux.shape
    
    # ignore NaNs when finding wavelength bounds
    finite = np.isfinite(wave)
    if not finite.any():
        raise ValueError("No finite wavelengths found!")
    wmin, wmax = wave[finite].min(), wave[finite].max()
    
    # transpose img so rows→wavelength, cols→integration
    plt.imshow(
        img.T,              # swap axes
        vmin=0.98, vmax=1.02,
        aspect='auto',
        origin='lower',
        extent=[0, n_int-1, wmin, wmax]
    )
    plt.xlabel("Integration Number")
    plt.ylabel("Wavelength (µm)")
    plt.title("Normalized Flux Image")
    plt.colorbar(label="Relative Flux")
    plt.savefig(f"{outdir}/flux_img_{name_str}.png", dpi=300)
    plt.close()



# ----------------------------------------
# covariance
# ----------------------------------------

def compute_cov_metric(random_seed=42):
    """
    Compute covariance metric for the spectrum file located in pipeline_outputs_directory/Stage3.

    Parameters:
    random_seed (int or None): Seed for reproducible noise generation (default: 42).

    Returns:
    float: Covariance metric (percent excess correlation).
    """
    # Define Stage3 output directory
    stage3_dir = os.path.join("pipeline_outputs_directory", "Stage3")
    # Find the FITS file ending with _box_spectra_fullres.fits
    pattern = os.path.join(stage3_dir, "*_box_spectra_fullres.fits")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No spectrum file matching *_box_spectra_fullres.fits found in {stage3_dir}")
    # Use the first match
    final_output_spectrum_file = matches[0]

    # Load & normalize
    spec = fits.getdata(final_output_spectrum_file, 3)
    spec /= np.nanmedian(spec[:100], axis=0)
    cov_matrix2 = np.corrcoef(spec[:100].T)

    # Prepare reproducible RNG
    rng = np.random.default_rng(random_seed)

    # Simulate noise with same per-column deviation
    dev = np.nanstd(spec[:100], axis=0)
    ss = np.empty_like(spec)
    for i in range(len(dev)):
        ss[:, i] = rng.normal(0, dev[i], spec.shape[0])
    cov_matrix = np.corrcoef(ss[:100].T)

    # Compute percent excess correlation
    cov_metric = (np.nanmean(np.abs(cov_matrix2)) / np.nanmean(np.abs(cov_matrix))) * 100 - 100
    return cov_metric

def compute_cov_metric_avg(n_seeds=10, start_seed=0):
    metrics = []
    for i in range(n_seeds):
        seed = start_seed + i
        cov = compute_cov_metric(random_seed=seed)
        metrics.append(cov)
    avg_cov = np.mean(metrics)
    return avg_cov, metrics

# ----------------------------------------
# photon noise
# ----------------------------------------

def plot_scatter_with_photon_noise(
    txtfile, rows,
    wave_range=None, smooth=None,
    spectrum_files=None, ngroup=None, baseline_ints=None,
    order=1, tframe=5.494, gain=1.6,
    style='line', ylim=None, save_path=None,
    tol=0.005
):
    """
    1) Plot P2P scatter rows (with smoothing+clipping) vs. wavelength.
    2) If `spectrum_files` is given, overplot photon-noise and 2× photon-noise (ppm).

    Args:
        txtfile (str): P2P scatter text file
        rows (list[int]): rows to plot (0-based or negative)
        wave_range (tuple, optional): (min_wave, max_wave) in same units as FITS
        smooth (int, optional): moving-average window
        spectrum_files (list[str]): FITS files (must include 'Wave' in ext 1)
        ngroup (float), baseline_ints (list[int]), order (int),
        tframe (float), gain (float): instrument parameters
        style (str): 'line' or 'scatter'
        ylim (tuple), save_path (str): plotting args
        tol (float): slack around wave_range ends for selecting pixels
    """

    # — Load scatter table —
    df = pd.read_csv(txtfile, sep=r'\s+', header=None).fillna(0)
    n_rows, n_cols = df.shape

    # — Determine valid rows —
    valid = []
    for r in rows:
        idx = r if r >= 0 else n_rows + r
        if 0 <= idx < n_rows:
            valid.append(idx)
        else:
            print(f"Warning: row {r} out of range, skipping.")
    if not valid:
        raise ValueError("No valid rows to plot.")
    labels = [str(i+1) for i in valid]

    # — Load wavelength axis —
    if not spectrum_files:
        raise ValueError("`spectrum_files` is required.")
    with fits.open(spectrum_files[0]) as hdus:
        wave_full = hdus[1].data.astype(float)
    Npix = wave_full.size

    # — Map wave_range to pixel indices via mask —
    if wave_range is not None:
        wmin, wmax = wave_range
        mask_range = (
            (wave_full >= wmin - tol) &
            (wave_full <= wmax + tol) &
            np.isfinite(wave_full)
        )
        if not mask_range.any():
            raise ValueError(
                f"Requested wave_range {wave_range} yields no finite pixels within ±{tol}."
            )
        idxs = np.where(mask_range)[0]
        start, end = idxs.min(), idxs.max()
    else:
        start, end = 0, Npix - 1

    # — Slice wavelength and mask NaNs —
    wave = wave_full[start:end+1]
    mask_wave = np.isfinite(wave)
    if not mask_wave.any():
        raise ValueError("No finite wavelengths in the selected slice.")

    plt.figure(figsize=(8, 4))

    # — Plot P2P scatter vs. wavelength —
    for idx, lab in zip(valid, labels):
        y_full = df.iloc[idx, :].values.astype(float)
        if smooth and smooth > 1:
            w = int(smooth)
            kern = np.ones(w) / w
            y_full = np.convolve(y_full, kern, mode='same')
        y = y_full[start:end+1] * 1e6  # convert to ppm
        x = wave[mask_wave]
        y = y[mask_wave]
        if style == 'line':
            plt.plot(x, y, linewidth=1.0, label=lab)
        else:
            plt.scatter(x, y, s=2, label=lab)

    # — Photon-noise overlay —
    if spectrum_files:
        base = format_out_frames(baseline_ints)
        with fits.open(spectrum_files[0]) as hdus:
            spec = hdus[3].data.astype(float) if order == 1 else hdus[7].data.astype(float)
        spec *= (tframe * gain * ngroup)
        ii = np.arange(spec.shape[-1])

        # Empirical scatter [ppm]
        scatter_vals = np.full(ii.shape, np.nan)
        for i in ii:
            pix = spec[:, i]
            denom = np.median(pix[base])
            if denom > 0 and np.isfinite(denom):
                noise = 0.5 * (pix[:-2] + pix[2:]) - pix[1:-1]
                scatter_vals[i] = np.median(np.abs(noise)) / denom
        data_ppm = median_filter(scatter_vals * 1e6, size=10)

        # Theoretical photon floor [ppm]
        med = np.median(spec[base], axis=0)
        phot = np.full_like(med, np.nan)
        good = (med > 0) & np.isfinite(med)
        phot[good] = np.sqrt(med[good]) / med[good]
        phot_ppm_filt = median_filter(phot, size=10)

        # Overlay on wavelength
        wn = wave_full[10:-10]
        valid = np.isfinite(wn) & np.isfinite(phot_ppm_filt[10:-10])
        plt.plot(wn[valid], phot_ppm_filt[10:-10][valid]*1e6,
                 'k-', lw=1.0, label='photon noise')
        plt.plot(wn[valid], 2*phot_ppm_filt[10:-10][valid]*1e6,
                 'k--', lw=1.0, label='2× photon noise')
        plt.ylabel("Precision (ppm)")
    else:
        plt.ylabel("Scatter")

    # — Finalize —
    plt.xlim(wave[mask_wave].min(), wave[mask_wave].max())
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("Wavelength (μm)")
    plt.legend(ncol=2 if spectrum_files else 1, fontsize='small')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()


# ----------------------------------------
# skip step list
# ----------------------------------------
def get_stage_skips(cfg, steps, always_skip=None, special_one_over_f=False):
    skips = set(always_skip or [])
    for step in steps:
        if cfg.get(step, 'run') == 'skip':
            if special_one_over_f and step.startswith('OneOverFStep'):
                skips.add('OneOverFStep')
            else:
                skips.add(step)
    return list(skips)





# ----------------------------------------
# main
# ----------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Coordinate‐descent optimizer for exoTEDRF Stages 1–3"
    )
    parser.add_argument(
        "--config", default="run_optimize.yaml",
        help="Path to your DMS config YAML"
    )
    
    args = parser.parse_args()

    # Load YAML config
    cfg = parse_config(args.config)

    # Set CRDS environment variables using values from YAML
    #os.environ.setdefault('CRDS_PATH', cfg.get('crds_cache_path', './crds_cache'))
    #os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
    #os.environ.setdefault('CRDS_CONTEXT', cfg.get('crds_context', 'jwst_1322.pmap'))


    baseline_ints = cfg.get('baseline_ints', [100, -100])
    name_str = cfg.get('name_tag', 'default_run')

    t0_total = time.perf_counter()
    cfg = parse_config(args.config)

    # get input data
    input_files = unpack_input_dir(
        cfg["input_dir"],
        mode=cfg["observing_mode"],
        filetag=cfg["input_filetag"],
        filter_detector=cfg["filter_detector"],
    ) 




    if not input_files:
        fancyprint(f"[WARN] No files in {cfg['input_dir']}, globbing *.fits")
        input_files = sorted(glob.glob(os.path.join(cfg["input_dir"], "*.fits")))
    if not input_files:
        raise RuntimeError(f"No FITS found in {cfg['input_dir']}")
    fancyprint(f"Using {len(input_files)} segment(s) from {cfg['input_dir']}")


    # Generate filenames for intermediate outputs
    filenames_int1 = make_step_filenames(input_files, outdir_s1, "darkcurrentstep")
    filenames_int2 = make_step_filenames(input_files, outdir_s1, "linearitystep")
    filenames_int3 = make_step_filenames(input_files, outdir_s1, "gainscalestep")
    filenames_int4 = make_step_filenames(input_files, outdir_s1, "gainscalestep")

    print("➡ will reuse these filenames_int1:", filenames_int1)
    print("➡ will reuse these filenames_int2:", filenames_int2)
    print("➡ will reuse these filenames_int3:", filenames_int3)
    print("➡ will reuse these filenames_int4:", filenames_int4)





    # --- parameter ranges ---
    # --- pick out everything prefixed "optimize_" ---
    optimize_cfg = {
        k[len("optimize_"):]: v
        for k, v in cfg.items()
        if k.startswith("optimize_")
    }

    # sweep over lists only
    param_ranges = {
        name: vals
        for name, vals in optimize_cfg.items()
        if isinstance(vals, list)
    }

    # everything else is a fixed parameter
    fixed_params = {
        name: val
        for name, val in optimize_cfg.items()
        if not isinstance(val, list)
    }

    param_order = list(param_ranges.keys())
    total_steps = sum(len(v) for v in param_ranges.values())

    stage1_keys   = [
        "rejection_threshold", "time_rejection_threshold","time_window",
        "nirspec_mask_width",   "soss_inner_mask_width",
        "soss_outer_mask_width","miri_drop_groups",
    ]
    stage2_keys   = [
        "space_thresh", "time_thresh",
        "miri_trace_width", "miri_background_width",
    ]
    stage3_keys   = ["extract_width"]

    # initialize current to medians
    current = {k: int(np.median(v)) for k,v in param_ranges.items()}
    current.update(fixed_params)


    # open global log
    logf = open(f"pipeline_outputs_directory/Files/Cost_{name_str}.txt","w")
    logc = open(f"pipeline_outputs_directory/Files/cov_{name_str}.txt","w")
    logs  = open(f"pipeline_outputs_directory/Files/Scatter_{name_str}.txt", "w")
    logf.write("\t".join(param_order)+"\tduration_s\tcost\n")

    count = 1
    
    # coordinate descent
    for key in param_order:
        fancyprint(
            f"→ Optimizing {key} "
            f"(fixed-other={{{', '.join(f'{k}:{current[k]}' for k in current if k!=key)}}})"
        )

        best_val = current[key]
        best_cost = None

        for trial in param_ranges[key]:
            fancyprint(f"Step {count}/{total_steps}: {key}={trial}")
            trial_params = {**current, key: trial}
            run_cfg = cfg.copy()
            run_cfg.update(trial_params)

            

            t0 = time.perf_counter()

            print('\n\n\n ####### \n trial  = ', trial)
            print('key  = ', key)
            print('param_order  = ', param_order)
            print('param_ranges[key]  = ', param_ranges[key])
            print('current  = ', current)

            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} starting {key}={trial}",
                "\n############################################\n",
                flush=True
            )

            # split out args
            s1_args = {}
            s2_args = {}
            s3_args = {}

            # inherit for JumpStep 
            if "time_window" in trial_params:
                s1_args["JumpStep"] = {"time_window":trial_params["time_window"]}

            # BadPixStep overrides
            badpix = {}
            if "box_size" in trial_params:
                badpix["box_size"] = trial_params["box_size"]
            if "window_size" in trial_params:
                badpix["window_size"] = trial_params["window_size"]
            if badpix:
                s2_args["BadPixStep"] = badpix

            stage1_steps = [
                'DQInitStep', 'EmiCorrStep', 'SaturationStep', 'ResetStep', 'SuperBiasStep',
                'RefPixStep', 'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep',
                'JumpStep', 'RampFitStep', 'GainScaleStep'
            ]

            stage2_steps = [
                'AssignWCSStep', 'Extract2DStep', 'SourceTypeStep', 'WaveCorrStep',
                'FlatFieldStep', 'BackgroundStep', 'OneOverFStep_int',
                'BadPixStep', 'PCAReconstructStep', 'TracingStep'
            ]
 
            stage3_steps = []

            if best_cost is None:
                # ===== Stage 1 =====
                always_skip1 = []
                stage1_skip = get_stage_skips(
                    cfg,
                    stage1_steps,
                    always_skip=always_skip1,
                    special_one_over_f=True
                )

                if 1 in cfg['run_stages']:
                    stage1_results = run_stage1(
                        input_files,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg['soss_background_file'],
                        baseline_ints=run_cfg['baseline_ints'],
                        oof_method=run_cfg['oof_method'],
                        superbias_method=run_cfg['superbias_method'],
                        soss_timeseries=run_cfg['soss_timeseries'],
                        soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                        save_results=run_cfg['save_results'],
                        pixel_masks=run_cfg['outlier_maps'],
                        force_redo=True, 
                        flag_up_ramp=run_cfg['flag_up_ramp'],
                        rejection_threshold=run_cfg['jump_threshold'],
                        flag_in_time=run_cfg['flag_in_time'],
                        time_rejection_threshold=run_cfg['time_jump_threshold'],
                        output_tag=run_cfg['output_tag'],
                        skip_steps=stage1_skip,
                        do_plot=False, 
                        soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                        soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                        nirspec_mask_width=run_cfg['nirspec_mask_width'],
                        centroids=run_cfg['centroids'],
                        hot_pixel_map=run_cfg['hot_pixel_map'],
                        miri_drop_groups=run_cfg['miri_drop_groups'],
                        **run_cfg.get('stage1_kwargs', {}),
                        **s1_args
                    )
                else:
                    stage1_results = input_files

                # ===== Stage 2 =====
                always_skip2 = []
                stage2_skip = get_stage_skips(
                    cfg,
                    stage2_steps,
                    always_skip=always_skip2,
                    special_one_over_f=False
                ) 

                if 2 in cfg['run_stages']:
                    stage2_results = run_stage2(
                        stage1_results,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg['soss_background_file'],
                        baseline_ints=run_cfg['baseline_ints'],
                        save_results=run_cfg['save_results'],
                        force_redo=True, 
                        space_thresh=run_cfg['space_outlier_threshold'],
                        time_thresh=run_cfg['time_outlier_threshold'],
                        remove_components=run_cfg['remove_components'],
                        pca_components=run_cfg['pca_components'],
                        soss_timeseries=run_cfg['soss_timeseries'],
                        soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                        oof_method=run_cfg['oof_method'],
                        output_tag=run_cfg['output_tag'],
                        smoothing_scale=run_cfg['smoothing_scale'],
                        skip_steps=stage2_skip,
                        generate_lc=run_cfg['generate_lc'],
                        soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                        soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                        nirspec_mask_width=run_cfg['nirspec_mask_width'],
                        pixel_masks=run_cfg['outlier_maps'],
                        generate_order0_mask=run_cfg['generate_order0_mask'],
                        f277w=run_cfg['f277w'],
                        do_plot=False,
                        centroids=run_cfg['centroids'],
                        miri_trace_width=run_cfg['miri_trace_width'],
                        miri_background_width=run_cfg['miri_background_width'],
                        miri_background_method=run_cfg['miri_background_method'],
                        **run_cfg.get('stage2_kwargs', {}),
                        **s2_args
                    )
                    stage2_results, centroids = stage2_results
                else:
                    stage2_results = stage1_results
                    centroids = cfg['centroids']

                # ===== Stage 3 =====
                always_skip3 = []
                stage3_skip = get_stage_skips(
                    cfg,
                    stage3_steps,
                    always_skip=always_skip3,
                    special_one_over_f=False
                )

                if 3 in cfg['run_stages']:
                    this_centroid = cfg['centroids'] if cfg['centroids'] is not None else centroids
                    stage3_results = run_stage3(
                        stage2_results,
                        save_results=run_cfg['save_results'],
                        force_redo=True,
                        extract_method=run_cfg['extract_method'],
                        soss_specprofile=run_cfg['soss_specprofile'],
                        centroids=this_centroid,
                        extract_width=run_cfg['extract_width'],
                        st_teff=run_cfg['st_teff'],
                        st_logg=run_cfg['st_logg'],
                        st_met=run_cfg['st_met'],
                        planet_letter=run_cfg['planet_letter'],
                        output_tag=run_cfg['output_tag'],
                        do_plot=False,
                        skip_steps=stage3_skip,
                        **run_cfg.get('stage3_kwargs', {}),
                        **s3_args
                    )
                else:
                    stage3_results = stage2_results

            
            else:
                if key == 'nirspec_mask_width':
                    # --- Stage 1 on darkcurrent‐stepped files ---
                    always_skip1 = ['DQInitStep', 'SaturationStep', 'DarkCurrentStep']
                    stage1_skip = get_stage_skips(
                        cfg,
                        stage1_steps,
                        always_skip=always_skip1,
                        special_one_over_f=True
                    )
                    if 1 in cfg['run_stages']:
                        stage1_results = run_stage1(
                            filenames_int1,
                            mode=run_cfg['observing_mode'],
                            soss_background_model=run_cfg['soss_background_file'],
                            baseline_ints=run_cfg['baseline_ints'],
                            oof_method=run_cfg['oof_method'],
                            superbias_method=run_cfg['superbias_method'],
                            soss_timeseries=run_cfg['soss_timeseries'],
                            soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                            save_results=run_cfg['save_results'],
                            pixel_masks=run_cfg['outlier_maps'],
                            force_redo=True,
                            flag_up_ramp=run_cfg['flag_up_ramp'],
                            rejection_threshold=run_cfg['jump_threshold'],
                            flag_in_time=run_cfg['flag_in_time'],
                            time_rejection_threshold=run_cfg['time_jump_threshold'],
                            output_tag=run_cfg['output_tag'],
                            skip_steps=stage1_skip,
                            do_plot=False,
                            soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                            soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                            nirspec_mask_width=run_cfg['nirspec_mask_width'],
                            centroids=run_cfg['centroids'],
                            hot_pixel_map=run_cfg['hot_pixel_map'],
                            miri_drop_groups=run_cfg['miri_drop_groups'],
                            **run_cfg.get('stage1_kwargs', {}),
                            **s1_args
                        )
                    else:
                        stage1_results = filenames_int1

                    # --- Stage 2 on those results ---
                    always_skip2 = []
                    stage2_skip = get_stage_skips(
                        cfg,
                        stage2_steps,
                        always_skip=always_skip2,
                        special_one_over_f=False
                    )
                    if 2 in cfg['run_stages']:
                        stage2_results, centroids = run_stage2(
                            stage1_results,
                            mode=run_cfg['observing_mode'],
                            soss_background_model=run_cfg['soss_background_file'],
                            baseline_ints=run_cfg['baseline_ints'],
                            save_results=run_cfg['save_results'],
                            force_redo=True,
                            space_thresh=run_cfg['space_outlier_threshold'],
                            time_thresh=run_cfg['time_outlier_threshold'],
                            remove_components=run_cfg['remove_components'],
                            pca_components=run_cfg['pca_components'],
                            soss_timeseries=run_cfg['soss_timeseries'],
                            soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                            oof_method=run_cfg['oof_method'],
                            output_tag=run_cfg['output_tag'],
                            smoothing_scale=run_cfg['smoothing_scale'],
                            skip_steps=stage2_skip,
                            generate_lc=run_cfg['generate_lc'],
                            soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                            soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                            nirspec_mask_width=run_cfg['nirspec_mask_width'],     
                            pixel_masks=run_cfg['outlier_maps'],
                            generate_order0_mask=run_cfg['generate_order0_mask'],
                            f277w=run_cfg['f277w'],
                            do_plot=False,
                            centroids=run_cfg['centroids'],
                            miri_trace_width=run_cfg['miri_trace_width'],
                            miri_background_width=run_cfg['miri_background_width'],
                            miri_background_method=run_cfg['miri_background_method'],
                            **run_cfg.get('stage2_kwargs', {}),
                            **s2_args
                        )
                    else:
                        stage2_results = stage1_results
                        centroids = cfg['centroids']

                    # --- Stage 3 on those results ---
                    always_skip3 = []
                    stage3_skip = get_stage_skips(
                        cfg,
                        stage3_steps,
                        always_skip=always_skip3,
                        special_one_over_f=False
                    )
                    if 3 in cfg['run_stages']:
                        this_centroid = cfg['centroids'] if cfg['centroids'] is not None else centroids
                        stage3_results = run_stage3(
                            stage2_results,
                            save_results=run_cfg['save_results'],
                            force_redo=True,
                            extract_method=run_cfg['extract_method'],
                            soss_specprofile=run_cfg['soss_specprofile'],
                            centroids=this_centroid,
                            extract_width=run_cfg['extract_width'],
                            st_teff=run_cfg['st_teff'],
                            st_logg=run_cfg['st_logg'],
                            st_met=run_cfg['st_met'],
                            planet_letter=run_cfg['planet_letter'],
                            output_tag=run_cfg['output_tag'],
                            do_plot=False,
                            skip_steps=stage3_skip,
                            **run_cfg.get('stage3_kwargs', {}),
                            **s3_args
                        )
                    else:
                        stage3_results = stage2_results           

                    
                elif key in ('time_jump_threshold','time_rejection_threshold', 'time_window'):

                    # --- Stage 1 on the “linearized” intermediates ---
                    always_skip1 = ['DQInitStep', 'SaturationStep', 'DarkCurrentStep',
                                    'OneOverFStep', 'LinearityStep']
                    stage1_skip = get_stage_skips(
                        cfg,
                        stage1_steps,
                        always_skip=always_skip1,
                        special_one_over_f=True
                    )

                    if 1 in cfg['run_stages']:
                        stage1_results = run_stage1(
                            filenames_int2,
                            mode=run_cfg['observing_mode'],
                            soss_background_model=run_cfg['soss_background_file'],
                            baseline_ints=run_cfg['baseline_ints'],
                            oof_method=run_cfg['oof_method'],
                            superbias_method=run_cfg['superbias_method'],
                            soss_timeseries=run_cfg['soss_timeseries'],
                            soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                            save_results=run_cfg['save_results'],
                            pixel_masks=run_cfg['outlier_maps'],
                            force_redo=True,
                            flag_up_ramp=run_cfg['flag_up_ramp'],
                            rejection_threshold=run_cfg['jump_threshold'],
                            flag_in_time=run_cfg['flag_in_time'],
                            time_rejection_threshold=run_cfg['time_jump_threshold'],
                            output_tag=run_cfg['output_tag'],
                            skip_steps=stage1_skip,
                            do_plot=False,
                            soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                            soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                            nirspec_mask_width=run_cfg['nirspec_mask_width'],
                            centroids=run_cfg['centroids'],
                            hot_pixel_map=run_cfg['hot_pixel_map'],
                            miri_drop_groups=run_cfg['miri_drop_groups'],
                            **run_cfg.get('stage1_kwargs', {}),
                            **s1_args
                        )
                    else:
                        stage1_results = filenames_int2

                    # --- Stage 2 on those results ---
                    always_skip2 = []
                    stage2_skip = get_stage_skips(
                        cfg,
                        stage2_steps,
                        always_skip=always_skip2,
                        special_one_over_f=False
                    )

                    if 2 in cfg['run_stages']:
                        stage2_results, centroids = run_stage2(
                            stage1_results,
                            mode=run_cfg['observing_mode'],
                            soss_background_model=run_cfg['soss_background_file'],
                            baseline_ints=run_cfg['baseline_ints'],
                            save_results=run_cfg['save_results'],
                            force_redo=True,
                            space_thresh=run_cfg['space_outlier_threshold'],
                            time_thresh=run_cfg['time_outlier_threshold'],
                            remove_components=run_cfg['remove_components'],
                            pca_components=run_cfg['pca_components'],
                            soss_timeseries=run_cfg['soss_timeseries'],
                            soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                            oof_method=run_cfg['oof_method'],
                            output_tag=run_cfg['output_tag'],
                            smoothing_scale=run_cfg['smoothing_scale'],
                            skip_steps=stage2_skip,
                            generate_lc=run_cfg['generate_lc'],
                            soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                            soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                            nirspec_mask_width=run_cfg['nirspec_mask_width'],
                            pixel_masks=run_cfg['outlier_maps'],
                            generate_order0_mask=run_cfg['generate_order0_mask'],
                            f277w=run_cfg['f277w'],
                            do_plot=False,
                            centroids=run_cfg['centroids'],
                            miri_trace_width=run_cfg['miri_trace_width'],
                            miri_background_width=run_cfg['miri_background_width'],
                            miri_background_method=run_cfg['miri_background_method'],
                            **run_cfg.get('stage2_kwargs', {}),
                            **s2_args
                        )
                    else:
                        stage2_results = stage1_results
                        centroids = cfg['centroids']

                    # --- Stage 3 on those results ---
                    always_skip3 = []
                    stage3_skip = get_stage_skips(
                        cfg,
                        stage3_steps,
                        always_skip=always_skip3,
                        special_one_over_f=False
                    )

                    if 3 in cfg['run_stages']:
                        this_centroid = cfg['centroids'] if cfg['centroids'] is not None else centroids
                        stage3_results = run_stage3(
                            stage2_results,
                            save_results=run_cfg['save_results'],
                            force_redo=True,
                            extract_method=run_cfg['extract_method'],
                            soss_specprofile=run_cfg['soss_specprofile'],
                            centroids=this_centroid,
                            extract_width=run_cfg['extract_width'],
                            st_teff=run_cfg['st_teff'],
                            st_logg=run_cfg['st_logg'],
                            st_met=run_cfg['st_met'],
                            planet_letter=run_cfg['planet_letter'],
                            output_tag=run_cfg['output_tag'],
                            do_plot=False,
                            skip_steps=stage3_skip,
                            **run_cfg.get('stage3_kwargs', {}),
                            **s3_args
                        )
                    else:
                        stage3_results = stage2_results

                


                elif key in ('space_outlier_threshold','space_thresh','time_outlier_threshold', 'time_thresh', 'box_size', 'window_size'):
                    # Stage 2 on precomputed Stage-1 intermediates (filenames_int3)
                    always_skip2 = []
                    stage2_skip = get_stage_skips(
                        cfg,
                        stage2_steps,
                        always_skip=always_skip2,
                        special_one_over_f=False
                    )

                    if 2 in cfg['run_stages']:
                        stage2_results, centroids = run_stage2(
                            filenames_int3,
                            mode=run_cfg['observing_mode'],
                            baseline_ints=run_cfg['baseline_ints'],
                            save_results=run_cfg['save_results'],
                            force_redo=True,
                            space_thresh=run_cfg['space_outlier_threshold'],
                            time_thresh=run_cfg['time_outlier_threshold'],
                            remove_components=run_cfg['remove_components'],
                            pca_components=run_cfg['pca_components'],
                            soss_timeseries=run_cfg['soss_timeseries'],
                            soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                            oof_method=run_cfg['oof_method'],
                            output_tag=run_cfg['output_tag'],
                            smoothing_scale=run_cfg['smoothing_scale'],
                            skip_steps=stage2_skip,
                            generate_lc=run_cfg['generate_lc'],
                            soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                            soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                            nirspec_mask_width=run_cfg['nirspec_mask_width'],
                            pixel_masks=run_cfg['outlier_maps'],
                            generate_order0_mask=run_cfg['generate_order0_mask'],
                            f277w=run_cfg['f277w'],
                            do_plot=False,
                            centroids=run_cfg['centroids'],
                            miri_trace_width=run_cfg['miri_trace_width'],
                            miri_background_width=run_cfg['miri_background_width'],
                            miri_background_method=run_cfg['miri_background_method'],
                            **run_cfg.get('stage2_kwargs', {}),
                            **s2_args
                        )
                    else:
                        stage2_results = filenames_int3
                        centroids = cfg['centroids']

                    # Stage 3
                    always_skip3 = []
                    stage3_skip = get_stage_skips(
                        cfg,
                        stage3_steps,
                        always_skip=always_skip3,
                        special_one_over_f=False
                    )

                    if 3 in cfg['run_stages']:
                        this_centroid = cfg['centroids'] if cfg['centroids'] is not None else centroids
                        stage3_results = run_stage3(
                            stage2_results,
                            save_results=run_cfg['save_results'],
                            force_redo=True,
                            extract_method=run_cfg['extract_method'],
                            soss_specprofile=run_cfg['soss_specprofile'],
                            centroids=this_centroid,
                            extract_width=run_cfg['extract_width'],
                            st_teff=run_cfg['st_teff'],
                            st_logg=run_cfg['st_logg'],
                            st_met=run_cfg['st_met'],
                            planet_letter=run_cfg['planet_letter'],
                            output_tag=run_cfg['output_tag'],
                            do_plot=False,
                            skip_steps=stage3_skip,
                            **run_cfg.get('stage3_kwargs', {}),
                            **s3_args
                        )
                    else:
                        stage3_results = stage2_results


                elif key == 'extract_width':
                    # Stage 2 on precomputed Stage-1 intermediates (filenames_int4)
                    always_skip2 = []
                    stage2_skip = get_stage_skips(
                        cfg,
                        stage2_steps,
                        always_skip=always_skip2,
                        special_one_over_f=False
                    )

                    if 2 in cfg['run_stages']:
                        stage2_results, centroids = run_stage2(
                            filenames_int4,
                            mode=run_cfg['observing_mode'],
                            baseline_ints=run_cfg['baseline_ints'],
                            save_results=run_cfg['save_results'],
                            force_redo=True,
                            space_thresh=run_cfg['space_outlier_threshold'],
                            time_thresh=run_cfg['time_outlier_threshold'],
                            remove_components=run_cfg['remove_components'],
                            pca_components=run_cfg['pca_components'],
                            soss_timeseries=run_cfg['soss_timeseries'],
                            soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                            oof_method=run_cfg['oof_method'],
                            output_tag=run_cfg['output_tag'],
                            smoothing_scale=run_cfg['smoothing_scale'],
                            skip_steps=stage2_skip,
                            generate_lc=run_cfg['generate_lc'],
                            soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                            soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                            nirspec_mask_width=run_cfg['nirspec_mask_width'],
                            pixel_masks=run_cfg['outlier_maps'],
                            generate_order0_mask=run_cfg['generate_order0_mask'],
                            f277w=run_cfg['f277w'],
                            do_plot=False,
                            centroids=run_cfg['centroids'],
                            miri_trace_width=run_cfg['miri_trace_width'],
                            miri_background_width=run_cfg['miri_background_width'],
                            miri_background_method=run_cfg['miri_background_method'],
                            **run_cfg.get('stage2_kwargs', {}),
                            **s2_args
                        )
                    else:
                        stage2_results = filenames_int4
                        centroids = cfg['centroids']

                    # Stage 3 with trial-specific extract_width
                    always_skip3 = []
                    stage3_skip = get_stage_skips(
                        cfg,
                        stage3_steps,
                        always_skip=always_skip3,
                        special_one_over_f=False
                    )

                    if 3 in cfg['run_stages']:
                        this_centroid = cfg['centroids'] if cfg['centroids'] is not None else centroids
                        stage3_results = run_stage3(
                            stage2_results,
                            save_results=run_cfg['save_results'],
                            force_redo=True,
                            extract_method=run_cfg['extract_method'],
                            soss_specprofile=run_cfg['soss_specprofile'],
                            centroids=this_centroid,
                            extract_width=run_cfg['extract_width'],
                            st_teff=run_cfg['st_teff'],
                            st_logg=run_cfg['st_logg'],
                            st_met=run_cfg['st_met'],
                            planet_letter=run_cfg['planet_letter'],
                            output_tag=run_cfg['output_tag'],
                            do_plot=False,
                            skip_steps=stage3_skip,
                            **run_cfg.get('stage3_kwargs', {}),
                            **s3_args
                        )
                    else:
                        stage3_results = stage2_results

            st2, st3 = stage2_results, stage3_results


            

            cost, scatter = cost_function(st3, baseline_ints=baseline_ints)
            
            covariance, all_covs = compute_cov_metric_avg(n_seeds=10, start_seed=0)


            dt = time.perf_counter() - t0
            fancyprint(f"→ cost = {cost:.12f} in {dt:.1f}s")

            # log it
            logf.write(
                "\t".join(str(trial_params[k]) for k in param_order)
                + f"\t{dt:.1f}\t{cost:.12f}\n"
            )

            line = " ".join(f"{x:.10g}" for x in scatter)  
            logs.write(line + "\n")
            logc.write(f"{covariance:.10f}\n")

            if best_cost is None or cost < best_cost:
                best_cost, best_val = cost, trial
                diagnostic_plot(st3, name_str, baseline_ints=baseline_ints, outdir=outdir_f)

            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} completed (dt={dt:.1f}s)",
                "\n############################################\n",
                flush=True
            )         

            count += 1

        current[key] = best_val
        fancyprint(f"Best {key} = {best_val} (cost={best_cost:.12f})")

    # total runtime
    t1 = time.perf_counter() - t0_total
    h, m = divmod(int(t1),3600)
    m, s = divmod(m,60)
    fancyprint(f"TOTAL runtime: {h}h {m:02d}min {s:04.1f}s")
    logf.close()
    logs.close()
    logc.close()

    fancyprint("=== FINAL OPTIMUM ===")
    fancyprint(current)
    plot_cost(name_str)

    cost_file = os.path.join(outdir_f, f"Cost_{name_str}.txt")
    df_cost  = pd.read_csv(cost_file, sep="\t")
    # locate the index of the minimum cost
    idx_min  = df_cost['cost'].idxmin()
    best_row = df_cost.loc[idx_min]

    # build a dict of only your swept parameters
    best_params = {
        col: int(best_row[col]) if float(best_row[col]).is_integer() else best_row[col]
        for col in param_order
    }

    fancyprint(f"Global best from cost table (row {idx_min}): {best_params}")

    # merge into a fresh config
    final_cfg = cfg.copy()
    final_cfg.update(best_params)

    # --------------------------------------------------------------------
    # Fast final validation: only Stage 2 + Stage 3 on precomputed Stage 1
    # --------------------------------------------------------------------
    fancyprint("Running fast final validation: only Stage 2 + Stage 3…")

    # Stage 2 on your precomputed Stage-1 outputs (filenames_int4)
    stage2_skip = []  
    stage2_results, centroids = run_stage2(
        filenames_int4,
        mode=final_cfg['observing_mode'],
        baseline_ints=final_cfg['baseline_ints'],
        save_results=final_cfg['save_results'],
        force_redo=True,
        space_thresh=final_cfg['space_outlier_threshold'],
        time_thresh=final_cfg['time_outlier_threshold'],
        remove_components=final_cfg['remove_components'],
        pca_components=final_cfg['pca_components'],
        soss_timeseries=final_cfg['soss_timeseries'],
        soss_timeseries_o2=final_cfg['soss_timeseries_o2'],
        oof_method=final_cfg['oof_method'],
        output_tag=final_cfg['output_tag'],
        smoothing_scale=final_cfg['smoothing_scale'],
        skip_steps=stage2_skip,
        generate_lc=final_cfg['generate_lc'],
        soss_inner_mask_width=final_cfg['soss_inner_mask_width'],
        soss_outer_mask_width=final_cfg['soss_outer_mask_width'],
        nirspec_mask_width=final_cfg['nirspec_mask_width'],
        pixel_masks=final_cfg['outlier_maps'],
        generate_order0_mask=final_cfg['generate_order0_mask'],
        f277w=final_cfg['f277w'],
        do_plot=True,
        centroids=final_cfg['centroids'],
        miri_trace_width=final_cfg['miri_trace_width'],
        miri_background_width=final_cfg['miri_background_width'],
        miri_background_method=final_cfg['miri_background_method'],
        **final_cfg.get('stage2_kwargs', {})
    )

    # Stage 3 with your best extract_width and other Stage-3 params
    final_centroids = final_cfg['centroids'] if final_cfg['centroids'] is not None else centroids
    stage3_results = run_stage3(
        stage2_results,
        save_results=final_cfg['save_results'],
        force_redo=True,
        extract_method=final_cfg['extract_method'],
        soss_specprofile=final_cfg['soss_specprofile'],
        centroids=final_centroids,
        extract_width=final_cfg['extract_width'],
        st_teff=final_cfg['st_teff'],
        st_logg=final_cfg['st_logg'],
        st_met=final_cfg['st_met'],
        planet_letter=final_cfg['planet_letter'],
        output_tag=final_cfg['output_tag'],
        do_plot=True,
        skip_steps=[],
        **final_cfg.get('stage3_kwargs', {})
    )

    fancyprint("Final validation complete.")

    # visualize the best scatter & photon floor
    outfile = os.path.join(outdir_f, f"Scatter_{name_str}.txt")
    specfile = glob.glob(os.path.join(outdir_s3, "*_box_spectra_fullres.fits"))[0]
    best_idx  = pd.read_csv(os.path.join(outdir_f, f"Cost_{name_str}.txt"), sep="\t")['cost'].idxmin()

    obs = cfg['observing_mode'].lower()
    wave_range     = cfg.get('wave_range_plot', None)
    ylim           = cfg.get('ylim_plot',      None)

    # pick instrument-specific photon-noise params
    if 'miri' in obs:
        ngroup, tframe, gain = 10, 5.494, 1.6
    elif 'nirspec' in obs:
        ngroup, tframe, gain = 70, 0.902, 1.0
    elif 'niriss' in obs:
        ngroup, tframe, gain = 50, 1.46, 2.25
    else:
        raise ValueError(f"Unrecognized observing_mode: {cfg['observing_mode']}")

    plot_scatter_with_photon_noise(
        txtfile=outfile,
        rows=[best_idx],
        wave_range=wave_range,
        smooth=21,
        spectrum_files=[specfile],
        ngroup=ngroup,
        baseline_ints=[100],
        order=1,
        tframe=tframe,
        gain=gain,
        ylim = ylim,
        style="line",
        save_path=os.path.join(outdir_f, "scatter_vs_photon_noise.png"),
        tol=0.005
    )




if __name__ == "__main__":
    main()
