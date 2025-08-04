#!/usr/bin/env python3

import os

os.environ.setdefault('CRDS_PATH', os.path.join(os.getcwd(),'Optimize_WASP39b','crds_cache'))
os.environ.setdefault('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
os.environ.setdefault('CRDS_CONTEXT',    'jwst_1322.pmap')

import glob
import time
import argparse
import numpy as np
import pandas as pd
from jwst import datamodels
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits

from exotedrf import utils
from exotedrf.utils import parse_config, unpack_input_dir, fancyprint
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3

#########################################

baseline_ints = [150 ]
cost_range = 'all' # Acceptable: baseline_ints, 'all', (lo,hi), [N], or [N1,N2]

name_str = 'P2P_spec_whole_V3'
uncal_indir = 'Optimize_WASP39b/DMS_uncal/'  # Where our uncalibrated files are found.
outdir_s1 = 'pipeline_outputs_directory/Stage1/'
outdir_s2 = 'pipeline_outputs_directory/Stage2/'
outdir_s3 = 'pipeline_outputs_directory/Stage3/'

utils.verify_path('pipeline_outputs_directory')
utils.verify_path('pipeline_outputs_directory/Files')
utils.verify_path('pipeline_outputs_directory/Stage1')
utils.verify_path('pipeline_outputs_directory/Stage2')
utils.verify_path('pipeline_outputs_directory/Stage3')
utils.verify_path('pipeline_outputs_directory/Stage4')


filenames = [uncal_indir+'jw01366003001_04101_00001-seg001_nrs1_uncal.fits',
            uncal_indir+'jw01366003001_04101_00001-seg002_nrs1_uncal.fits',
            uncal_indir+'jw01366003001_04101_00001-seg003_nrs1_uncal.fits']


# ----------------------------------------
# plot cost
# ----------------------------------------

def plot_cost(name_str, table_height=0.4):
    # Load data
    df = pd.read_csv(f"pipeline_outputs_directory/Files/Cost_{name_str}.txt", delimiter="\t")

    # Drop rows with non-numeric cost values
    df = df[pd.to_numeric(df["cost"], errors="coerce").notna()].reset_index(drop=True)

    # Get parameter columns (excluding duration_s and cost columns)
    param_cols = df.columns[:-3]

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

# now for stage1’s dark‐current outputs:
filenames_int1 = make_step_filenames(filenames, outdir_s1, "darkcurrentstep")
# for stage1’s linearity outputs:
filenames_int2 = make_step_filenames(filenames, outdir_s1, "linearitystep")
# stage‐1 → gainscalestep  (used when skipping later stage1 steps)
filenames_int3 = make_step_filenames(filenames, outdir_s1, "gainscalestep")
# for stage2’s badpixstep outputs you’d do:
filenames_int4 = make_step_filenames(filenames, outdir_s2, "badpixstep")




# ----------------------------------------
# cost (P2P-based)
# ----------------------------------------

def cost_function(st3, cost_range=baseline_ints):
    """
    Compute a combined white-light + spectral “spikiness” metric.

    Parameters
    ----------
    st3 : dict-like
        Must contain 'Flux' → 2D array of shape (n_int, n_wave).
    cost_range : one of
        • list of one int [N]: use ptp2 over ptp2_spec_wave[0:N]
        • list of two ints [N1, N2]: use ptp2 over first N1 *and* last N2 pixels. N1 positive, N2 negative
        • tuple of two ints (lo, hi): use ptp2_spec_wave[lo:hi]
        • 'all' : use entire ptp2_spec_wave
        • None : same as list of ints in baseline_ints

    Returns
    -------
    cost : float
        w1*ptp2_white + w2*ptp2_spec
    ptp2_spec_wave : 1D np.ndarray
        The per-wavelength ptp2 values.
    """


    w1, w2 = 0.0, 1.0
    flux = np.asarray(st3['Flux'], float)

    # --- white-light term ---
    white = np.nansum(flux, axis=1)
    white = white[~np.isnan(white)]
    norm_white = white / np.median(white)
    d2_white = 0.5*(norm_white[:-2] + norm_white[2:]) - norm_white[1:-1]
    ptp2_white = np.nanmedian(np.abs(d2_white))

    # --- spectral term (ptp2 per wavelength) ---
    wave_meds = np.nanmedian(flux, axis=0, keepdims=True)
    norm_spec = flux / wave_meds
    d2_spec = 0.5*(norm_spec[:-2] + norm_spec[2:]) - norm_spec[1:-1]
    ptp2_spec_wave = np.nanmedian(np.abs(d2_spec), axis=0)

   
    cr = cost_range
    # default to baseline_ints if None
    if cr is None:
        cr = baseline_ints

    if isinstance(cr, str):
        if cr == 'all':
            ptp2_spec = np.nanmedian(ptp2_spec_wave)
        else:
            raise ValueError(f"Invalid cost_range={cr!r}. Acceptable: baseline_ints, 'all', (lo,hi), [N], or [N1,N2].")

    elif isinstance(cr, tuple) and len(cr) == 2:
        lo, hi = cr
        ptp2_spec = np.nanmedian(ptp2_spec_wave[lo:hi])

    elif isinstance(cr, list):
        if len(cr) == 1:
            N = cr[0]
            ptp2_spec = np.nanmedian(ptp2_spec_wave[:N])
        elif len(cr) == 2:
            N1, N2 = cr
            first_med = np.nanmedian(ptp2_spec_wave[:N1])
            last_med  = np.nanmedian(ptp2_spec_wave[N2:])
            ptp2_spec = 0.5*(first_med + last_med)
        else:
            raise ValueError(f"Invalid cost_range={cr!r}. Acceptable: baseline_ints, 'all', (lo,hi), [N], or [N1,N2].")

    else:
        raise ValueError(f"Invalid cost_range={cr!r}. Acceptable: baseline_ints, 'all', (lo,hi), [N], or [N1,N2].")

    cost = w1 * ptp2_white + w2 * ptp2_spec
    return cost, ptp2_spec_wave



# ----------------------------------------
# diagnostic_plot
# ----------------------------------------

def diagnostic_plot(st3, name_str):
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
    outdir = "pipeline_outputs_directory/Files"
    os.makedirs(outdir, exist_ok=True)

    # grab the flux array
    flux = np.asarray(st3['Flux'], dtype=float)

    # ---  white light curve ---
    white = np.nansum(flux, axis=1)
    white = white[~np.isnan(white)]
    norm_white = white / np.median(white)

    # 1) simple normalized white curve
    plt.figure()
    plt.plot(norm_white, marker='.')
    plt.xlabel("Integration Number")
    plt.ylabel("Normalized White Flux")
    plt.title("Normalized White-light Curve")
    plt.grid(True)
    plt.savefig(f"{outdir}/norm_white_{name_str}.png", dpi=300)
    plt.close()

    # 3) normalized flux image (integrations vs. spectral pixels)
    plt.figure()
    img = flux / np.nanmedian(flux, axis=0, keepdims=True)
    plt.imshow(img, aspect='auto', vmin=0.98, vmax=1.02, origin='lower')
    plt.xlabel("Spectral Pixel")
    plt.ylabel("Integration Number")
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



# ----------------------------------------
# main
# ----------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Coordinate‐descent optimizer for exoTEDRF Stages 1–3"
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


    # --- parameter ranges ---
    param_ranges = {}
    if args.instrument == "NIRISS":
        param_ranges.update({
            "soss_inner_mask_width": [20, 40, 80],
            "soss_outer_mask_width": [35, 70, 140],
            "jump_threshold":         [5, 15, 30],
            "time_jump_threshold":    [3, 10, 20],
        })
    elif args.instrument == "NIRSPEC":
        param_ranges.update({
            "nirspec_mask_width":        list(range(10,25,2)),
            "time_rejection_threshold":  list(range(5,11,1)),
            "time_window":               list(range(3,12,2)),
        })
    else:  # MIRI
        param_ranges.update({
            # add MIRI‐specific ranges here if desired
        })

    # common parameters
    param_ranges.update({
        "space_thresh":  list(range(5,16,1)),
        "time_thresh":   list(range(5,16,1)),
        "box_size":      list(range(2,9,1)),
        "window_size":   list(range(3,12,2)),
        "extract_width": list(range(3,11,1)),
    })

    param_order   = list(param_ranges.keys())
    total_steps   = sum(len(v) for v in param_ranges.values())

    stage1_keys   = [
        "rejection_threshold", "time_rejection_threshold",
        "nirspec_mask_width",   "soss_inner_mask_width",
        "soss_outer_mask_width","miri_drop_groups",
    ]
    stage2_keys   = [
        "space_thresh", "time_thresh",
        "time_window",
        "miri_trace_width", "miri_background_width",
    ]
    stage3_keys   = ["extract_width"]

    # initialize current to medians
    current = {k: int(np.median(v)) for k,v in param_ranges.items()}

    # open global log
    logf = open(f"pipeline_outputs_directory/Files/Cost_{name_str}.txt","w")
    logc = open(f"pipeline_outputs_directory/Files/cov_{name_str}.txt","w")
    logs  = open(f"pipeline_outputs_directory/Files/Scatter_{name_str}.txt", "w")
    logf.write("\t".join(param_order)+"\tduration_s\tcost\n")

    count = 1
    best_cost = None
    # coordinate descent
    for key in param_order:
        fancyprint(
            f"→ Optimizing {key} "
            f"(fixed-other={{{', '.join(f'{k}:{current[k]}' for k in current if k!=key)}}})"
        )

        best_val = current[key]

        for trial in param_ranges[key]:
            fancyprint(f"Step {count}/{total_steps}: {key}={trial}")
            trial_params = {**current, key: trial}

            

            t0 = time.perf_counter()

            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} starting {key}={trial}",
                "\n############################################\n",
                flush=True
            )

            # split out args
            s1_args = {k:trial_params[k] for k in stage1_keys  if k in trial_params}
            s2_args = {k:trial_params[k] for k in stage2_keys  if k in trial_params}
            s3_args = {k:trial_params[k] for k in stage3_keys  if k in trial_params}

            # inherit for JumpStep
            if "time_window" in trial_params:
                s1_args["JumpStep"] = {"time_window":trial_params["time_window"]}

            # BadPixStep overrides
            badpix = {}
            if "box_size"   in trial_params: badpix["box_size"]   = trial_params["box_size"]
            if "window_size" in trial_params: badpix["window_size"] = trial_params["window_size"]
            if badpix:
                s2_args["BadPixStep"] = badpix

            if best_cost == None:
                # Stage 1
                st1 = run_stage1(
                    filenames,
                    mode=cfg["observing_mode"],
                    baseline_ints=baseline_ints,
                    flag_up_ramp=False,
                    save_results=True,
                    force_redo=True,
                    skip_steps=[],
                    **s1_args
                )
                
                # Stage 2
                st2, centroids = run_stage2(
                    st1,
                    mode=cfg["observing_mode"],
                    baseline_ints=baseline_ints,
                    save_results=True,
                    force_redo=True,
                    skip_steps=[],
                    **s2_args,
                    **cfg.get("stage2_kwargs",{})
                )
                if isinstance(centroids,np.ndarray):
                    centroids = pd.DataFrame(centroids.T,columns=["xpos","ypos"])

                # Stage 3
                st3 = run_stage3(
                    st2,
                    centroids=centroids,
                    save_results=True,
                    force_redo=True,
                    skip_steps=[],
                    **s3_args,
                    **cfg.get("stage3_kwargs",{})
                )
            
            else:
                if key in ('nirspec_mask_width'):

                    st1 = run_stage1(
                        filenames_int1,
                        mode=cfg["observing_mode"],
                        baseline_ints=baseline_ints,
                        flag_up_ramp=False,
                        save_results=True,
                        force_redo=True,
                        skip_steps=['DQInitStep','SaturationStep','DarkCurrentStep'],
                        **s1_args
                    )

                    st2, centroids = run_stage2(
                        st1,
                        mode=cfg["observing_mode"],
                        baseline_ints=baseline_ints,
                        save_results=True,
                        force_redo=True,
                        skip_steps=[],
                        **s2_args,
                        **cfg.get("stage2_kwargs",{})
                    )
                    if isinstance(centroids,np.ndarray):
                        centroids = pd.DataFrame(centroids.T,columns=["xpos","ypos"])
                    
                    st3 = run_stage3(
                        st2,
                        centroids=centroids,
                        save_results=True,
                        force_redo=True,
                        skip_steps=[],
                        **s3_args,
                        **cfg.get("stage3_kwargs",{})
                    )                    

                    
                elif key in ('time_rejection_threshold', 'time_window'):
                                        
                    st1 = run_stage1(
                        filenames_int2,
                        mode=cfg["observing_mode"],
                        baseline_ints=baseline_ints,
                        flag_up_ramp=False,
                        save_results=True,
                        force_redo=True,
                        skip_steps=['DQInitStep','SaturationStep','DarkCurrentStep','OneOverFStep','LinearityStep'],
                        **s1_args
                    )

                    st2, centroids = run_stage2(
                        st1,
                        mode=cfg["observing_mode"],
                        baseline_ints=baseline_ints,
                        save_results=True,
                        force_redo=True,
                        skip_steps=[],
                        **s2_args,
                        **cfg.get("stage2_kwargs",{})
                    )
                    if isinstance(centroids,np.ndarray):
                        centroids = pd.DataFrame(centroids.T,columns=["xpos","ypos"])
                    
                    st3 = run_stage3(
                        st2,
                        centroids=centroids,
                        save_results=True,
                        force_redo=True,
                        skip_steps=[],
                        **s3_args,
                        **cfg.get("stage3_kwargs",{})
                    )                             
                elif key in ('space_thresh', 'time_thresh', 'box_size', 'window_size'):

                    st2, centroids = run_stage2(
                        filenames_int3,
                        mode=cfg["observing_mode"],
                        baseline_ints=baseline_ints,
                        save_results=True,
                        force_redo=True,
                        skip_steps=['AssignWCSStep','Extract2DStep','SourceTypeStep','WaveCorrStep'],
                        **s2_args,
                        **cfg.get("stage2_kwargs",{})
                    )
                    if isinstance(centroids,np.ndarray):
                        centroids = pd.DataFrame(centroids.T,columns=["xpos","ypos"])
                    
                    st3 = run_stage3(
                        st2,
                        centroids=centroids,
                        save_results=True,
                        force_redo=True,
                        skip_steps=[],
                        **s3_args,
                        **cfg.get("stage3_kwargs",{})
                    )
                elif key in ('extract_width'):
      
                    st2, centroids = run_stage2(
                        filenames_int4,
                        mode=cfg["observing_mode"],
                        baseline_ints=baseline_ints,
                        save_results=True,
                        force_redo=True,
                        skip_steps=['AssignWCSStep','Extract2DStep','SourceTypeStep','WaveCorrStep','BadPixStep'],
                        **s2_args,
                        **cfg.get("stage2_kwargs",{})
                    )
                    if isinstance(centroids,np.ndarray):
                        centroids = pd.DataFrame(centroids.T,columns=["xpos","ypos"])
                    
                    st3 = run_stage3(
                        st2,
                        centroids=centroids,
                        save_results=True,
                        force_redo=True,
                        skip_steps=[],
                        **s3_args,
                        **cfg.get("stage3_kwargs",{})
                    )

            

            cost, scatter = cost_function(st3, cost_range=baseline_ints)
            
            covariance = compute_cov_metric(42)


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
                diagnostic_plot(st3, name_str)


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

if __name__ == "__main__":
    main()
