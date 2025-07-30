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

from exotedrf import utils
from exotedrf.utils import parse_config, unpack_input_dir, fancyprint
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3

#########################################
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


filenames_1 = [uncal_indir+'jw01366003001_04101_00001-seg001_nrs1_uncal.fits',
            uncal_indir+'jw01366003001_04101_00001-seg002_nrs1_uncal.fits',
            uncal_indir+'jw01366003001_04101_00001-seg003_nrs1_uncal.fits']


# ----------------------------------------
# plot cost
# ----------------------------------------

def plot_cost(name_str, table_height=0.5):
    # Load data
    df = pd.read_csv(f"pipeline_outputs_directory/Files/Cost_{name_str}.txt", delimiter="\t")

    # Drop rows with non-numeric cost values
    df = df[pd.to_numeric(df["cost_whole_set"], errors="coerce").notna()].reset_index(drop=True)

    # Get parameter columns (excluding duration_s and cost columns)
    param_cols = df.columns[:-3]

    # Track changed parameter and labels
    labels = []
    sweep_lines = []
    prev_row = None
    last_changed_param = None

    for idx, row in df.iterrows():
        if prev_row is None:
            value = int(row["time_window"]) if float(row["time_window"]).is_integer() else row["time_window"]
            label = f"time_window={value}"
            changed_param = "time_window"
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
        min_idx = df.iloc[start:end]["cost_whole_set"].idxmin()
        colors[min_idx] = 'green'

    # Best overall row
    best_row = df.loc[df["cost_whole_set"].idxmin(), param_cols.tolist() + ["cost_whole_set"]].copy()

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
    ax_plot.scatter(df["changed_label"], df["cost_whole_set"], color=colors)
    for x in sweep_lines:
        ax_plot.axvline(x=x - 0.5, color='gray', linestyle='--', linewidth=1)

    ax_plot.set_xticks(range(len(df)))
    ax_plot.set_xticklabels(df["changed_label"], rotation=90, fontsize=8)
    ax_plot.set_ylabel("Cost")
    ax_plot.set_xlabel("Changed Parameter")
    ax_plot.set_title(f"Cost by Single Parameter Sweep: {name_str}")

    # Hide table axis
    ax_table.axis("off")

    # Render the table
    table = ax_table.table(
        cellText=best_df.values,
        colLabels=best_df.columns,
        cellLoc='center',
        loc='center'
    )

    table.scale(1, 1)  # slightly expand cells

    # Set consistent, visible font size
    desired_fontsize = 12
    for key, cell in table.get_celld().items():
        cell.set_fontsize(desired_fontsize)

    plt.subplots_adjust(hspace=0.3)  # add vertical space between plot and table
    plt.savefig(f"pipeline_outputs_directory/Files/plot_cost_{name_str}.png", dpi=300)
    plt.show()





# ----------------------------------------
# cost (P2P-based)
# ----------------------------------------

def cost_function(st3):
    w1= 0.0
    w2 = 1.0
    flux = np.asarray(st3['Flux'], float)

    # white-light
    white = np.nansum(flux, axis=1)
    white = white[~np.isnan(white)]
    norm_white = white/np.median(white)
    d2_white = 0.5*(norm_white[:-2] + norm_white[2:]) - norm_white[1:-1]
    ptp2_white = np.nanmedian(np.abs(d2_white))

    # spectral
    wave_meds = np.nanmedian(flux, axis=0, keepdims=True)
    norm_spec = flux / wave_meds
    d2_spec = 0.5*(norm_spec[:-2] + norm_spec[2:]) - norm_spec[1:-1]
    ptp2_spec_wave = np.nanmedian(np.abs(d2_spec), axis=0)
    ptp2_spec = np.nanmedian(ptp2_spec_wave)

    return w1*ptp2_white + w2*ptp2_spec, ptp2_spec_wave

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
        norm_white_err_{name_str}.png
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
    plt.plot(norm_white, marker='o')
    plt.xlabel("Integration Number")
    plt.ylabel("Normalized White Flux")
    plt.title("Normalized White-light Curve")
    plt.grid(True)
    plt.savefig(f"{outdir}/norm_white_{name_str}.png", dpi=300)
    plt.close()

    # 2) white curve with per-integration scatter errorbars
    plt.figure()
    x = np.arange(len(norm_white))
    # for errorbars use the std-across-spectral-pixels for each integration
    normed_spec = flux / np.nanmedian(flux, axis=0, keepdims=True)
    yerr = np.nanstd(normed_spec, axis=1)
    plt.errorbar(x, norm_white, yerr=yerr, fmt="o-", capsize=3, elinewidth=1)
    plt.xlabel("Integration Number")
    plt.ylabel("Normalized White Flux")
    plt.title("Normalized White-light Curve w/ Errorbar")
    plt.grid(True)
    plt.savefig(f"{outdir}/norm_white_err_{name_str}.png", dpi=300)
    plt.close()

    # 3) normalized flux image (integrations vs. spectral pixels)
    plt.figure()
    img = flux / np.nanmedian(flux, axis=0, keepdims=True)
    plt.imshow(img, aspect='auto', vmin=0.99, vmax=1.01, origin='lower')
    plt.xlabel("Spectral Pixel")
    plt.ylabel("Integration Number")
    plt.title("Normalized Flux Image")
    plt.colorbar(label="Relative Flux")
    plt.savefig(f"{outdir}/flux_img_{name_str}.png", dpi=300)
    plt.close()



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
            "nirspec_mask_width":        list(range(10,31,2)),
            "time_rejection_threshold":  list(range(5,16,1)),
            "time_window":               list(range(5,12,2)),
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
        "window_size":   list(range(3,10,2)),
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
    logs  = open(f"pipeline_outputs_directory/Files/Scatter_{name_str}.txt", "w")
    logf.write("\t".join(param_order)+"\tduration_s\tcost_whole_set\tcost_baseline\n")

    count = 1
    # coordinate descent
    for key in param_order:
        fancyprint(
            f"→ Optimizing {key} "
            f"(fixed-other={{{', '.join(f'{k}:{current[k]}' for k in current if k!=key)}}})"
        )
        best_cost, best_val = None, current[key]

        for trial in param_ranges[key]:
            fancyprint(f"Step {count}/{total_steps}: {key}={trial}")
            trial_params = {**current, key: trial}
            baseline_ints = [100]

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
                    filenames_1,
                    mode=cfg["observing_mode"],
                    baseline_ints=baseline_ints,
                    flag_up_ramp=False,
                    save_results=True,
                    force_redo=False,
                    skip_steps=[],
                    **s1_args
                )
                
                # Stage 2
                st2, centroids = run_stage2(
                    st1,
                    mode=cfg["observing_mode"],
                    baseline_ints=baseline_ints,
                    save_results=True,
                    force_redo=False,
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
                    force_redo=False,
                    skip_steps=[],
                    **s3_args,
                    **cfg.get("stage3_kwargs",{})
                )
            
            else:
                if key in ('nirspec_mask_width'):
                    filenames_int1 = [outdir_s1+'jw01366003001_04101_00001-seg001_nrs1_darkcurrentstep.fits',
                                outdir_s1+'jw01366003001_04101_00001-seg002_nrs1_darkcurrentstep.fits',
                                outdir_s1+'jw01366003001_04101_00001-seg003_nrs1_darkcurrentstep.fits']


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
                    filenames_int2 = [outdir_s1+'jw01366003001_04101_00001-seg001_nrs1_linearitystep.fits',
                                outdir_s1+'jw01366003001_04101_00001-seg002_nrs1_linearitystep.fits',
                                outdir_s1+'jw01366003001_04101_00001-seg003_nrs1_linearitystep.fits']
                                        
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
                    filenames_int3 = [outdir_s1+'jw01366003001_04101_00001-seg001_nrs1_gainscalestep.fits',
                                outdir_s1+'jw01366003001_04101_00001-seg002_nrs1_gainscalestep.fits',
                                outdir_s1+'jw01366003001_04101_00001-seg003_nrs1_gainscalestep.fits']
                                                            
                    
                    #st1 = run_stage1(
                    #    filenames_int3,
                    #    mode=cfg["observing_mode"],
                    #    baseline_ints=baseline_ints,
                    #    flag_up_ramp=False,
                    #    save_results=True,
                    #    force_redo=True,
                    #    skip_steps=['DQInitStep','SaturationStep','DarkCurrentStep','OneOverFStep','LinearityStep','JumpStep','RampFitStep','GainScaleStep'],
                    #    **s1_args
                    #)

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
                    filenames_int4 = [outdir_s2+'jw01366003001_04101_00001-seg001_nrs1_badpixstep.fits',
                                outdir_s2+'jw01366003001_04101_00001-seg002_nrs1_badpixstep.fits',
                                outdir_s2+'jw01366003001_04101_00001-seg003_nrs1_badpixstep.fits']           

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


            cost, scatter        = cost_function(st3)
            flux100 = st3["Flux"].values[:100]
            cost_base, _   = cost_function({"Flux": flux100})

            dt = time.perf_counter() - t0
            fancyprint(f"→ cost = {cost:.12f} in {dt:.1f}s")

            # log it
            logf.write(
                "\t".join(str(trial_params[k]) for k in param_order)
                + f"\t{dt:.1f}\t{cost:.12f}\t{cost_base:.12f}\n"
            )

            line = " ".join(f"{x:.10g}" for x in scatter)  
            logs.write(line + "\n")

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

    fancyprint("=== FINAL OPTIMUM ===")
    fancyprint(current)
    plot_cost(name_str)

if __name__ == "__main__":
    main()
