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

from exotedrf.utils import parse_config, unpack_input_dir, fancyprint
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ----------------------------------------
# plot histogram
# ----------------------------------------

def histogram(name_str, table_height=0.5):
    # Load data
    df = pd.read_csv(f"OUT/Cost_{name_str}.txt", delimiter="\t")

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
    desired_fontsize = 14
    for key, cell in table.get_celld().items():
        cell.set_fontsize(desired_fontsize)

    plt.subplots_adjust(hspace=0.3)  # add vertical space between plot and table
    plt.savefig(f"OUT/histogram_{name_str}.png", dpi=300)
    plt.show()



#$1

# ----------------------------------------
# cost (P2P-based)
# ----------------------------------------

name_str = 'P2P_spec_whole_small'

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
    ptp2_spec = np.nanmedian(np.nanmedian(np.abs(d2_spec), axis=0))

    return w1*ptp2_white + w2*ptp2_spec

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

    # 1. Define the paths to all three files
    seg_paths = [
        os.path.join(cfg["input_dir"], "jw01366003001_04101_00001-seg001_nrs1_uncal.fits"),
        os.path.join(cfg["input_dir"], "jw01366003001_04101_00001-seg002_nrs1_uncal.fits"),
        os.path.join(cfg["input_dir"], "jw01366003001_04101_00001-seg003_nrs1_uncal.fits"),
    ]
    # 2. Open the first file
    first = datamodels.open(seg_paths[0])
    all_data = [first.data]  # Save its .data array

    # 3. Open the next files and collect their .data arrays
    for path in seg_paths[1:]:
        dm = datamodels.open(path)
        all_data.append(dm.data)  # Add their data to the list
        dm.close()

    # 4. Stack all data arrays together along axis 0 (integrations)
    stacked_data = np.concatenate(all_data, axis=0)

    # 5. Create a copy of the first datamodel and insert the full stacked data
    seg0 = first.copy()
    seg0.data = stacked_data
    seg0.meta.exposure.nints = stacked_data.shape[0]  # update number of integrations

    # 6. Use seg0 just like before:
    K = min(60, seg0.data.shape[0])
    dm_slice = seg0.copy()
    dm_slice.data = seg0.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end = K
    dm_slice.meta.exposure.nints = K

    first.close()  # clean up



    # Parameter to SWEEP sweep
    param_ranges = {}
    if args.instrument == "NIRISS":
        param_ranges.update({
            "soss_inner_mask_width": [20, 40, 80],
            "soss_outer_mask_width": [35, 70, 140],
            "jump_threshold": [5, 15, 30],
            "time_jump_threshold": [3, 10, 20],
        })
    elif args.instrument == "NIRSPEC":
        param_ranges.update({
            'time_window':              list(range(5,9,2)), # works
            ##'rejection_threshold':     list(range(10,21,1)), # works for Flag_up_ramp = True
            #'time_rejection_threshold': list(range(5,16,1)), # works           
            #"nirspec_mask_width":       list(range(10,31,2)), # works
        })
    else:  # MIRI
        param_ranges.update({
            #"miri_drop_groups": [6, 12, 24],
            #"jump_threshold": [5, 15, 30],
            #"time_jump_threshold": [3, 10, 20],
            #"miri_trace_width": [10, 20, 40], 
            #"miri_background_width": [7, 14, 28],
        })
    # for all instruments
    param_ranges.update({
        #"space_thresh": list(range(5,16,1)),
        #"time_thresh":  list(range(5,16,1)),
        #"box_size":     list(range(2,9,1)),  
        #"window_size":  list(range(3,10,2)),  
        #"extract_width": list(range(3,11,1 )),      
        "extract_width": [6]
    })

    param_order = list(param_ranges.keys())
    current = {k: int(np.median(v)) for k, v in param_ranges.items()}
    total_steps = sum(len(v) for v in param_ranges.values())

    # Logging
    logf = open(f"OUT/Cost_{name_str}.txt", "w")
    logf.write("\t".join(param_order) + "\tduration_s\tcost_whole_set\tcost_baseline\n")

    stage1_keys = [
        "rejection_threshold", "time_rejection_threshold",
        "nirspec_mask_width", "soss_inner_mask_width",
        "soss_outer_mask_width", "miri_drop_groups",
    ]
    
    stage2_keys = [
        "space_thresh", "time_thresh",      
        "time_window",                      
        "miri_trace_width", "miri_background_width",
    ]


    stage3_keys = ["extract_width"]

    count = 1
    # coordinate descent
    for key in param_order:
        fancyprint(
            f"→ Optimizing {key} "
            f"(fixed-other={{{', '.join([f'{k}:{current[k]}' for k in current if k != key])}}})"
        )
        best_cost, best_val = None, current[key]
        for trial in param_ranges[key]:
            fancyprint(f"Step {count}/{total_steps}: {key}={trial}")
            trial_params = {**current, key: trial}

            nints = dm_slice.data.shape[0]
            print("\033[1;91mnints is:\033[0m", nints)
            baseline_ints = [100]

            t0 = time.perf_counter()
            s1_args = {k: trial_params[k] for k in stage1_keys if k in trial_params}
            s2_args = {k: trial_params[k] for k in stage2_keys if k in trial_params}
            s3_args = {k: trial_params[k] for k in stage3_keys if k in trial_params}

            # Inherit Parameters (2nd level)
            if "time_window" in trial_params:
                s1_args["JumpStep"] = {"time_window": trial_params["time_window"]}
            
            badpix = {}
            if "box_size"   in trial_params:
                badpix["box_size"]   = trial_params["box_size"]
            if "window_size" in trial_params:
                badpix["window_size"] = trial_params["window_size"]
            if badpix:
                s2_args["BadPixStep"] = badpix

            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} starting {key}={trial}",
                "\n############################################\n",
                flush=True
            )

            st1 = run_stage1( 
                [dm_slice],
                mode=cfg["observing_mode"],
                baseline_ints=baseline_ints,
                flag_up_ramp=False, # Problem
                save_results=False,
                skip_steps=[],
                **s1_args
            )

            st2, centroids = run_stage2(
                st1,
                mode=cfg["observing_mode"],
                baseline_ints=baseline_ints,
                save_results=False,
                skip_steps=[],
                **s2_args,
                **cfg.get("stage2_kwargs", {})
            ) 
            if isinstance(centroids, np.ndarray):
                centroids = pd.DataFrame(centroids.T, columns=["xpos", "ypos"])
            st3 = run_stage3(
                st2,
                centroids=centroids,
                save_results=False,
                skip_steps=[],
                **s3_args,
                **cfg.get("stage3_kwargs", {})
            )


            dt = time.perf_counter() - t0

            # Compute cost on the full dataset
            cost = cost_function(st3)

            # Compute cost on the first 100 integrations only
            cost_baseline = cost_function({"Flux": st3["Flux"][:100]})

            fancyprint(f"→ cost = {cost:.12f} in {dt:.1f}s")

            logf.write(
                "\t".join(str(trial_params[k]) for k in param_order)
                + f"\t{dt:.1f}\t{cost:.12f}\t{cost_baseline:.12f}\n"
            )
            if best_cost is None or cost < best_cost:
                best_cost, best_val = cost, trial

                

                flux = np.asarray(st3['Flux'], dtype=float)
                white = np.nansum(flux, axis=1)
                white = white[~np.isnan(white)]
                med_white = np.median(white)
                norm_white = white / med_white
                norm_med_white = np.median(norm_white)
                norm_mad_white = np.nanmedian(np.abs(norm_white - norm_med_white))
                spec = flux
                norm_spec = spec / np.nanmedian(spec, axis=0, keepdims=True)
                mad_spec_per_int = np.nanmedian(np.abs(norm_spec - 1.0), axis=1)
                norm_mad_spec = np.nanmedian(mad_spec_per_int)

                # 1) Normalized white-light curve
                plt.figure()
                plt.plot(norm_white, marker='o')
                plt.xlabel("Integration Number")
                plt.ylabel("Normalized White Flux")
                plt.title("Normalized White-light Curve")
                plt.grid(True)           
                plt.savefig(f"OUT/norm_white_{name_str}.png", dpi=300)
                plt.close()

                plt.figure()
                x = np.arange(len(norm_white))
                normed_spec = flux / np.nanmedian(flux, axis=0, keepdims=True)
                yerr = np.nanstd(normed_spec, axis=1)
                plt.errorbar(x, norm_white, yerr=yerr,fmt="o-", capsize=3, elinewidth=1)
                plt.xlabel("Integration Number")
                plt.ylabel("Normalized White Flux")
                plt.title("Normalized White-light Curve with Errobar")
                plt.grid(True)                # turn on the grid
                plt.savefig(f"OUT/norm_white_error_{name_str}.png", dpi=300)
                plt.close()

                # 2) Normalized flux image
                plt.figure()
                plt.imshow(flux / np.nanmedian(flux, axis=0), aspect='auto', vmin=0.99, vmax=1.01)
                plt.xlabel("Spectral Pixel")
                plt.ylabel("Integration Number")
                plt.title("Normalized Flux Image")
                plt.savefig(f"OUT/flux_{name_str}.png", dpi=300)
                plt.close()



            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} completed (dt={dt:.1f}s)",
                "\n############################################\n",
                flush=True
            )            

            print("\033[1m\033[94m========== Cost ==========\033[0m",'\n', cost,'\n')

            count += 1

        current[key] = best_val
        fancyprint(f"Best {key} = {best_val} (cost={best_cost:.12f})")

    # total runtime
    t1 = time.perf_counter() - t0_total
    h, m = divmod(int(t1), 3600)
    m, s = divmod(m, 60)
    runtime_str = f"TOTAL runtime: {h}h {m:02d}min {s:04.1f}s\n"
    fancyprint(runtime_str)
    logf.write(runtime_str)

    logf.close()
    fancyprint("=== FINAL OPTIMUM ===")
    fancyprint(current)
    histogram(name_str)

if __name__ == "__main__":
    main()




"""
#$2

# ----------------------------------------
# cost (P2P-based)
# ----------------------------------------

name_str = 'P2P_50-50_whole'

def cost_function(st3):
    w1= 0.5
    w2 = 0.5
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
    ptp2_spec = np.nanmedian(np.nanmedian(np.abs(d2_spec), axis=0))

    return w1*ptp2_white + w2*ptp2_spec

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

    # 1. Define the paths to all three files
    seg_paths = [
        os.path.join(cfg["input_dir"], "jw01366003001_04101_00001-seg001_nrs1_uncal.fits"),
        os.path.join(cfg["input_dir"], "jw01366003001_04101_00001-seg002_nrs1_uncal.fits"),
        os.path.join(cfg["input_dir"], "jw01366003001_04101_00001-seg003_nrs1_uncal.fits"),
    ]
    # 2. Open the first file
    first = datamodels.open(seg_paths[0])
    all_data = [first.data]  # Save its .data array

    # 3. Open the next files and collect their .data arrays
    for path in seg_paths[1:]:
        dm = datamodels.open(path)
        all_data.append(dm.data)  # Add their data to the list
        dm.close()

    # 4. Stack all data arrays together along axis 0 (integrations)
    stacked_data = np.concatenate(all_data, axis=0)

    # 5. Create a copy of the first datamodel and insert the full stacked data
    seg0 = first.copy()
    seg0.data = stacked_data
    seg0.meta.exposure.nints = stacked_data.shape[0]  # update number of integrations

    # 6. Use seg0 just like before:
    K = min(500, seg0.data.shape[0])
    dm_slice = seg0.copy()
    dm_slice.data = seg0.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end = K
    dm_slice.meta.exposure.nints = K

    first.close()  # clean up



    # Parameter to SWEEP sweep
    param_ranges = {}
    if args.instrument == "NIRISS":
        param_ranges.update({
            "soss_inner_mask_width": [20, 40, 80],
            "soss_outer_mask_width": [35, 70, 140],
            "jump_threshold": [5, 15, 30],
            "time_jump_threshold": [3, 10, 20],
        })
    elif args.instrument == "NIRSPEC":
        param_ranges.update({
            'time_window':              list(range(5,12,2)), # works
            #'rejection_threshold':     list(range(10,21,1)), # works for Flag_up_ramp = True
            'time_rejection_threshold': list(range(5,16,1)), # works           
            "nirspec_mask_width":       list(range(10,31,2)), # works
        })
    else:  # MIRI
        param_ranges.update({
            #"miri_drop_groups": [6, 12, 24],
            #"jump_threshold": [5, 15, 30],
            #"time_jump_threshold": [3, 10, 20],
            #"miri_trace_width": [10, 20, 40], 
            #"miri_background_width": [7, 14, 28],
        })
    # for all instruments
    param_ranges.update({
        "space_thresh": list(range(5,16,1)),
        "time_thresh":  list(range(5,16,1)),
        "box_size":     list(range(2,9,1)),  
        "window_size":  list(range(3,10,2)),  
        "extract_width": list(range(3,11,1 )),      
    })

    param_order = list(param_ranges.keys())
    current = {k: int(np.median(v)) for k, v in param_ranges.items()}
    total_steps = sum(len(v) for v in param_ranges.values())

    # Logging
    logf = open(f"OUT/Cost_{name_str}.txt", "w")
    logf.write("\t".join(param_order) + "\tduration_s\tcost_whole_set\tcost_baseline\n")

    stage1_keys = [
        "rejection_threshold", "time_rejection_threshold",
        "nirspec_mask_width", "soss_inner_mask_width",
        "soss_outer_mask_width", "miri_drop_groups",
    ]
    
    stage2_keys = [
        "space_thresh", "time_thresh",      
        "time_window",                      
        "miri_trace_width", "miri_background_width",
    ]


    stage3_keys = ["extract_width"]

    count = 1
    # coordinate descent
    for key in param_order:
        fancyprint(
            f"→ Optimizing {key} "
            f"(fixed-other={{{', '.join([f'{k}:{current[k]}' for k in current if k != key])}}})"
        )
        best_cost, best_val = None, current[key]
        for trial in param_ranges[key]:
            fancyprint(f"Step {count}/{total_steps}: {key}={trial}")
            trial_params = {**current, key: trial}

            nints = dm_slice.data.shape[0]
            print("\033[1;91mnints is:\033[0m", nints)
            baseline_ints = [100]

            t0 = time.perf_counter()
            s1_args = {k: trial_params[k] for k in stage1_keys if k in trial_params}
            s2_args = {k: trial_params[k] for k in stage2_keys if k in trial_params}
            s3_args = {k: trial_params[k] for k in stage3_keys if k in trial_params}

            # Inherit Parameters (2nd level)
            if "time_window" in trial_params:
                s1_args["JumpStep"] = {"time_window": trial_params["time_window"]}
            
            badpix = {}
            if "box_size"   in trial_params:
                badpix["box_size"]   = trial_params["box_size"]
            if "window_size" in trial_params:
                badpix["window_size"] = trial_params["window_size"]
            if badpix:
                s2_args["BadPixStep"] = badpix

            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} starting {key}={trial}",
                "\n############################################\n",
                flush=True
            )

            st1 = run_stage1( 
                [dm_slice],
                mode=cfg["observing_mode"],
                baseline_ints=baseline_ints,
                flag_up_ramp=False, # Problem
                save_results=False,
                skip_steps=[],
                **s1_args
            )

            st2, centroids = run_stage2(
                st1,
                mode=cfg["observing_mode"],
                baseline_ints=baseline_ints,
                save_results=False,
                skip_steps=[],
                **s2_args,
                **cfg.get("stage2_kwargs", {})
            ) 
            if isinstance(centroids, np.ndarray):
                centroids = pd.DataFrame(centroids.T, columns=["xpos", "ypos"])
            st3 = run_stage3(
                st2,
                centroids=centroids,
                save_results=False,
                skip_steps=[],
                **s3_args,
                **cfg.get("stage3_kwargs", {})
            )


            dt = time.perf_counter() - t0

            dm_baseline = st3.copy()
            dm_baseline.data = st3.data[:100]
            dm_baseline.meta.exposure.integration_start = 1
            dm_baseline.meta.exposure.integration_end = 100
            dm_baseline.meta.exposure.nints = 100

            cost = cost_function(st3)
            cost_baseline = cost_function(dm_baseline)

            fancyprint(f"→ cost = {cost:.12f} in {dt:.1f}s")

            logf.write(
                "\t".join(str(trial_params[k]) for k in param_order)
                + f"\t{dt:.1f}\t{cost:.12f}\t{cost_baseline:.12f}\n"
            )
            if best_cost is None or cost < best_cost:
                best_cost, best_val = cost, trial

                

                flux = np.asarray(st3['Flux'], dtype=float)
                white = np.nansum(flux, axis=1)
                white = white[~np.isnan(white)]
                med_white = np.median(white)
                norm_white = white / med_white
                norm_med_white = np.median(norm_white)
                norm_mad_white = np.nanmedian(np.abs(norm_white - norm_med_white))
                spec = flux
                norm_spec = spec / np.nanmedian(spec, axis=0, keepdims=True)
                mad_spec_per_int = np.nanmedian(np.abs(norm_spec - 1.0), axis=1)
                norm_mad_spec = np.nanmedian(mad_spec_per_int)

                # 1) Normalized white-light curve
                plt.figure()
                plt.plot(norm_white, marker='o')
                plt.xlabel("Integration Number")
                plt.ylabel("Normalized White Flux")
                plt.title("Normalized White-light Curve")
                plt.grid(True)           
                plt.savefig(f"OUT/norm_white_{name_str}.png", dpi=300)
                plt.close()

                plt.figure()
                x = np.arange(len(norm_white))
                normed_spec = flux / np.nanmedian(flux, axis=0, keepdims=True)
                yerr = np.nanstd(normed_spec, axis=1)
                plt.errorbar(x, norm_white, yerr=yerr,fmt="o-", capsize=3, elinewidth=1)
                plt.xlabel("Integration Number")
                plt.ylabel("Normalized White Flux")
                plt.title("Normalized White-light Curve with Errobar")
                plt.grid(True)                # turn on the grid
                plt.savefig(f"OUT/norm_white_error_{name_str}.png", dpi=300)
                plt.close()

                # 2) Normalized flux image
                plt.figure()
                plt.imshow(flux / np.nanmedian(flux, axis=0), aspect='auto', vmin=0.99, vmax=1.01)
                plt.xlabel("Spectral Pixel")
                plt.ylabel("Integration Number")
                plt.title("Normalized Flux Image")
                plt.savefig(f"OUT/flux_{name_str}.png", dpi=300)
                plt.close()



            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} completed (dt={dt:.1f}s)",
                "\n############################################\n",
                flush=True
            )            

            print("\033[1m\033[94m========== Cost ==========\033[0m",'\n', cost,'\n')

            count += 1

        current[key] = best_val
        fancyprint(f"Best {key} = {best_val} (cost={best_cost:.12f})")

    # total runtime
    t1 = time.perf_counter() - t0_total
    h, m = divmod(int(t1), 3600)
    m, s = divmod(m, 60)
    runtime_str = f"TOTAL runtime: {h}h {m:02d}min {s:04.1f}s\n"
    fancyprint(runtime_str)
    logf.write(runtime_str)

    logf.close()
    fancyprint("=== FINAL OPTIMUM ===")
    fancyprint(current)
    histogram(name_str)

if __name__ == "__main__":
    main()





#$3

# ----------------------------------------
# cost (P2P-based)
# ----------------------------------------

name_str = 'P2P_white_whole'

def cost_function(st3):
    w1= 1.0
    w2 = 0.0
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
    ptp2_spec = np.nanmedian(np.nanmedian(np.abs(d2_spec), axis=0))

    return w1*ptp2_white + w2*ptp2_spec

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

    # 1. Define the paths to all three files
    seg_paths = [
        os.path.join(cfg["input_dir"], "jw01366003001_04101_00001-seg001_nrs1_uncal.fits"),
        os.path.join(cfg["input_dir"], "jw01366003001_04101_00001-seg002_nrs1_uncal.fits"),
        os.path.join(cfg["input_dir"], "jw01366003001_04101_00001-seg003_nrs1_uncal.fits"),
    ]
    # 2. Open the first file
    first = datamodels.open(seg_paths[0])
    all_data = [first.data]  # Save its .data array

    # 3. Open the next files and collect their .data arrays
    for path in seg_paths[1:]:
        dm = datamodels.open(path)
        all_data.append(dm.data)  # Add their data to the list
        dm.close()

    # 4. Stack all data arrays together along axis 0 (integrations)
    stacked_data = np.concatenate(all_data, axis=0)

    # 5. Create a copy of the first datamodel and insert the full stacked data
    seg0 = first.copy()
    seg0.data = stacked_data
    seg0.meta.exposure.nints = stacked_data.shape[0]  # update number of integrations

    # 6. Use seg0 just like before:
    K = min(500, seg0.data.shape[0])
    dm_slice = seg0.copy()
    dm_slice.data = seg0.data[:K]
    dm_slice.meta.exposure.integration_start = 1
    dm_slice.meta.exposure.integration_end = K
    dm_slice.meta.exposure.nints = K

    first.close()  # clean up



    # Parameter to SWEEP sweep
    param_ranges = {}
    if args.instrument == "NIRISS":
        param_ranges.update({
            "soss_inner_mask_width": [20, 40, 80],
            "soss_outer_mask_width": [35, 70, 140],
            "jump_threshold": [5, 15, 30],
            "time_jump_threshold": [3, 10, 20],
        })
    elif args.instrument == "NIRSPEC":
        param_ranges.update({
            'time_window':              list(range(5,12,2)), # works
            #'rejection_threshold':     list(range(10,21,1)), # works for Flag_up_ramp = True
            'time_rejection_threshold': list(range(5,16,1)), # works           
            "nirspec_mask_width":       list(range(10,31,2)), # works
        })
    else:  # MIRI
        param_ranges.update({
            #"miri_drop_groups": [6, 12, 24],
            #"jump_threshold": [5, 15, 30],
            #"time_jump_threshold": [3, 10, 20],
            #"miri_trace_width": [10, 20, 40], 
            #"miri_background_width": [7, 14, 28],
        })
    # for all instruments
    param_ranges.update({
        "space_thresh": list(range(5,16,1)),
        "time_thresh":  list(range(5,16,1)),
        "box_size":     list(range(2,9,1)),  
        "window_size":  list(range(3,10,2)),  
        "extract_width": list(range(3,11,1 )),      
    })

    param_order = list(param_ranges.keys())
    current = {k: int(np.median(v)) for k, v in param_ranges.items()}
    total_steps = sum(len(v) for v in param_ranges.values())

    # Logging
    logf = open(f"OUT/Cost_{name_str}.txt", "w")
    logf.write("\t".join(param_order) + "\tduration_s\tcost_whole_set\tcost_baseline\n")

    stage1_keys = [
        "rejection_threshold", "time_rejection_threshold",
        "nirspec_mask_width", "soss_inner_mask_width",
        "soss_outer_mask_width", "miri_drop_groups",
    ]
    
    stage2_keys = [
        "space_thresh", "time_thresh",      
        "time_window",                      
        "miri_trace_width", "miri_background_width",
    ]


    stage3_keys = ["extract_width"]

    count = 1
    # coordinate descent
    for key in param_order:
        fancyprint(
            f"→ Optimizing {key} "
            f"(fixed-other={{{', '.join([f'{k}:{current[k]}' for k in current if k != key])}}})"
        )
        best_cost, best_val = None, current[key]
        for trial in param_ranges[key]:
            fancyprint(f"Step {count}/{total_steps}: {key}={trial}")
            trial_params = {**current, key: trial}

            nints = dm_slice.data.shape[0]
            print("\033[1;91mnints is:\033[0m", nints)
            baseline_ints = [100]

            t0 = time.perf_counter()
            s1_args = {k: trial_params[k] for k in stage1_keys if k in trial_params}
            s2_args = {k: trial_params[k] for k in stage2_keys if k in trial_params}
            s3_args = {k: trial_params[k] for k in stage3_keys if k in trial_params}

            # Inherit Parameters (2nd level)
            if "time_window" in trial_params:
                s1_args["JumpStep"] = {"time_window": trial_params["time_window"]}
            
            badpix = {}
            if "box_size"   in trial_params:
                badpix["box_size"]   = trial_params["box_size"]
            if "window_size" in trial_params:
                badpix["window_size"] = trial_params["window_size"]
            if badpix:
                s2_args["BadPixStep"] = badpix

            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} starting {key}={trial}",
                "\n############################################\n",
                flush=True
            )

            st1 = run_stage1( 
                [dm_slice],
                mode=cfg["observing_mode"],
                baseline_ints=baseline_ints,
                flag_up_ramp=False, # Problem
                save_results=False,
                skip_steps=[],
                **s1_args
            )

            st2, centroids = run_stage2(
                st1,
                mode=cfg["observing_mode"],
                baseline_ints=baseline_ints,
                save_results=False,
                skip_steps=[],
                **s2_args,
                **cfg.get("stage2_kwargs", {})
            ) 
            if isinstance(centroids, np.ndarray):
                centroids = pd.DataFrame(centroids.T, columns=["xpos", "ypos"])
            st3 = run_stage3(
                st2,
                centroids=centroids,
                save_results=False,
                skip_steps=[],
                **s3_args,
                **cfg.get("stage3_kwargs", {})
            )


            dt = time.perf_counter() - t0

            dm_baseline = st3.copy()
            dm_baseline.data = st3.data[:100]
            dm_baseline.meta.exposure.integration_start = 1
            dm_baseline.meta.exposure.integration_end = 100
            dm_baseline.meta.exposure.nints = 100

            cost = cost_function(st3)
            cost_baseline = cost_function(dm_baseline)

            fancyprint(f"→ cost = {cost:.12f} in {dt:.1f}s")

            logf.write(
                "\t".join(str(trial_params[k]) for k in param_order)
                + f"\t{dt:.1f}\t{cost:.12f}\t{cost_baseline:.12f}\n"
            )
            if best_cost is None or cost < best_cost:
                best_cost, best_val = cost, trial

                

                flux = np.asarray(st3['Flux'], dtype=float)
                white = np.nansum(flux, axis=1)
                white = white[~np.isnan(white)]
                med_white = np.median(white)
                norm_white = white / med_white
                norm_med_white = np.median(norm_white)
                norm_mad_white = np.nanmedian(np.abs(norm_white - norm_med_white))
                spec = flux
                norm_spec = spec / np.nanmedian(spec, axis=0, keepdims=True)
                mad_spec_per_int = np.nanmedian(np.abs(norm_spec - 1.0), axis=1)
                norm_mad_spec = np.nanmedian(mad_spec_per_int)

                # 1) Normalized white-light curve
                plt.figure()
                plt.plot(norm_white, marker='o')
                plt.xlabel("Integration Number")
                plt.ylabel("Normalized White Flux")
                plt.title("Normalized White-light Curve")
                plt.grid(True)           
                plt.savefig(f"OUT/norm_white_{name_str}.png", dpi=300)
                plt.close()

                plt.figure()
                x = np.arange(len(norm_white))
                normed_spec = flux / np.nanmedian(flux, axis=0, keepdims=True)
                yerr = np.nanstd(normed_spec, axis=1)
                plt.errorbar(x, norm_white, yerr=yerr,fmt="o-", capsize=3, elinewidth=1)
                plt.xlabel("Integration Number")
                plt.ylabel("Normalized White Flux")
                plt.title("Normalized White-light Curve with Errobar")
                plt.grid(True)                # turn on the grid
                plt.savefig(f"OUT/norm_white_error_{name_str}.png", dpi=300)
                plt.close()

                # 2) Normalized flux image
                plt.figure()
                plt.imshow(flux / np.nanmedian(flux, axis=0), aspect='auto', vmin=0.99, vmax=1.01)
                plt.xlabel("Spectral Pixel")
                plt.ylabel("Integration Number")
                plt.title("Normalized Flux Image")
                plt.savefig(f"OUT/flux_{name_str}.png", dpi=300)
                plt.close()



            print(
                "\n############################################",
                f"\n Step: {count}/{total_steps} completed (dt={dt:.1f}s)",
                "\n############################################\n",
                flush=True
            )            

            print("\033[1m\033[94m========== Cost ==========\033[0m",'\n', cost,'\n')

            count += 1

        current[key] = best_val
        fancyprint(f"Best {key} = {best_val} (cost={best_cost:.12f})")

    # total runtime
    t1 = time.perf_counter() - t0_total
    h, m = divmod(int(t1), 3600)
    m, s = divmod(m, 60)
    runtime_str = f"TOTAL runtime: {h}h {m:02d}min {s:04.1f}s\n"
    fancyprint(runtime_str)
    logf.write(runtime_str)

    logf.close()
    fancyprint("=== FINAL OPTIMUM ===")
    fancyprint(current)
    histogram(name_str)

if __name__ == "__main__":
    main()



"""
print("DONE")

