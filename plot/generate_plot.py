import os
import wandb
import matplotlib.pyplot as plt

ENTITY = "erlandpg"       
PROJECT = "ml710_FINALFINALFINALFINALproject"     
RUN_PATHS = [
    "ll9v3d15",
    "2cltph5u",
    "uaa4v0j1",
    "ta505fjg",
    "oj93ibcj"

]

RUN_PATHS = [f"{ENTITY}/{PROJECT}/{run_id}" for run_id in RUN_PATHS]

RUN_NAMES = [               
    
    
    
]

CUSTOM_RUN_NAMES = [
    "BS-1",
    "BS-8",
    "BS-16",
    "BS-32",
    "BS-64",
]

METRIC_KEY = "throughput" 

SMOOTHING_FACTOR =  0.9 

PLOT_TITLE = f"W&B Runs Comparison: {METRIC_KEY}"
X_AXIS_LABEL = "Steps"
Y_AXIS_LABEL = f"{METRIC_KEY} (Smoothed)" 
PLOT_GRID = True

def smooth_ema(series, factor):
    """Applies Exponential Moving Average smoothing."""
    if factor <= 0 or factor >= 1:
        return series
    return series.ewm(alpha=1 - factor, adjust=True).mean()

def _plot_single_run(run, metric_key, smoothing_factor, label_override=None):
    """Fetches history and plots data for a single wandb.Run object."""
    
    run_name_for_plot = label_override if label_override is not None else run.name
    print(f"  Processing run: '{run.name}' (ID: {run.id}) -> Plot Label: '{run_name_for_plot}'")

    try:
        history = run.history(keys=[metric_key, '_step'], pandas=True)
        print(f"    History fetched. Shape: {history.shape}")
        history = history.dropna(subset=[metric_key, '_step'])

        if history.empty:
            print(f"    WARNING: No valid data points found for metric '{metric_key}'. Skipping plot.")
            return False

        x_values = history['_step']
        y_values = history[metric_key]

        y_values_smoothed = smooth_ema(y_values, smoothing_factor)

        plt.plot(x_values, y_values_smoothed, label=run_name_for_plot, linewidth=2)
        print("    Plotted data.")
        return True

    except KeyError as e:
        print(f"    ERROR: Metric key '{metric_key}' or '_step' not found in run '{run.id}'. Details: {e}")
        return False
    except Exception as e:
        print(f"    ERROR: Unexpected error processing history for run '{run.id}'. Details: {e}")
        return False

def _plot_runs_by_path(api, run_paths, custom_run_names, metric_key, smoothing_factor):
    """Plots runs specified by paths, using custom names if provided."""
    print(f"Plotting {len(run_paths)} runs identified by path...")
    plot_count = 0
    for i, run_path in enumerate(run_paths):
        print(f"\nFetching run path: {run_path}")
        try:
            run = api.run(run_path)
            
            label = custom_run_names[i] if custom_run_names else None 
            if _plot_single_run(run, metric_key, smoothing_factor, label_override=label):
                plot_count += 1
        except wandb.errors.CommError as e:
            print(f"  ERROR: Could not fetch run '{run_path}'. Check path/permissions. Details: {e}")
        except Exception as e:
            print(f"  ERROR: Unexpected error for run path '{run_path}'. Details: {e}")
    return plot_count > 0

def _plot_runs_by_name(api, entity, project, run_names, custom_run_names, metric_key, smoothing_factor):
    """Plots runs specified by names, handling duplicates and custom names appropriately."""
    full_project_path = f"{entity}/{project}"
    print(f"Searching for runs by name in project: {full_project_path}")
    print(f"Run names to search for: {run_names}")
    plot_count = 0

    for i, run_name_to_find in enumerate(run_names):
        print(f"\nSearching for run name: '{run_name_to_find}'...")
        try:
            runs_found = api.runs(full_project_path, filters={"display_name": run_name_to_find})

            if len(runs_found) == 0:
                print(f"  WARNING: No run found with name '{run_name_to_find}'. Skipping.")
                continue

            print(f"  Found {len(runs_found)} run(s) with name '{run_name_to_find}'.")

            if len(runs_found) == 1:
                
                run = runs_found[0]
                label = custom_run_names[i] if custom_run_names else None
                if _plot_single_run(run, metric_key, smoothing_factor, label_override=label):
                    plot_count += 1
            else:
                
                print("  Handling duplicate names: Using 'name (id)' format for labels.")
                for run in runs_found:
                    label = f"{run.name} ({run.id})" 
                    if _plot_single_run(run, metric_key, smoothing_factor, label_override=label):
                        plot_count += 1

        except wandb.errors.CommError as e:
            print(f"  ERROR: Could not query runs for project '{full_project_path}'. Details: {e}")
        except Exception as e:
            print(f"  ERROR: Unexpected error searching for run name '{run_name_to_find}'. Details: {e}")
    return plot_count > 0

def plot_wandb_runs(
    metric_key,
    run_paths=None,
    entity=None,
    project=None,
    run_names=None,
    custom_run_names=None, 
    smoothing_factor=0.0,
    title="W&B Run Comparison",
    xlabel="Steps",
    ylabel=None,
    grid=True
    ):
    """
    Plots metrics from W&B runs, identified by path or name, with optional custom labels.
    """
    
    use_path_method = bool(run_paths)
    use_name_method = bool(entity and project and run_names)

    if use_path_method and use_name_method:
        raise ValueError("Configure EITHER run_paths OR entity/project/run_names, not both.")
    if not use_path_method and not use_name_method:
        raise ValueError("No runs specified. Configure EITHER run_paths OR entity/project/run_names.")

    
    if custom_run_names:
        if use_path_method and len(custom_run_names) != len(run_paths):
            raise ValueError(f"CUSTOM_RUN_NAMES length ({len(custom_run_names)}) must match RUN_PATHS length ({len(run_paths)}).")
        if use_name_method and len(custom_run_names) != len(run_names):
            raise ValueError(f"CUSTOM_RUN_NAMES length ({len(custom_run_names)}) must match RUN_NAMES length ({len(run_names)}).")

    
    if ylabel is None:
        smooth_label = f' (Smoothed {smoothing_factor:.1f})' if smoothing_factor > 0 else ''
        ylabel = f"{metric_key}{smooth_label}"

    print("Initializing W&B API...")
    try:
        api = wandb.Api(timeout=25)
        print("API initialized.")
    except Exception as e:
        print(f"FATAL: Failed to initialize W&B API: {e}")
        return

    
    plt.figure(figsize=(12, 7))
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    
    plot_successful = False
    if use_path_method:
        plot_successful = _plot_runs_by_path(api, run_paths, custom_run_names, metric_key, smoothing_factor)
    elif use_name_method:
        plot_successful = _plot_runs_by_name(api, entity, project, run_names, custom_run_names, metric_key, smoothing_factor)

    
    print("\nFinalizing plot...")
    if plot_successful:
        
        num_lines = len(plt.gca().lines)
        legend_fontsize = 10 if num_lines < 10 else 8 
        plt.legend(fontsize=legend_fontsize)
        if grid:
            plt.grid(True, linestyle='--', alpha=0.6)
    else:
        print("WARNING: No data was successfully plotted.")

    plt.tight_layout()
    print("Displaying plot.")
    plt.show()
    print("Saving plot to 'wandb_runs_comparison.png'.")
    plt.savefig(os.path.join("plot", "wandb_runs_comparison.png"), dpi=300)

if __name__ == "__main__":
    method1_configured = bool(RUN_PATHS)
    method2_configured = bool(ENTITY and PROJECT and RUN_NAMES) and ENTITY != "your_entity" and PROJECT != "your_project" and RUN_NAMES

    if not method1_configured and not method2_configured:
         print("\n" + "="*30 + " WARNING " + "="*30)
         print("Please configure EITHER RUN_PATHS OR ENTITY/PROJECT/RUN_NAMES.")
         print("="*70 + "\n")
    elif method1_configured and method2_configured:
         print("\n" + "="*30 + " WARNING " + "="*30)
         print("Please configure EITHER Method 1 OR Method 2, not both.")
         print("="*70 + "\n")
    elif CUSTOM_RUN_NAMES and method1_configured and len(CUSTOM_RUN_NAMES) != len(RUN_PATHS):
         print("\n" + "="*30 + " WARNING " + "="*30)
         print(f"Length mismatch: CUSTOM_RUN_NAMES ({len(CUSTOM_RUN_NAMES)}) vs RUN_PATHS ({len(RUN_PATHS)}).")
         print("="*70 + "\n")
    elif CUSTOM_RUN_NAMES and method2_configured and len(CUSTOM_RUN_NAMES) != len(RUN_NAMES):
         print("\n" + "="*30 + " WARNING " + "="*30)
         print(f"Length mismatch: CUSTOM_RUN_NAMES ({len(CUSTOM_RUN_NAMES)}) vs RUN_NAMES ({len(RUN_NAMES)}).")
         print("="*70 + "\n")
    elif METRIC_KEY == "your_metric_key":
        print("\n" + "="*30 + " WARNING " + "="*30)
        print("Please update the METRIC_KEY.")
        print("="*70 + "\n")
    else:
        custom_names_to_pass = CUSTOM_RUN_NAMES if CUSTOM_RUN_NAMES else None

        plot_wandb_runs(
            metric_key="statistical_efficiency",
            run_paths=RUN_PATHS if method1_configured else None,
            entity=ENTITY if method2_configured else None,
            project=PROJECT if method2_configured else None,
            run_names=RUN_NAMES if method2_configured else None,
            custom_run_names=custom_names_to_pass, 
            smoothing_factor=SMOOTHING_FACTOR,
            title=PLOT_TITLE,
            xlabel=X_AXIS_LABEL,
            ylabel=Y_AXIS_LABEL,
            grid=PLOT_GRID
        )

        plot_wandb_runs(
            metric_key="throughput",
            run_paths=RUN_PATHS if method1_configured else None,
            entity=ENTITY if method2_configured else None,
            project=PROJECT if method2_configured else None,
            run_names=RUN_NAMES if method2_configured else None,
            custom_run_names=custom_names_to_pass, 
            smoothing_factor=SMOOTHING_FACTOR,
            title=PLOT_TITLE,
            xlabel=X_AXIS_LABEL,
            ylabel=Y_AXIS_LABEL,
            grid=PLOT_GRID
        )