import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import copy # For deepcopy if more complex structures are involved
import time
import psutil
import gc
import logging
import seaborn as sns
from scipy import stats
import multiprocessing
import traceback


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
from scipy.optimize import curve_fit
import scikit_posthocs as sp
from pathlib import Path
from ctbn_markov import AnticonvulsantCTBNMarkovModel
from legacy_markov import AnticonvulsantMarkovModel
from legacy_hh import HHModel

# Configure matplotlib for publication quality
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Set random seed for reproducibility
np.random.seed(42)

# Constants
MODEL_NAMES = {
    'ctbn_stiff': 'Anticonvulsant CTBN Markov',
    'legacy_markov': 'Anticonvulsant Legacy Markov',
    'legacy_hh': 'Legacy HH'
}

# Colors for plots (using seaborn's colorblind-friendly palette)
colorblind_palette_hex = sns.color_palette('colorblind').as_hex()
COLORS = {
    'ctbn_stiff': colorblind_palette_hex[0],
    'legacy_markov': colorblind_palette_hex[1],
    'legacy_hh': colorblind_palette_hex[2]
    # Add more if other models are plotted, e.g.:
    # 'model4': colorblind_palette_hex[3],
}

# Line styles for scaling plots
LINE_STYLES = {
    'ctbn_stiff': '-',
    'legacy_markov': '-.',
    'legacy_hh': ':'
}

# Marker styles for scaling plots
MARKERS = {
    'ctbn_stiff': 'o',
    'legacy_markov': '^',
    'legacy_hh': 'D'
}

# Create output directories
os.makedirs('data/benchmark', exist_ok=True)
os.makedirs('data/benchmark/figures', exist_ok=True)
os.makedirs('data/benchmark/data', exist_ok=True)


def measure_memory_and_time(func, model_for_static_analysis, *args, **kwargs):
    """Measure peak memory usage, static memory, and execution time of a function.
    
    Args:
        func: Function to measure
        model_for_static_analysis: Model instance to analyze for static memory
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        tuple: (result, execution_time_ms, peak_memory_mb, static_memory_kb, time_per_step_ms)
    """
    import psutil
    import sys
    import numpy as np
    import copy
    import multiprocessing
    import statistics
    
    process = psutil.Process()
    
    # PHASE 1: Multiple WARMUP runs - Run the function several times to stabilize JIT
    num_warmup = 3  # Increased from 1 to 3 for better JIT stabilization
    for _ in range(num_warmup):
        _ = func(*args, **kwargs)
        gc.collect()  # Clear memory after each warmup
    
    # Force garbage collection before measurement
    gc.collect()
    
    # Get model information
    model = model_for_static_analysis
    model_name = "Unknown"
    if hasattr(model, '__class__'):
        model_name = model.__class__.__name__
    
    # Measure specific memory characteristics based on model type
    static_memory_kb = 1.0  # Default fallback value
    
    if model is not None:
        # 1. Measure basic model size
        model_size = sys.getsizeof(model) / 1024  # KB
        
        # 2. Check for numpy arrays in the model
        array_size = 0
        attr_count = 0
        matrix_count = 0
        
        if hasattr(model, '__dict__'):
            for attr_name, attr_value in model.__dict__.items():
                attr_count += 1
                # Check for numpy arrays
                if isinstance(attr_value, np.ndarray):
                    array_size += attr_value.size * attr_value.itemsize / 1024  # KB
                    if attr_value.ndim > 1:
                        matrix_count += 1
        
        # 3. Model complexity measures
        method_count = len([method for method in dir(model) if callable(getattr(model, method)) and not method.startswith('__')])
        
        # Assign static memory value based on a consistent calculation
        # model_size is sys.getsizeof(model) / 1024 (KB)
        # array_size is sum of np.ndarray sizes in model.__dict__ (KB)
        static_memory_kb = max(1.0, model_size + array_size) # Consistent base + model object + numpy arrays
    
    # Remove random jitter which contributes to variability
    # static_memory_kb is now deterministic based on model attributes
    
    # Multiple timing measurements to reduce variance
    num_timing_runs = 3
    execution_times = []
    step_times = []
    
    for _ in range(num_timing_runs):
        # Execute function and time it
        start_time = time.time()
        result = func(*args, **kwargs)
        exec_time = (time.time() - start_time) * 1000  # ms
        execution_times.append(exec_time)
        
        # Enhanced step count detection for different model types
        steps = None
        
        # Method 1: Direct time attribute access in result
        if hasattr(result, 'time') and isinstance(result.time, (list, np.ndarray)) and len(result.time) > 1:
            steps = len(result.time)
        
        # Method 2: For tuple results (some models return multiple values)
        elif isinstance(result, tuple) and len(result) > 0:
            if hasattr(result[0], 'time') and isinstance(result[0].time, (list, np.ndarray)) and len(result[0].time) > 1:
                steps = len(result[0].time)
        
        # Method 3: For model-specific attributes
        if steps is None:
            # Try to get step count from SimSwp (e.g., in HHModel)
            if hasattr(model, 'SimSwp') and isinstance(model.SimSwp, np.ndarray) and model.SimSwp.size > 1:
                steps = model.SimSwp.size
            # Try to get step count from time (in model itself, not result)
            elif hasattr(model, 'time') and isinstance(model.time, (list, np.ndarray)) and len(model.time) > 1:
                steps = len(model.time)
            # Check sweep data for AnticonvulsantCTBNMarkovModel
            elif hasattr(model, 'sweep_data') and isinstance(model.sweep_data, dict):
                if 'time' in model.sweep_data and len(model.sweep_data['time']) > 1:
                    steps = len(model.sweep_data['time'])
        
        if steps is not None:
            step_times.append(exec_time / steps)
        
        # Give the system a moment to stabilize between runs
        time.sleep(0.1)
        gc.collect()
    
    # Use median instead of mean to reduce impact of outliers
    execution_time = statistics.median(execution_times)
    
    # Calculate time_per_step using median if we have step data
    if step_times:
        time_per_step = statistics.median(step_times)
    else:
        # Model-specific fallback mechanisms with improved accuracy
        steps = None
        
        # Fallback method 1: Extract from protocol parameters
        if hasattr(model, 'prot'):
            if hasattr(model.prot, 'Total_duration_ms') and hasattr(model.prot, 'sampint'):
                # Calculate steps based on protocol duration and sampling interval
                steps = int(model.prot.Total_duration_ms / model.prot.sampint) + 1
            elif hasattr(model.prot, 'Total_duration_ms'):
                # Use protocol duration and default sampling interval
                assumed_timestep = 0.025  # ms
                steps = int(model.prot.Total_duration_ms / assumed_timestep) + 1
        
        # Fallback method 2: Extract from model parameters
        if steps is None and hasattr(model, 'sampint'):
            if hasattr(model, 'protocol_duration_ms'):
                steps = int(model.protocol_duration_ms / model.sampint) + 1
            elif hasattr(model, 'prot') and hasattr(model.prot, 'Total_duration_ms'):
                steps = int(model.prot.Total_duration_ms / model.sampint) + 1
        
        # Fallback method 3: Model-specific knowledge
        if steps is None:
            if model_name == "LegacyHHModel":
                # HH model typically uses default protocol with 40,000 points
                steps = 40000
            elif model_name == "LegacyAnticonvulsantMarkovModel":
                # Legacy Markov typically uses around 8,000 steps
                steps = 8000
            elif model_name == "AnticonvulsantCTBNMarkovModel":
                # CTBN models use dynamic stepping but typical count is ~4000
                steps = 4000
            else:
                # Very last resort fallback
                logging.warning(f"Using last-resort fallback step count for unknown model {model_name}")
                steps = 5000
        
        time_per_step = execution_time / steps
        logging.info(f"Model {model_name}: Used fallback step count {steps} to calculate time per step")
    
    # Get peak memory after execution
    peak_memory_info = process.memory_info()
    peak_memory = peak_memory_info.rss / (1024 * 1024)  # MB

    return result, execution_time, peak_memory, static_memory_kb, time_per_step

def initialize_models():
    """Initialize all four models with default parameters.
    
    Returns:
        dict: Dictionary containing all model instances.
    """
    print("Initializing models...")
    
    models = {
        'ctbn_stiff': AnticonvulsantCTBNMarkovModel(),
        'legacy_markov': AnticonvulsantMarkovModel(),
        'legacy_hh': HHModel()
    }
    
    # Create similar voltage protocols for all models
    voltages = np.linspace(-80, 40, 13)
    for model_name, model in models.items():
        model.create_default_protocol(target_voltages=voltages)
        
        # Set specific drug type and concentration for anticonvulsant models
        if model_name in ['ctbn_stiff', 'legacy_markov']:  
            # Set drug type and concentration
            if hasattr(model, 'set_drug_type') and hasattr(model, 'set_drug_concentration'):
                model.set_drug_type('LTG')
                model.set_drug_concentration(25.0)  # Assuming concentration is in µM
                # print(f"    Drug set to LTG, 25.0 µM for {model_class.__name__}") # Optional: for debugging
            else:
                print(f"    Warning: Could not set drug_type or drug_concentration for {model_class.__name__}. Methods not found.")
            
    return models

def benchmark_computational_efficiency(models, n_repeats=30):
    """Benchmark computational efficiency metrics for all models.
    Execution time is reported as the geometric mean across all sweeps in the protocol.
    Peak memory is the maximum peak memory observed across all sweeps.
    Static memory is measured once per model configuration.
    Time per step is the arithmetic mean of time per step across all sweeps.

    Args:
        models: Dictionary of model instances
        n_repeats: Number of repeats for statistical significance
        
    Returns:
        pd.DataFrame: Computational efficiency benchmark results
    """
    print("Benchmarking computational efficiency...")
    
    results = []
    
    for model_name, model in models.items():
        print(f"  Testing {MODEL_NAMES[model_name]}...")
        
        num_sweeps = getattr(model, 'NumSwps', 0)
        if num_sweeps == 0:
            print(f"    Skipping {model_name} as it has no sweeps defined (NumSwps attribute missing or zero).")
            # Ensure model.create_default_protocol() has been called if NumSwps is expected.
            # This might happen if initialize_models() didn't set it up.
            if hasattr(model, 'create_default_protocol'):
                print(f"    Attempting to create default protocol for {model_name}")
                model.create_default_protocol() # Try to create it
                num_sweeps = getattr(model, 'NumSwps', 0)
                if num_sweeps == 0:
                    print(f"    Still no sweeps after attempting to create default protocol for {model_name}. Skipping.")
                    continue
                else:
                    print(f"    Successfully created default protocol. Proceeding with {num_sweeps} sweeps.")
            else:
                continue

        for repeat in range(n_repeats):
            try:
                # Get static memory once for this repeat/model configuration.
                # The func (model.Sweep) and args (0) are used here; static_mem is independent of sweep content.
                _, _, _, static_mem_for_repeat, _ = \
                    measure_memory_and_time(model.Sweep, model, 0) # SwpNo=0 used for this static mem call

                sweep_exec_times_for_gmean = []
                sweep_peak_mems = []
                sweep_time_steps = []

                for swp_idx in range(num_sweeps):
                    # Run the sweep and measure performance metrics for *this sweep*
                    _, exec_time_swp, peak_mem_swp, _, time_step_swp = \
                        measure_memory_and_time(model.Sweep, model, swp_idx)
                    
                    if exec_time_swp is not None and exec_time_swp > 0:
                        sweep_exec_times_for_gmean.append(exec_time_swp)
                    if peak_mem_swp is not None and not np.isnan(peak_mem_swp):
                        sweep_peak_mems.append(peak_mem_swp)
                    if time_step_swp is not None and not np.isnan(time_step_swp):
                        sweep_time_steps.append(time_step_swp)

                # Aggregate results for the repeat
                final_exec_time = stats.gmean(sweep_exec_times_for_gmean) if sweep_exec_times_for_gmean else np.nan
                final_peak_mem = np.max(sweep_peak_mems) if sweep_peak_mems else np.nan
                final_time_per_step = np.mean(sweep_time_steps) if sweep_time_steps else np.nan
                
                results.append({
                    'model': model_name,
                    'model_name': MODEL_NAMES[model_name],
                    'repeat': repeat,
                    'execution_time_ms': final_exec_time,
                    'peak_memory_mb': final_peak_mem,
                    'static_memory_kb': static_mem_for_repeat,
                    'time_per_step_ms': final_time_per_step
                })
            except Exception as e:
                print(f"    Error testing {MODEL_NAMES.get(model_name, model_name)} during repeat {repeat+1}, sweep {swp_idx if 'swp_idx' in locals() else 'N/A'} (in benchmark_computational_efficiency): {e}")
                import traceback
                traceback.print_exc() # Add more detailed error logging
                results.append({
                    'model': model_name,
                    'model_name': MODEL_NAMES[model_name],
                    'repeat': repeat,
                    'execution_time_ms': np.nan,
                    'peak_memory_mb': np.nan,
                    'static_memory_kb': np.nan,
                    'time_per_step_ms': np.nan
                })
    
    df = pd.DataFrame(results)
    
    # Save raw data
    df.to_csv('data/benchmark/data/anticonvulsant_computational_efficiency.csv', index=False)
    
    return df

# Worker function for parallel scaling benchmark - must be at module level for pickling
def parallel_scaling_worker(sweep_no):
    # This function avoids passing unpicklable model objects
    # It will be called by each process with just the sweep number
    # The global model will be used within each process
    return 1.0  # Just return a dummy value for timing purposes


# Worker function for actual parallel execution via ProcessPoolExecutor
# This function must be at the module level for pickling.
def _actual_parallel_sweep_worker(args):
    """
    Worker function that takes a model instance and a sweep index,
    and calls the Sweep method on the model.
    Args:
        args (tuple): A tuple containing (model_instance, sweep_idx).
    Returns:
        None
    """
    model_instance, sweep_idx = args
    try:
        model_instance.Sweep(sweep_idx)
    except Exception as e:
        # It's good practice for workers to catch their own exceptions
        # and potentially log them or return an error indicator.
        # For now, just print, as the main loop also has a try-except.
        # Ensure this print goes to a log or is handled if running in a context where stdout is not monitored.
        print(f"Error in _actual_parallel_sweep_worker for model {type(model_instance).__name__}, sweep {sweep_idx}: {e}")
        # import traceback
        # traceback.print_exc() # Consider logging this instead of printing if deployed
    return None

def benchmark_parallel_scaling(models, max_processes=multiprocessing.cpu_count(), n_repeats=5):
    """Benchmark scaling performance with increasing computational resources.
    Measures the time to complete n_processes identical tasks (model.Sweep(0))
    when run in parallel using n_processes workers.

    Args:
        models: Dictionary of model instances.
        max_processes: Maximum number of processes to test (default: all CPU cores).
        n_repeats: Number of repeats for statistical significance.
        
    Returns:
        pd.DataFrame: Parallel scaling benchmark results.
    """
    print("Benchmarking parallel scaling (with actual parallel execution)...")
    
    results = []
    # Ensure process_counts starts from 1 up to max_processes
    # Clamp max_processes to be at least 1 if cpu_count() returned 0 or less for some reason.
    actual_max_processes = max(1, max_processes)
    process_counts = range(1, actual_max_processes + 1)
    
    from concurrent.futures import ProcessPoolExecutor
    import time
    
    for model_name, model_instance in models.items():
        print(f"  Testing {MODEL_NAMES.get(model_name, model_name)}...")
        
        # Baseline: serial execution time for one model.Sweep(0) task.
        # This is T_1 in speedup calculations.
        # Ensure model.Sweep is the correct method and model_instance is the correct object.
        _, baseline_time_ms, _, _, _ = measure_memory_and_time(model_instance.Sweep, model_instance, 0)
        
        if baseline_time_ms is None or np.isnan(baseline_time_ms):
            print(f"    Skipping {model_name} due to invalid baseline_time: {baseline_time_ms}")
            continue

        for n_proc in process_counts:
            for repeat_idx in range(n_repeats):
                try:
                    start_wall_time = time.time()
                    
                    # Prepare arguments for each of the n_proc tasks.
                    # Each task is effectively model_instance.Sweep(0).
                    # A copy of model_instance will be pickled and sent to each worker.
                    tasks_args = [(model_instance, 0) for _ in range(n_proc)]
                    
                    with ProcessPoolExecutor(max_workers=n_proc) as executor:
                        # The list() call ensures all tasks are submitted and completed.
                        list(executor.map(_actual_parallel_sweep_worker, tasks_args))
                    
                    # Wall-clock time to complete all n_proc tasks in parallel.
                    parallel_run_time_sec = time.time() - start_wall_time
                    measured_parallel_time_ms = parallel_run_time_sec * 1000

                    effective_parallel_time_per_task_ms = np.nan
                    speedup = np.nan
                    efficiency = np.nan

                    if n_proc > 0 and measured_parallel_time_ms > 0:
                        effective_parallel_time_per_task_ms = measured_parallel_time_ms / n_proc
                        # Speedup S(N) = T_1 / (T_N / N) = (N * T_1) / T_N
                        # where T_1 is baseline_time_ms, N is n_proc, T_N is measured_parallel_time_ms
                        speedup = (baseline_time_ms * n_proc) / measured_parallel_time_ms
                        # Or, using effective_parallel_time_per_task_ms: speedup = baseline_time_ms / effective_parallel_time_per_task_ms
                        # Let's use the latter for clarity if effective_parallel_time_per_task_ms is valid
                        if effective_parallel_time_per_task_ms > 0:
                             speedup = baseline_time_ms / effective_parallel_time_per_task_ms
                        else:
                             speedup = np.nan # Avoid division by zero if effective time is zero
                        efficiency = speedup / n_proc
                    
                    results.append({
                        'model': model_name,
                        'model_name': MODEL_NAMES.get(model_name, model_name),
                        'n_processes': n_proc,
                        'repeat': repeat_idx,
                        'execution_time_ms': effective_parallel_time_per_task_ms, # Effective time per task
                        'total_parallel_time_ms': measured_parallel_time_ms, # Actual wall time for the batch
                        'baseline_time_ms': baseline_time_ms, # For reference
                        'speedup_factor': speedup,
                        'parallel_efficiency': efficiency,
                        'scaling_type': 'task_parallel' # N independent tasks on N processors
                    })
                except Exception as e:
                    print(f"    Error during parallel scaling benchmark for {model_name} with {n_proc} processes, repeat {repeat_idx+1}: {e}")
                    # import traceback
                    # traceback.print_exc() # Log this for debugging
                    results.append({
                        'model': model_name,
                        'model_name': MODEL_NAMES.get(model_name, model_name),
                        'n_processes': n_proc,
                        'repeat': repeat_idx,
                        'execution_time_ms': np.nan,
                        'total_parallel_time_ms': np.nan,
                        'baseline_time_ms': baseline_time_ms,
                        'speedup_factor': np.nan,
                        'parallel_efficiency': np.nan,
                        'scaling_type': 'task_parallel'
                    })
    
    df_results = pd.DataFrame(results)
    
    # Save raw data
    output_path = 'data/benchmark/data/anticonvulsant_parallel_scaling_actual.csv' # New filename to avoid overwriting old results
    df_results.to_csv(output_path, index=False)
    print(f"Actual parallel scaling results saved to {output_path}")
    
    return df_results

def benchmark_weak_scaling(models, max_factor=4, n_repeats=5):
    """Benchmark weak scaling (increasing problem size with computational resources).
    
    Args:
        models: Dictionary of model instances
        max_factor: Maximum scaling factor for problem size
        n_repeats: Number of repeats for statistical significance
        
    Returns:
        pd.DataFrame: Weak scaling benchmark results
    """
    print("Benchmarking weak scaling...")
    
    results = []
    scaling_factors = list(range(1, max_factor + 1))
    
    for model_name, model in models.items():
        print(f"  Testing {MODEL_NAMES[model_name]}...")
        
        # Save original sample count and channel count
        original_samples = None
        if hasattr(model, 'nsamp'):
            original_samples = model.nsamp
        elif hasattr(model, 'num_samples'):
            original_samples = model.num_samples
        
        original_channels = getattr(model, 'numchan', 1)
        
        # Get baseline execution time
        _, baseline_time, _, _, _ = measure_memory_and_time(model.Sweep, model, 0)
        
        for factor in scaling_factors:
            try:
                # Scale both problem size and computational resources proportionally
                
                # 1. Scale problem size (channels and/or samples)
                if hasattr(model, 'set_num_channels'):
                    model.set_num_channels(original_channels * factor)
                elif hasattr(model, 'numchan'):
                    model.numchan = original_channels * factor
                
                if original_samples is not None:
                    if hasattr(model, 'nsamp'):
                        model.nsamp = original_samples * factor
                    elif hasattr(model, 'num_samples'):
                        model.num_samples = original_samples * factor
                
                for repeat in range(n_repeats):
                    # 2. Measure execution time and memory with scaled problem
                    _, exec_time, peak_mem, _, _ = measure_memory_and_time(model.Sweep, model, 0)
                    
                    # In ideal weak scaling, execution time stays constant
                    # Efficiency = baseline_time / execution_time (closer to 1 is better)
                    weak_scaling_efficiency = baseline_time / exec_time if exec_time and exec_time > 0 else 0 # Avoid division by zero or negative time
                    
                    results.append({
                        'model': model_name,
                        'model_name': MODEL_NAMES[model_name],
                        'scaling_factor': factor,
                        'repeat': repeat,
                        'execution_time_ms': exec_time,
                        'peak_memory_mb': peak_mem, # Added missing peak_memory_mb
                        'weak_scaling_efficiency': weak_scaling_efficiency,
                        'scaling_type': 'weak'
                    })
            except Exception as e:
                print(f"    Error at scaling factor {factor} for {model_name}: {e}")
                break
            
        # Restore original settings
        if hasattr(model, 'set_num_channels'):
            model.set_num_channels(original_channels)
        elif hasattr(model, 'numchan'):
            model.numchan = original_channels
            
        if original_samples is not None:
            if hasattr(model, 'nsamp'):
                model.nsamp = original_samples
            elif hasattr(model, 'num_samples'):
                model.num_samples = original_samples
    
    df = pd.DataFrame(results)
    
    # Save raw data
    df.to_csv('data/benchmark/data/anticonvulsant_weak_scaling.csv', index=False)
    
    return df

def get_model_description(model_name):
    """Get a description for each model.

    Args:
        model_name: Name of the model
    Returns:
        str: Description of the model
    """
    descriptions = {
        'ctbn_stiff': "Continuous Time Bayesian Network with ODE solver",
        'legacy_markov': "Traditional Markov model implementation",
        'legacy_hh': "Hodgkin-Huxley model implementation"
    }
    return descriptions.get(model_name, "")

def perform_statistical_analysis(df, metric_column, group_column='model_name'):
    """Perform statistical analysis on benchmark results.
    
    Args:
        df: DataFrame containing benchmark results
        metric_column: Column name for the metric to analyze
        group_column: Column name for grouping (default: 'model_name')
        
    Returns:
        dict: Statistical analysis results including ANOVA, Kruskal-Wallis, and post-hoc tests
    """
    try:
        # Filter out NaN values
        filtered_df = df.dropna(subset=[metric_column])
        if len(filtered_df) < 3:
            return {
                'error': f"Not enough data for statistical analysis on {metric_column}",
                'has_significant_difference': False
            }
        
        # Create groups for analysis
        groups = []
        group_names = []
        
        for name, group in filtered_df.groupby(group_column):
            if len(group) > 0:
                groups.append(group[metric_column].values)
                group_names.append(name)
        
        if len(groups) < 2:
            return {
                'error': f"Not enough groups for comparison on {metric_column}",
                'has_significant_difference': False
            }
        
        # 1. Check normality with Shapiro-Wilk test
        normality_results = {}
        all_normal = True
        for i, name in enumerate(group_names):
            if len(groups[i]) >= 3:  # Shapiro-Wilk needs at least 3 samples
                stat, p = stats.shapiro(groups[i])
                normality_results[name] = {'statistic': stat, 'p_value': p, 'is_normal': p > 0.05}
                if p <= 0.05:
                    all_normal = False
            else:
                normality_results[name] = {'statistic': None, 'p_value': None, 'is_normal': False}
                all_normal = False
        
        # 2. Test for homogeneity of variances with Levene's test
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            var_stat, var_p = stats.levene(*groups)
            equal_variances = var_p > 0.05
        else:
            var_stat, var_p = None, None
            equal_variances = False
        
        # 3. Choose appropriate test based on normality and variance homogeneity
        if all_normal and equal_variances:
            # Use one-way ANOVA
            anova_stat, anova_p = stats.f_oneway(*groups)
            test_used = 'ANOVA'
            test_statistic = anova_stat
            p_value = anova_p
        else:
            # Use Kruskal-Wallis H-test (non-parametric alternative)
            try:
                kw_stat, kw_p = stats.kruskal(*groups)
                test_used = 'Kruskal-Wallis'
                test_statistic = kw_stat
                p_value = kw_p
            except Exception as e:
                return {
                    'error': f"Kruskal-Wallis test failed: {str(e)}",
                    'has_significant_difference': False
                }
        
        # 4. Post-hoc tests
        posthoc_results = None
        if p_value < 0.05:
            # Significant difference found, perform post-hoc tests
            try:
                if all_normal and equal_variances:
                    # Use Tukey's HSD test
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    posthoc = pairwise_tukeyhsd(filtered_df[metric_column].values, 
                                              filtered_df[group_column].values,
                                              alpha=0.05)
                    posthoc_results = {
                        'test': 'Tukey HSD',
                        'summary': str(posthoc)
                    }
                else:
                    # Use Dunn's test with Bonferroni correction
                    posthoc_df = sp.posthoc_dunn(filtered_df, 
                                                val_col=metric_column, 
                                                group_col=group_column, 
                                                p_adjust='bonferroni')
                    posthoc_results = {
                        'test': "Dunn's test with Bonferroni correction",
                        'p_values': posthoc_df.to_dict()
                    }
            except Exception as e:
                posthoc_results = {
                    'error': f"Post-hoc test failed: {str(e)}"
                }
        
        # 5. Effect size calculation (Cohen's d for pairwise comparisons)
        effect_sizes = {}
        if len(group_names) >= 2:
            for i in range(len(group_names)):
                for j in range(i+1, len(group_names)):
                    name_i = group_names[i]
                    name_j = group_names[j]
                    
                    data_i = groups[i]
                    data_j = groups[j]
                    
                    # Calculate Cohen's d
                    mean_i_val = np.mean(data_i)
                    mean_j_val = np.mean(data_j)
                    mean_diff = mean_i_val - mean_j_val
                    std_i = np.std(data_i, ddof=1) # ddof=1 for sample standard deviation
                    std_j = np.std(data_j, ddof=1)
                    n_i = len(data_i)
                    n_j = len(data_j)

                    # DEBUG PRINT BLOCK FOR COHEN'S D VERIFICATION
                    if 'metric_column' in locals() and metric_column == 'execution_time_ms':
                        # Check if the current pair is CTBN Markov and Legacy Markov
                        is_target_comparison = (name_i == "CTBN Markov" and name_j == "Legacy Markov") or \
                                             (name_i == "Legacy Markov" and name_j == "CTBN Markov")
                        if is_target_comparison:
                            print(f"\nDEBUG: Cohen's d calculation for {name_i} vs {name_j} (Execution Time) - using loop data")
                            print(f"DEBUG: Data for {name_i} (from loop): Mean={np.mean(data_i):.4f}, Std={np.std(data_i, ddof=1):.4f}, N={len(data_i)}")
                            print(f"DEBUG: Data for {name_j} (from loop): Mean={np.mean(data_j):.4f}, Std={np.std(data_j, ddof=1):.4f}, N={len(data_j)}")
                            
                            # Direct extraction for verification
                            ctbn_data_direct = filtered_df[filtered_df[group_column] == 'CTBN Markov'][metric_column].values
                            legacy_data_direct = filtered_df[filtered_df[group_column] == 'Legacy Markov'][metric_column].values
                            
                            if len(ctbn_data_direct) > 0 and len(legacy_data_direct) > 0:
                                mean_ctbn_direct = np.mean(ctbn_data_direct)
                                std_ctbn_direct = np.std(ctbn_data_direct, ddof=1)
                                n_ctbn_direct = len(ctbn_data_direct)
                                
                                mean_legacy_direct = np.mean(legacy_data_direct)
                                std_legacy_direct = np.std(legacy_data_direct, ddof=1)
                                n_legacy_direct = len(legacy_data_direct)
                                
                                print(f"DEBUG: Direct data CTBN Markov: Mean={mean_ctbn_direct:.4f}, Std={std_ctbn_direct:.4f}, N={n_ctbn_direct}")
                                print(f"DEBUG: Direct data Legacy Markov: Mean={mean_legacy_direct:.4f}, Std={std_legacy_direct:.4f}, N={n_legacy_direct}")
                                
                                if n_ctbn_direct > 1 and n_legacy_direct > 1:
                                    mean_diff_direct = mean_ctbn_direct - mean_legacy_direct # Order might affect sign, but abs is used for Cohen's d
                                    pooled_std_direct = np.sqrt(((n_ctbn_direct - 1) * std_ctbn_direct**2 + (n_legacy_direct - 1) * std_legacy_direct**2) / (n_ctbn_direct + n_legacy_direct - 2))
                                    print(f"DEBUG: Direct Mean Diff: {abs(mean_diff_direct):.4f}, Direct Pooled Std: {pooled_std_direct:.4f}")
                                    if pooled_std_direct > 0:
                                        cohen_d_direct_calc = abs(mean_diff_direct / pooled_std_direct)
                                        print(f"DEBUG: Direct Calculated Cohen's d: {cohen_d_direct_calc:.4f}")
                            else:
                                print("DEBUG: Could not find direct data for CTBN Markov or Legacy Markov.")
                            print("DEBUG: End of debug block\n")
                    # END DEBUG PRINT BLOCK

                    if n_i > 1 and n_j > 1:
                        pooled_std = np.sqrt(((n_i - 1) * std_i**2 + (n_j - 1) * std_j**2) / (n_i + n_j - 2))
                    else:
                        pooled_std = 0
                    if pooled_std > 0:
                        cohen_d = abs(mean_diff / pooled_std)
                    else:
                        cohen_d = 0
                        
                    effect_sizes[f"{name_i} vs {name_j}"] = {
                        "Cohen's d": cohen_d,
                        "Interpretation": get_effect_size_interpretation(cohen_d)
                    }
        
        return {
            'metric': metric_column,
            'group_means': {name: np.mean(group) for name, group in zip(group_names, groups)},
            'group_medians': {name: np.median(group) for name, group in zip(group_names, groups)},
            'group_std_devs': {name: np.std(group, ddof=1) for name, group in zip(group_names, groups)},
            'normality_test': normality_results,
            'equal_variance_test': {
                'test': "Levene's test",
                'statistic': var_stat,
                'p_value': var_p,
                'has_equal_variances': equal_variances
            },
            'omnibus_test': {
                'test': test_used,
                'statistic': test_statistic,
                'p_value': p_value,
                'has_significant_difference': p_value < 0.05
            },
            'post_hoc_test': posthoc_results,
            'effect_sizes': effect_sizes
        }
    except Exception as e:
        return {
            'error': f"Statistical analysis failed: {str(e)}",
            'traceback': traceback
        }

# Statistical Analysis Functions

def perform_pairwise_comparison(df, metric_column, model1, model2, group_column='model'):
    """
    Perform direct pairwise statistical comparison between two specific models.
    
    Args:
        df: DataFrame with benchmark data
        metric_column: Column name of the metric to analyze
        model1: First model to compare
        model2: Second model to compare
        group_column: Column used to group data (default: 'model')
        
    Returns:
        dict: Statistical comparison results between the two models
    """
    results = {}
    
    # Filter data for the two models
    df_subset = df[df[group_column].isin([model1, model2])].copy()
    if len(df_subset) == 0:
        return {"error": f"No data found for models {model1} and {model2}"}
    
    # Descriptive statistics
    results['descriptive'] = df_subset.groupby(group_column)[metric_column].describe()
    
    # Test for normality (Shapiro-Wilk)
    normality_results = {}
    for group in [model1, model2]:
        group_data = df_subset[df_subset[group_column] == group][metric_column]
        if len(group_data) < 3:
            normality_results[group] = {"statistic": None, "p-value": None, "normal": None}
            continue
            
        try:
            stat, p = stats.shapiro(group_data)
            normality_results[group] = {"statistic": stat, "p-value": p, "normal": p > 0.05}
        except Exception as e:
            normality_results[group] = {"statistic": None, "p-value": None, "normal": None}
    
    results['normality'] = normality_results
    
    # Determine if both groups are normally distributed
    both_normal = all(result.get("normal") for result in normality_results.values() if result.get("normal") is not None)
    
    # Get data for each model
    data1 = df_subset[df_subset[group_column] == model1][metric_column].values
    data2 = df_subset[df_subset[group_column] == model2][metric_column].values
    
    # Variance homogeneity test (Levene's test) if both are normal
    equal_variance = None
    if both_normal:
        try:
            stat, p = stats.levene(data1, data2)
            results['variance_homogeneity'] = {"statistic": stat, "p-value": p, "equal_variance": p > 0.05}
            equal_variance = p > 0.05
        except Exception as e:
            results['variance_homogeneity'] = {"statistic": None, "p-value": None, "equal_variance": None}
    
    # Choose appropriate test based on normality and variance homogeneity
    if both_normal:
        if equal_variance:
            # Parametric: Independent t-test with equal variances
            try:
                stat, p = stats.ttest_ind(data1, data2, equal_var=True)
                results['comparison'] = {
                    "test": "Independent t-test (equal variances)", 
                    "statistic": stat, 
                    "p-value": p, 
                    "significant": p < 0.05
                }
            except Exception as e:
                results['comparison'] = {
                    "test": "Independent t-test (equal variances)", 
                    "error": str(e)}
        else:
            # Parametric: Welch's t-test (unequal variances)
            try:
                stat, p = stats.ttest_ind(data1, data2, equal_var=False)
                results['comparison'] = {
                    "test": "Welch's t-test (unequal variances)", 
                    "statistic": stat, 
                    "p-value": p, 
                    "significant": p < 0.05
                }
            except Exception as e:
                results['comparison'] = {
                    "test": "Welch's t-test (unequal variances)", 
                    "error": str(e)}
    else:
        # Non-parametric: Mann-Whitney U test
        try:
            stat, p = stats.mannwhitneyu(data1, data2)
            results['comparison'] = {
                "test": "Mann-Whitney U test", 
                "statistic": stat, 
                "p-value": p, 
                "significant": p < 0.05
            }
        except Exception as e:
            results['comparison'] = {
                "test": "Mann-Whitney U test", 
                "error": str(e)}
    
    # Add effect size calculation (Cohen's d)
    try:
        mean1, std1 = np.mean(data1), np.std(data1, ddof=1)
        mean2, std2 = np.mean(data2), np.std(data2, ddof=1)
        # Pooled standard deviation
        n1, n2 = len(data1), len(data2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        # Cohen's d
        d = np.abs(mean1 - mean2) / pooled_std
        results['effect_size'] = {
            "metric": "Cohen's d",
            "value": d,
            "interpretation": "small" if d < 0.5 else "medium" if d < 0.8 else "large"
        }
    except Exception as e:
        results['effect_size'] = {"metric": "Cohen's d", "error": str(e)}
    
    return results

def perform_statistical_analysis(df, metric_column, group_column='model'):
    """
    Perform statistical analysis on benchmark data.
    
    Args:
        df: DataFrame with benchmark data
        metric_column: Column name of the metric to analyze
        group_column: Column used to group data (default: 'model')
        
    Returns:
        dict: Statistical results including descriptive stats, normality, ANOVA/Kruskal, and post-hoc tests
    """
    results = {}
    
    # Check for NaN values
    if df[metric_column].isna().any():
        print(f"Warning: NaN values found in {metric_column}, cleaning data")
        df = df.dropna(subset=[metric_column])
        
    if len(df) == 0:
        print(f"Error: No valid data for {metric_column}")
        return results
        
    # Descriptive statistics
    results['descriptive'] = df.groupby(group_column)[metric_column].describe()
    
    # Test for normality (Shapiro-Wilk)
    normality_results = {}
    for group in df[group_column].unique():
        group_data = df[df[group_column] == group][metric_column]
        if len(group_data) < 3:
            normality_results[group] = {"statistic": None, "p-value": None, "normal": None}
            continue
            
        try:
            stat, p = stats.shapiro(group_data)
            normality_results[group] = {"statistic": stat, "p-value": p, "normal": p > 0.05}
        except Exception as e:
            print(f"Error testing normality for {group}: {e}")
            normality_results[group] = {"statistic": None, "p-value": None, "normal": None}
    
    results['normality'] = normality_results
    
    # Determine if all groups are normally distributed
    all_normal = all(result.get("normal") for result in normality_results.values() if result.get("normal") is not None)
    
    # Variance homogeneity test (Levene's test)
    try:
        groups = [df[df[group_column] == group][metric_column].values for group in df[group_column].unique()]
        stat, p = stats.levene(*groups)
        results['variance_homogeneity'] = {"statistic": stat, "p-value": p, "equal_variance": p > 0.05}
    except Exception as e:
        print(f"Error testing variance homogeneity: {e}")
        results['variance_homogeneity'] = {"statistic": None, "p-value": None, "equal_variance": None}
    
    # ANOVA or Kruskal-Wallis test
    if all_normal and results['variance_homogeneity'].get("equal_variance"):
        # Parametric: One-way ANOVA
        try:
            stat, p = stats.f_oneway(*groups)
            results['group_comparison'] = {"test": "ANOVA", "statistic": stat, "p-value": p, "significant": p < 0.05}
        except Exception as e:
            print(f"Error performing ANOVA: {e}")
            results['group_comparison'] = {"test": "ANOVA", "statistic": None, "p-value": None, "significant": None}
    else:
        # Non-parametric: Kruskal-Wallis H-test
        try:
            stat, p = stats.kruskal(*groups)
            results['group_comparison'] = {"test": "Kruskal-Wallis", "statistic": stat, "p-value": p, "significant": p < 0.05}
        except Exception as e:
            print(f"Error performing Kruskal-Wallis test: {e}")
            results['group_comparison'] = {"test": "Kruskal-Wallis", "statistic": None, "p-value": None, "significant": None}
    
    # Post-hoc tests (if group differences are significant)
    if results['group_comparison'].get("significant"):
        model_names = df[group_column].unique()
        
        if all_normal and results['variance_homogeneity'].get("equal_variance"):
            # Parametric: Tukey's HSD test
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                tukeyhsd = pairwise_tukeyhsd(df[metric_column], df[group_column], alpha=0.05)
                results['posthoc'] = {
                    "test": "Tukey HSD",
                    "summary": str(tukeyhsd),
                    "significant_pairs": [(model_names[i], model_names[j]) 
                                         for i, j in zip(*np.where(tukeyhsd.reject))]
                }
            except Exception as e:
                print(f"Error performing Tukey's test: {e}")
                results['posthoc'] = {"test": "Tukey HSD", "summary": None, "significant_pairs": None}
        else:
            # Non-parametric: Dunn's test
            try:
                p_values = sp.posthoc_dunn(df, val_col=metric_column, group_col=group_column)
                significant_pairs = []
                for i, model1 in enumerate(model_names):
                    for j, model2 in enumerate(model_names):
                        if i < j and p_values.loc[model1, model2] < 0.05:
                            significant_pairs.append((model1, model2))
                
                results['posthoc'] = {
                    "test": "Dunn",
                    "p_values": p_values,
                    "significant_pairs": significant_pairs
                }
            except Exception as e:
                print(f"Error performing Dunn's test: {e}")
                results['posthoc'] = {"test": "Dunn", "p_values": None, "significant_pairs": None}
    
    # Effect size - Cohen's d for all pairs
    effect_sizes = {}

    # ---- START DIAGNOSTIC PRINTS ----
    if metric_column == 'execution_time_ms':
        print(f"\nDIAGNOSTIC (Active Function - execution_time_ms):")
        print(f"DIAGNOSTIC: group_column = {group_column}")
        try:
            unique_groups = df[group_column].unique()
            print(f"DIAGNOSTIC: df[group_column].unique() = {unique_groups}")
        except KeyError as e:
            print(f"DIAGNOSTIC: Error accessing df[{group_column}].unique(): {e}")
        print(f"DIAGNOSTIC: df.columns = {df.columns.tolist()}")
        print(f"DIAGNOSTIC: df.head() = \n{df.head()}\n")
    # ---- END DIAGNOSTIC PRINTS ----

    for i, group1 in enumerate(df[group_column].unique()):
        for j, group2 in enumerate(df[group_column].unique()):
            if i < j:
                data1 = df[df[group_column] == group1][metric_column].values
                data2 = df[df[group_column] == group2][metric_column].values

                # DEBUG PRINT BLOCK FOR COHEN'S D VERIFICATION (in the second function def)
                if metric_column == 'execution_time_ms':
                    is_target_pair = (group1 == "CTBN Markov" and group2 == "Legacy Markov") or \
                                     (group1 == "Legacy Markov" and group2 == "CTBN Markov")
                    if is_target_pair:
                        mean1_debug = np.mean(data1)
                        std1_debug = np.std(data1, ddof=1)
                        n1_debug = len(data1)
                        mean2_debug = np.mean(data2)
                        std2_debug = np.std(data2, ddof=1)
                        n2_debug = len(data2)
                        print(f"\nDEBUG (Active Function): Cohen's d for {group1} vs {group2} (Execution Time)")
                        print(f"DEBUG: Data for {group1}: Mean={mean1_debug:.4f}, Std={std1_debug:.4f}, N={n1_debug}")
                        print(f"DEBUG: Data for {group2}: Mean={mean2_debug:.4f}, Std={std2_debug:.4f}, N={n2_debug}")
                        if n1_debug > 1 and n2_debug > 1:
                            mean_diff_debug = mean1_debug - mean2_debug
                            pooled_std_debug_val = np.sqrt(((n1_debug - 1) * std1_debug**2 + (n2_debug - 1) * std2_debug**2) / (n1_debug + n2_debug - 2))
                            print(f"DEBUG: Mean Diff: {abs(mean_diff_debug):.4f}, Pooled Std: {pooled_std_debug_val:.4f}")
                            if pooled_std_debug_val > 0:
                                cohen_d_debug_calc = abs(mean_diff_debug / pooled_std_debug_val)
                                print(f"DEBUG: Calculated Cohen's d: {cohen_d_debug_calc:.4f}")
                        print("DEBUG: End of debug block (Active Function)\n")
                # END DEBUG PRINT BLOCK
                
                try:
                    # Cohen's d effect size
                    mean1, mean2 = np.mean(data1), np.mean(data2)
                    pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                        (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                        (len(data1) + len(data2) - 2))
                    
                    if pooled_std == 0:  # Avoid division by zero
                        effect_size = float('inf') if mean1 != mean2 else 0
                    else:
                        effect_size = abs(mean1 - mean2) / pooled_std
                        
                    # Interpret effect size
                    if effect_size < 0.2:
                        interpretation = "negligible"
                    elif effect_size < 0.5:
                        interpretation = "small"
                    elif effect_size < 0.8:
                        interpretation = "medium"
                    else:
                        interpretation = "large"
                        
                    effect_sizes[f"{group1} vs {group2}"] = {
                        "effect_size": effect_size,
                        "interpretation": interpretation
                    }
                except Exception as e:
                    print(f"Error calculating effect size between {group1} and {group2}: {e}")
                    effect_sizes[f"{group1} vs {group2}"] = {"effect_size": None, "interpretation": None}
    
    results['effect_sizes'] = effect_sizes
    
    return results

def plot_time_complexity(df_comp, save_path='data/benchmark/figures/time_complexity.png'):
    """Create publication-quality plot for time complexity comparison.
    
    Args:
        df_comp: DataFrame with computational efficiency data
        save_path: Path to save the figure
    """
    # Prepare data
    summary = df_comp.groupby('model').agg({
        'execution_time_ms': ['mean', 'std'],
        'time_per_step_ms': ['mean', 'std']
    })
    
    # Reset MultiIndex for easier access
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.reset_index(inplace=True)
    
    # Sort by mean execution time
    summary = summary.sort_values('execution_time_ms_mean')
    
    # Set up the figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    
    # Plot 1: Total Execution Time
    bars1 = ax1.bar(
        x=summary['model'].map(MODEL_NAMES),
        height=summary['execution_time_ms_mean'],
        yerr=summary['execution_time_ms_std'],
        color=[COLORS[model] for model in summary['model']],
        capsize=5,
        alpha=0.8,
        edgecolor='black',
        linewidth=1
    )
    
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Model Execution Time Comparison')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add values above bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', rotation=0)
    
    # Plot 2: Memory Usage
    bars2 = ax2.bar(
        x=summary['model'].map(MODEL_NAMES),
        height=summary['memory_usage_mb_mean'],
        yerr=summary['memory_usage_mb_std'],
        color=[COLORS[model] for model in summary['model']],
        capsize=5,
        alpha=0.8,
        edgecolor='black',
        linewidth=1
    )
    
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Model Memory Usage Comparison')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add values above bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', rotation=0)
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('data/benchmark/figures/computational_efficiency.png', dpi=300, bbox_inches='tight')
    plt.savefig('data/benchmark/figures/computational_efficiency.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    return {'computational_efficiency': fig}

def get_effect_size_interpretation(d):
    """
    Interpret Cohen's d effect size
    
    Args:
        d: Cohen's d effect size
        
    Returns:
        str: Interpretation of effect size
    """
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def visualize_computational_efficiency(df):
    print("DEBUG: Entered visualize_computational_efficiency") # ADDED DEBUG
    """
    Create publication-quality plots for computational efficiency metrics.
    Plots only 'CTBN Markov' and 'Legacy Markov' models.

    Args:
        df: DataFrame with computational efficiency benchmark results
        
    Returns:
        list: List of figure names that were generated
    """
    figures = []

    # Filter for specific models
    models_to_plot = ['ctbn_stiff', 'legacy_markov']
    df_filtered = df[df['model'].isin(models_to_plot)].copy() # Use .copy() to avoid SettingWithCopyWarning

    if df_filtered.empty:
        print("Warning: No data found for CTBN Markov or Legacy Markov in visualize_computational_efficiency. Skipping plots.")
        return figures

    # Group by model and compute statistics
    grouped = df_filtered.groupby('model')
    
    print("DEBUG: visualize_computational_efficiency - Before Execution Time Plot") # ADDED DEBUG
    # 1. Execution Time Comparison
    plt.figure(figsize=(10, 7))
    
    agg_stats = grouped['execution_time_ms'].agg(['mean', 'std', 'count'])
    means = agg_stats['mean']
    stds = agg_stats['std']
    counts = agg_stats['count']
    sems = stds / np.sqrt(counts)
    
    # Sort by mean execution time
    models_sorted = means.sort_values().index
    
    # Plot bars with error bars
    bars = plt.bar(
        x=np.arange(len(models_sorted)),
        height=[means.get(m, np.nan) for m in models_sorted], # Use .get for safety
        yerr=[sems.get(m, np.nan) for m in models_sorted],   # Use SEM for error bars
        capsize=10,
        color=[COLORS.get(m) for m in models_sorted],      # Use .get for safety
        edgecolor='black',
        linewidth=1
    )
    
    # Add value labels on top of bars
    for bar, model_key in zip(bars, models_sorted):
        height = bar.get_height()
        sem_val = sems.get(model_key, 0) # Use SEM for positioning text
        if not np.isnan(height) and not np.isnan(sem_val): # Ensure values are valid
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + sem_val + 0.05 * height if not np.isnan(height + sem_val) else height * 1.05, # Adjust offset based on SEM
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
    
    # Configure axis
    plt.title('Execution Time Comparison (CTBN vs Legacy Markov)', fontsize=14, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.xticks(np.arange(len(models_sorted)), [MODEL_NAMES.get(m, m) for m in models_sorted], rotation=45, ha='right') # Use .get for safety
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend with model colors for the plotted models
    handles = [mpl.patches.Patch(color=COLORS[m], label=MODEL_NAMES[m]) for m in models_sorted if m in COLORS and m in MODEL_NAMES]
    if handles: # Only show legend if there are handles
        plt.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('data/benchmark/figures/execution_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('data/benchmark/figures/execution_time_comparison.svg', format='svg')
    figures.append('execution_time_comparison')
    print("DEBUG: visualize_computational_efficiency - After Execution Time Plot") # ADDED DEBUG
    plt.close()
    
    print("DEBUG: visualize_computational_efficiency - Before Memory Usage Plot") # ADDED DEBUG
    # 2. Memory Usage Comparison
    plt.figure(figsize=(10, 7))
    
    if 'peak_memory_mb' in df_filtered.columns:
        memory_col = 'peak_memory_mb'
        ylabel = 'Peak Memory (MB)'
    elif 'memory_usage_mb' in df_filtered.columns: # Fallback for older data
        memory_col = 'memory_usage_mb'
        ylabel = 'Memory Usage (MB)'
    else:
        print(f"Warning: Neither 'peak_memory_mb' nor 'memory_usage_mb' found in columns for memory plot. Skipping.")
        plt.close() # Close the empty figure
        return figures # Or continue to next plot if appropriate

    # Re-group if memory_col was 'memory_usage_mb' and required filtering negative values
    # This assumes df_filtered already has the correct models.
    if memory_col == 'memory_usage_mb':
        df_mem_plot_filtered = df_filtered[df_filtered[memory_col] > 0].copy()
        if df_mem_plot_filtered.empty:
            print("Warning: No positive memory usage values found for selected models. Skipping memory plot.")
            plt.close()
            return figures # Or continue
        current_grouped = df_mem_plot_filtered.groupby('model')
    else:
        current_grouped = grouped # Use the initial grouping for peak_memory_mb

    agg_stats_mem = current_grouped[memory_col].agg(['mean', 'std', 'count'])
    means = agg_stats_mem['mean']
    stds = agg_stats_mem['std']
    counts = agg_stats_mem['count']
    sems_mem = stds / np.sqrt(counts)
    
    # Sort by mean memory usage (using models_sorted from the initial filtering for consistency if possible,
    # or re-sort if current_grouped is different)
    if current_grouped is not grouped:
         models_sorted_mem = means.sort_values().index
    else:
         models_sorted_mem = models_sorted # Use the same order as execution time if data is consistent

    bars = plt.bar(
        x=np.arange(len(models_sorted_mem)),
        height=[means.get(m, np.nan) for m in models_sorted_mem],
        yerr=[sems_mem.get(m, np.nan) for m in models_sorted_mem], # Use SEM for error bars
        capsize=10,
        color=[COLORS.get(m) for m in models_sorted_mem],
        edgecolor='black',
        linewidth=1
    )
    
    for bar, model_key in zip(bars, models_sorted_mem):
        height = bar.get_height()
        sem_val = sems_mem.get(model_key, 0) # Use SEM for positioning text
        if not np.isnan(height) and not np.isnan(sem_val): # Ensure values are valid
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + sem_val + 0.05 * height if not np.isnan(height + sem_val) else height * 1.05, # Adjust offset based on SEM
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
    
    plt.title('Memory Usage Comparison (CTBN vs Legacy Markov)', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(np.arange(len(models_sorted_mem)), [MODEL_NAMES.get(m, m) for m in models_sorted_mem], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    handles = [mpl.patches.Patch(color=COLORS[m], label=MODEL_NAMES[m]) for m in models_sorted_mem if m in COLORS and m in MODEL_NAMES]
    if handles:
        plt.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('data/benchmark/figures/memory_usage_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('data/benchmark/figures/memory_usage_comparison.svg', format='svg')
    figures.append('memory_usage_comparison')
    print("DEBUG: visualize_computational_efficiency - After Memory Usage Plot") # ADDED DEBUG
    plt.close()
    figures.append('memory_usage_comparison')
    
    # 3. Time per Step Comparison
    plt.figure(figsize=(10, 7))
    
    if 'time_per_step_ms' in df_filtered.columns:
        time_col = 'time_per_step_ms'
        ylabel = 'Time per Step (ms)'
    elif 'inference_time_us' in df_filtered.columns: # Fallback for older data
        time_col = 'inference_time_us'
        ylabel = 'Inference Time (μs)'
    else:
        print(f"Warning: Neither 'time_per_step_ms' nor 'inference_time_us' found for time per step plot. Skipping.")
        plt.close()
        return figures

    means = grouped[time_col].mean() # Use initial 'grouped' as it's based on df_filtered
    stds = grouped[time_col].std()
    
    # models_sorted is already defined from the execution time plot and based on df_filtered
    
    bars = plt.bar(
        x=np.arange(len(models_sorted)),
        height=[means.get(m, np.nan) for m in models_sorted],
        yerr=[stds.get(m, np.nan) for m in models_sorted],
        capsize=10,
        color=[COLORS.get(m) for m in models_sorted],
        edgecolor='black',
        linewidth=1
    )
    
    for bar, model_key in zip(bars, models_sorted):
        height = bar.get_height()
        std_val = stds.get(model_key, 0)
        if not np.isnan(height) and not np.isnan(std_val):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + std_val + 0.1 * height if not np.isnan(height + std_val) else height * 1.1,
                f'{height:.4f}',  # Increased precision here
                ha='center',
                va='bottom',
                fontweight='bold'
            )
            
    plt.title('Time per Step Comparison (CTBN vs Legacy Markov)', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(np.arange(len(models_sorted)), [MODEL_NAMES.get(m, m) for m in models_sorted], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    handles = [mpl.patches.Patch(color=COLORS[m], label=MODEL_NAMES[m]) for m in models_sorted if m in COLORS and m in MODEL_NAMES]
    if handles:
        plt.legend(handles=handles, loc='upper right')
        
    plt.tight_layout()
    plt.savefig('data/benchmark/figures/time_per_step_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('data/benchmark/figures/time_per_step_comparison.svg', format='svg')
    plt.close()
    figures.append('time_per_step_comparison')
    
    print("DEBUG: Exiting visualize_computational_efficiency") # ADDED DEBUG
    return figures

def visualize_scaling(df_weak, df_parallel):
    """
    Create publication-quality plots for scaling analysis.
    Plots only 'CTBN Markov' and 'Legacy Markov' models.

    Args:
        df_weak: DataFrame with weak scaling results
        df_parallel: DataFrame with parallel scaling results
        
    Returns:
        dict: Dictionary with paths to saved figures
    """
    figures = {}
    models_to_plot = ['ctbn_stiff', 'legacy_markov']

    # 1. Weak Scaling Efficiency Plot
    if df_weak is not None and not df_weak.empty:
        df_weak_filtered = df_weak[df_weak['model'].isin(models_to_plot)].copy()
        if not df_weak_filtered.empty:
            plt.figure(figsize=(10, 6))
            scaling_factors = sorted(df_weak_filtered['scaling_factor'].unique())
            
            for model_key in sorted(df_weak_filtered['model'].unique()):
                if model_key not in MODEL_NAMES or model_key not in COLORS or \
                   model_key not in LINE_STYLES or model_key not in MARKERS:
                    print(f"Warning: Model key '{model_key}' missing in plotting dictionaries. Skipping for weak scaling.")
                    continue

                model_data = df_weak_filtered[df_weak_filtered['model'] == model_key]
                # Group by scaling factor and compute mean, std, and count for SEM
                efficiency_stats = model_data.groupby('scaling_factor')['weak_scaling_efficiency'].agg(['mean', 'std', 'count'])
                
                y_values = [efficiency_stats['mean'].get(f, np.nan) for f in scaling_factors]
                std_values = [efficiency_stats['std'].get(f, np.nan) for f in scaling_factors]
                count_values = [efficiency_stats['count'].get(f, 0) for f in scaling_factors] # Default count to 0 to avoid div by zero if factor missing
                
                sem_values = [s / np.sqrt(c) if c > 0 else 0 for s, c in zip(std_values, count_values)]

                plt.errorbar(scaling_factors, y_values, yerr=sem_values, capsize=5,
                         color=COLORS[model_key], 
                         linestyle=LINE_STYLES[model_key],
                         marker=MARKERS[model_key],
                         linewidth=2,
                         markersize=8,
                         label=MODEL_NAMES[model_key])
            
            plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Ideal Efficiency')
            plt.title('Weak Scaling Efficiency (CTBN vs Legacy Markov)', fontsize=14, fontweight='bold')
            plt.xlabel('Problem Size Scaling Factor', fontsize=12)
            plt.ylabel('Weak Scaling Efficiency', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            if any(label in plt.gca().get_legend_handles_labels()[1] for label in [MODEL_NAMES.get(m) for m in models_to_plot]): # Check if any relevant labels were added
                 plt.legend()
            plt.xticks(scaling_factors)
            plt.ylim(0.9, 1.1)
            
            plt.tight_layout()
            plt.savefig('data/benchmark/figures/weak_scaling_efficiency.png', dpi=300, bbox_inches='tight')
            plt.savefig('data/benchmark/figures/weak_scaling_efficiency.svg', format='svg')
            figures['weak_scaling_efficiency'] = 'data/benchmark/figures/weak_scaling_efficiency.png'
            plt.close()
        else:
            print("Warning: No data for CTBN Markov or Legacy Markov in weak scaling data. Skipping weak scaling plot.")
    else:
        print("Warning: Weak scaling DataFrame is None or empty. Skipping weak scaling plot.")

    # 2. Strong Scaling Speedup Plot
    if df_parallel is not None and not df_parallel.empty:
        df_parallel_filtered = df_parallel[df_parallel['model'].isin(models_to_plot)].copy()
        if not df_parallel_filtered.empty:
            plt.figure(figsize=(10, 6))
            process_counts = sorted(df_parallel_filtered['n_processes'].unique())
            
            for model_key in sorted(df_parallel_filtered['model'].unique()):
                if model_key not in MODEL_NAMES or model_key not in COLORS or \
                   model_key not in LINE_STYLES or model_key not in MARKERS:
                    print(f"Warning: Model key '{model_key}' missing in plotting dictionaries. Skipping for strong scaling speedup.")
                    continue

                model_data = df_parallel_filtered[df_parallel_filtered['model'] == model_key]
                speedup_stats = model_data.groupby('n_processes')['speedup_factor'].agg(['mean', 'std', 'count'])
                y_values = [speedup_stats['mean'].get(p, np.nan) for p in process_counts]
                std_values = [speedup_stats['std'].get(p, np.nan) for p in process_counts]
                count_values = [speedup_stats['count'].get(p, 0) for p in process_counts]

                sem_values = [s / np.sqrt(c) if c > 0 else 0 for s, c in zip(std_values, count_values)]

                plt.errorbar(process_counts, y_values, yerr=sem_values, capsize=5,
                         color=COLORS[model_key], 
                         linestyle=LINE_STYLES[model_key],
                         marker=MARKERS[model_key],
                         linewidth=2,
                         markersize=8,
                         label=MODEL_NAMES[model_key])
            
            plt.plot(process_counts, process_counts, color='gray', linestyle='--', alpha=0.7, label='Ideal Speedup')
            plt.title('Strong Scaling Speedup (CTBN vs Legacy Markov)', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Processes', fontsize=12)
            plt.ylabel('Speedup Factor', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            if any(label in plt.gca().get_legend_handles_labels()[1] for label in [MODEL_NAMES.get(m) for m in models_to_plot]):
                 plt.legend()
            plt.xticks(process_counts)
            
            plt.tight_layout()
            plt.savefig('data/benchmark/figures/strong_scaling_speedup.png', dpi=300, bbox_inches='tight')
            plt.savefig('data/benchmark/figures/strong_scaling_speedup.svg', format='svg')
            figures['strong_scaling_speedup'] = 'data/benchmark/figures/strong_scaling_speedup.png'
            plt.close()
        else:
            print("Warning: No data for CTBN Markov or Legacy Markov in parallel scaling data. Skipping strong scaling speedup plot.")
    else:
        print("Warning: Parallel scaling DataFrame is None or empty. Skipping strong scaling speedup plot.")

    # 3. Strong Scaling Efficiency Plot
    if df_parallel is not None and not df_parallel.empty:
        df_parallel_filtered = df_parallel[df_parallel['model'].isin(models_to_plot)].copy() # Re-filter or use from above if scope allows
        if not df_parallel_filtered.empty: # Check again on this specific filtered df
            plt.figure(figsize=(10, 6))
            process_counts = sorted(df_parallel_filtered['n_processes'].unique()) # Recalculate for this scope

            for model_key in sorted(df_parallel_filtered['model'].unique()):
                if model_key not in MODEL_NAMES or model_key not in COLORS or \
                   model_key not in LINE_STYLES or model_key not in MARKERS:
                    print(f"Warning: Model key '{model_key}' missing in plotting dictionaries. Skipping for strong scaling efficiency.")
                    continue
                
                model_data = df_parallel_filtered[df_parallel_filtered['model'] == model_key]
                efficiency_data_grouped = model_data.groupby('n_processes')['parallel_efficiency'].mean()
                y_values = [efficiency_data_grouped.get(p, np.nan) for p in process_counts]
                
                plt.plot(process_counts, y_values,
                         color=COLORS[model_key], 
                         linestyle=LINE_STYLES[model_key],
                         marker=MARKERS[model_key],
                         linewidth=2,
                         markersize=8,
                         label=MODEL_NAMES[model_key])
            
            plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Ideal Efficiency')
            plt.title('Strong Scaling Efficiency (CTBN vs Legacy Markov)', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Processes', fontsize=12)
            plt.ylabel('Parallel Efficiency', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            if any(label in plt.gca().get_legend_handles_labels()[1] for label in [MODEL_NAMES.get(m) for m in models_to_plot]):
                 plt.legend()
            plt.xticks(process_counts)
            
            plt.tight_layout()
            plt.savefig('data/benchmark/figures/strong_scaling_efficiency.png', dpi=300, bbox_inches='tight')
            plt.savefig('data/benchmark/figures/strong_scaling_efficiency.svg', format='svg')
            figures['strong_scaling_efficiency'] = 'data/benchmark/figures/strong_scaling_efficiency.png'
            plt.close()
        # This else is covered by the strong scaling speedup plot's "else"
    # This outer else is also covered by the strong scaling speedup plot's "else"
    
    return figures

def generate_full_report(df_comp, df_weak=None, df_parallel=None, output_dir='data/benchmark'):
    """
    Generate a comprehensive report from existing benchmark data.
    
    Args:
        df_comp: DataFrame with computational efficiency results
        df_weak: DataFrame with weak scaling results (optional)
        df_parallel: DataFrame with parallel scaling results (optional)
        output_dir: Directory to save the report
        
    Returns:
        str: Path to the generated report
    """
    # Create output directories
    os.makedirs(f'{output_dir}/report', exist_ok=True)
    
    # Generate report content
    print("Generating comprehensive benchmark report...")
    report_content = generate_results_report(df_comp, df_weak, df_parallel)
    
    # Save report to Markdown file
    report_path = f'{output_dir}/report/benchmark_results.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    # Convert to HTML
    try:
        import markdown
        html_content = markdown.markdown(report_content, extensions=['tables', 'fenced_code'])
        
        # Add CSS for better styling
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Ion Channel Model Benchmark Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2, h3, h4 {{ color: #2980b9; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                img {{ max-width: 100%; height: auto; display: block; margin: 20px 0; }}
                code {{ background-color: #f5f5f5; padding: 2px 5px; border-radius: 3px; }}
                pre {{ background-color: #f5f5f5; padding: 10px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Save HTML report
        html_path = f'{output_dir}/report/benchmark_results.html'
        with open(html_path, 'w') as f:
            f.write(html_template)
        
        return html_path
    except ImportError:
        print("Could not convert to HTML (markdown package not available). Markdown report still available.")
        return report_path
    except Exception as e:
        print(f"Error generating HTML report: {str(e)}")
        return report_path

def generate_results_report(df_comp, df_weak=None, df_parallel=None):
    """
    Generate a comprehensive results report in Markdown format.
    
    Args:
        df_comp: DataFrame with computational efficiency results
        df_weak: DataFrame with weak scaling results (optional)
        df_parallel: DataFrame with parallel scaling results (optional)
        
    Returns:
        str: Markdown report content
    """
    # Create directories for report data
    os.makedirs('data/benchmark/figures', exist_ok=True)
    os.makedirs('data/benchmark/data', exist_ok=True)
    os.makedirs('data/benchmark/report', exist_ok=True)
    
    # Generate visualizations
    viz_comp = visualize_computational_efficiency(df_comp)
    
    if df_weak is not None and df_parallel is not None:
        viz_scaling = visualize_scaling(df_weak, df_parallel)
    else:
        viz_scaling = {}
    
    # Begin report content
    report = []

    # List models with their descriptions
    for model_id, model_name in MODEL_NAMES.items():
        report.append(f"- **{model_name}**: {get_model_description(model_id)}")
    
    report.append("\nWe evaluate these frameworks across multiple dimensions including computational efficiency, " +
                 "specialized CTBN queries, parallelization capabilities, and large-scale simulations.")
    
    # Key findings section
    report.append("\n## Key Findings")
    
    # Analyze computational efficiency results
    stats_exec = perform_statistical_analysis(df_comp, 'execution_time_ms')
    print(f"DEBUG: stats_exec (execution_time_ms omnibus): {stats_exec}")
    fastest_model = df_comp.groupby('model')['execution_time_ms'].mean().idxmin()
    
    report.append("\n### Computational Efficiency")
    report.append(f"\n- The **{MODEL_NAMES[fastest_model]}** model demonstrates the fastest execution time, " + 
                 f"averaging {df_comp[df_comp['model'] == fastest_model]['execution_time_ms'].mean():.2f} ms per simulation.")
    
    # Add statistical significance statement if available
    if 'omnibus_test' in stats_exec and stats_exec['omnibus_test'].get('has_significant_difference'):
        report.append(f"\n- Statistical testing ({stats_exec['omnibus_test']['test']}) confirms significant " + 
                     f"differences in execution times between models (p={stats_exec['omnibus_test']['p_value']:.4f}).")
    
    # Memory usage stats
    mem_efficient = df_comp.groupby('model')['peak_memory_mb'].mean().idxmin()
    report.append(f"\n- **{MODEL_NAMES[mem_efficient]}** is the most memory-efficient model, " + 
                 f"using {df_comp[df_comp['model'] == mem_efficient]['peak_memory_mb'].mean():.2f} MB on average.")
    
    # Scaling performance
    if df_parallel is not None:
        report.append("\n### Parallel Scaling Performance")
        
        # Find model with best parallel efficiency
        max_procs = df_parallel['n_processes'].max()
        best_scaling = None
        best_efficiency = 0
        
        for model in df_parallel['model'].unique():
            model_data = df_parallel[(df_parallel['model'] == model) & 
                                    (df_parallel['n_processes'] == max_procs)]
            if len(model_data) > 0:
                mean_efficiency = model_data['parallel_efficiency'].mean()
                if mean_efficiency > best_efficiency:
                    best_efficiency = mean_efficiency
                    best_scaling = model
        
        if best_scaling is not None:
            report.append(f"\n- **{MODEL_NAMES[best_scaling]}** demonstrates the best parallel scaling efficiency " +
                         f"({best_efficiency:.2f} at {max_procs} processes).")
    
    # Detailed results sections
    report.append("\n## Detailed Results")
    
    # Computational efficiency section
    report.append("\n### 1. Computational Efficiency")
    report.append("\n![Computational Efficiency Comparison](../figures/computational_efficiency.png)")
    
    report.append("\n**Figure 1**: Comparison of execution time, memory usage, and computational efficiency per time step across models.")
    
    # Add statistical analysis summary
    report.append("\n#### Statistical Analysis")
    report.append("\n**Execution Time:**")
    
    # Add specific pairwise comparisons
    any_exec_time_pairwise_significant = False
    report.append("\n**Specific Model Comparisons:**")
    
    # CTBN Markov vs Legacy HH
    ctbn_vs_hh = perform_pairwise_comparison(df_comp, 'execution_time_ms', 'ctbn_stiff', 'legacy_hh')
    if 'comparison' in ctbn_vs_hh and 'error' not in ctbn_vs_hh['comparison']:
        p_value = ctbn_vs_hh['comparison'].get('p-value', ctbn_vs_hh['comparison'].get('p_value'))
        test_name = ctbn_vs_hh['comparison'].get('test', 'Statistical test')
        significant = ctbn_vs_hh['comparison'].get('significant', False)
        sig_symbol = '*' if significant else 'not'
        
        # Add effect size interpretation if available
        effect_size_text = ""
        if 'effect_size' in ctbn_vs_hh and 'error' not in ctbn_vs_hh['effect_size']:
            effect_size = ctbn_vs_hh['effect_size'].get('value')
            effect_interp = ctbn_vs_hh['effect_size'].get('interpretation')
            effect_size_text = f" Effect size (Cohen's d): {effect_size:.2f} ({effect_interp})"
            
        report.append(f"\n- **CTBN Markov vs Legacy HH:** {test_name}, p={p_value:.4f} ({sig_symbol} significant).{effect_size_text}")
        
        # Add mean time comparison
        mean_ctbn = ctbn_vs_hh['descriptive'].loc['ctbn_stiff']['mean']
        mean_hh = ctbn_vs_hh['descriptive'].loc['legacy_hh']['mean']
        pct_diff = ((mean_hh - mean_ctbn) / mean_hh) * 100
        faster_slower = "faster" if pct_diff > 0 else "slower"
        report.append(f"  * CTBN Markov is {abs(pct_diff):.1f}% {faster_slower} than Legacy HH ({mean_ctbn:.2f} ms vs {mean_hh:.2f} ms)")
        if significant:
            any_exec_time_pairwise_significant = True
    
    # CTBN Markov vs Legacy Markov
    ctbn_vs_markov = perform_pairwise_comparison(df_comp, 'execution_time_ms', 'ctbn_stiff', 'legacy_markov')
    print(f"DEBUG: ctbn_vs_markov (execution_time_ms pairwise CTBN vs Legacy): {ctbn_vs_markov}")
    if 'comparison' in ctbn_vs_markov and 'error' not in ctbn_vs_markov['comparison']:
        p_value = ctbn_vs_markov['comparison'].get('p-value', ctbn_vs_markov['comparison'].get('p_value'))
        test_name = ctbn_vs_markov['comparison'].get('test', 'Statistical test')
        significant = ctbn_vs_markov['comparison'].get('significant', False)
        sig_symbol = '*' if significant else 'not'
        
        # Add effect size interpretation if available
        effect_size_text = ""
        if 'effect_size' in ctbn_vs_markov and 'error' not in ctbn_vs_markov['effect_size']:
            effect_size = ctbn_vs_markov['effect_size'].get('value')
            effect_interp = ctbn_vs_markov['effect_size'].get('interpretation')
            effect_size_text = f" Effect size (Cohen's d): {effect_size:.2f} ({effect_interp})"
            
        report.append(f"\n- **CTBN Markov vs Legacy Markov:** {test_name}, p={p_value:.4f} ({sig_symbol} significant).{effect_size_text}")
        
        # Add mean time comparison
        mean_ctbn = ctbn_vs_markov['descriptive'].loc['ctbn_stiff']['mean']
        mean_markov = ctbn_vs_markov['descriptive'].loc['legacy_markov']['mean']
        pct_diff = ((mean_markov - mean_ctbn) / mean_markov) * 100
        faster_slower = "faster" if pct_diff > 0 else "slower"
        report.append(f"  * CTBN Markov is {abs(pct_diff):.1f}% {faster_slower} than Legacy Markov ({mean_ctbn:.2f} ms vs {mean_markov:.2f} ms)")
        if significant:
            any_exec_time_pairwise_significant = True
    
    # Also add memory usage comparisons
    report.append("\n**Memory Usage:**")
    
    # CTBN Markov vs Legacy HH - Memory Usage
    ctbn_vs_hh_mem = perform_pairwise_comparison(df_comp, 'peak_memory_mb', 'ctbn_stiff', 'legacy_hh')
    if 'comparison' in ctbn_vs_hh_mem and 'error' not in ctbn_vs_hh_mem['comparison']:
        p_value = ctbn_vs_hh_mem['comparison'].get('p-value', ctbn_vs_hh_mem['comparison'].get('p_value'))
        test_name = ctbn_vs_hh_mem['comparison'].get('test', 'Statistical test')
        significant = ctbn_vs_hh_mem['comparison'].get('significant', False)
        sig_symbol = '*' if significant else 'not'
        
        # Add effect size interpretation if available
        effect_size_text = ""
        if 'effect_size' in ctbn_vs_hh_mem and 'error' not in ctbn_vs_hh_mem['effect_size']:
            effect_size = ctbn_vs_hh_mem['effect_size'].get('value')
            effect_interp = ctbn_vs_hh_mem['effect_size'].get('interpretation')
            effect_size_text = f" Effect size (Cohen's d): {effect_size:.2f} ({effect_interp})"
            
        report.append(f"\n- **CTBN Markov vs Legacy HH (Memory):** {test_name}, p={p_value:.4f} ({sig_symbol} significant).{effect_size_text}")
        
        # Add mean memory usage comparison
        mean_ctbn = ctbn_vs_hh_mem['descriptive'].loc['ctbn_stiff']['mean']
        mean_hh = ctbn_vs_hh_mem['descriptive'].loc['legacy_hh']['mean']
        pct_diff = ((mean_hh - mean_ctbn) / mean_hh) * 100
        efficient_text = "more memory-efficient" if pct_diff > 0 else "less memory-efficient"
        report.append(f"  * CTBN Markov is {abs(pct_diff):.1f}% {efficient_text} than Legacy HH ({mean_ctbn:.2f} MB vs {mean_hh:.2f} MB)")
    
    # CTBN Markov vs Legacy Markov - Memory Usage
    ctbn_vs_markov_mem = perform_pairwise_comparison(df_comp, 'peak_memory_mb', 'ctbn_stiff', 'legacy_markov')
    print(f"DEBUG: ctbn_vs_markov_mem (peak_memory_mb pairwise CTBN vs Legacy): {ctbn_vs_markov_mem}")
    if 'comparison' in ctbn_vs_markov_mem and 'error' not in ctbn_vs_markov_mem['comparison']:
        p_value = ctbn_vs_markov_mem['comparison'].get('p-value', ctbn_vs_markov_mem['comparison'].get('p_value'))
        test_name = ctbn_vs_markov_mem['comparison'].get('test', 'Statistical test')
        significant = ctbn_vs_markov_mem['comparison'].get('significant', False)
        sig_symbol = '*' if significant else 'not'
        
        # Add effect size interpretation if available
        effect_size_text = ""
        if 'effect_size' in ctbn_vs_markov_mem and 'error' not in ctbn_vs_markov_mem['effect_size']:
            effect_size = ctbn_vs_markov_mem['effect_size'].get('value')
            effect_interp = ctbn_vs_markov_mem['effect_size'].get('interpretation')
            effect_size_text = f" Effect size (Cohen's d): {effect_size:.2f} ({effect_interp})"
            
        report.append(f"\n- **CTBN Markov vs Legacy Markov (Memory):** {test_name}, p={p_value:.4f} ({sig_symbol} significant).{effect_size_text}")
        
        # Add mean memory usage comparison
        mean_ctbn = ctbn_vs_markov_mem['descriptive'].loc['ctbn_stiff']['mean']
        mean_markov = ctbn_vs_markov_mem['descriptive'].loc['legacy_markov']['mean']
        pct_diff = ((mean_markov - mean_ctbn) / mean_markov) * 100
        efficient_text = "more memory-efficient" if pct_diff > 0 else "less memory-efficient"
        report.append(f"  * CTBN Markov is {abs(pct_diff):.1f}% {efficient_text} than Legacy Markov ({mean_ctbn:.2f} MB vs {mean_markov:.2f} MB)")
    
    if 'omnibus_test' in stats_exec and stats_exec['omnibus_test'].get('significant'):
        report.append(f"\n- {stats_exec['omnibus_test']['test']} test: p = {stats_exec['omnibus_test']['p_value']:.4f} (significant)")
        
        if 'post_hoc_test' in stats_exec and stats_exec['post_hoc_test']:
            report.append("\n- Post-hoc analysis:")
            
            if 'significant_pairs' in stats_exec['post_hoc_test'] and stats_exec['post_hoc_test']['significant_pairs']:
                for pair in stats_exec['post_hoc_test']['significant_pairs']:
                    model1, model2 = pair
                    report.append(f"\n  - {MODEL_NAMES[model1]} vs {MODEL_NAMES[model2]}: significant difference")
    if not any_exec_time_pairwise_significant:
        report.append("\n- No statistically significant differences were found in the primary pairwise execution time comparisons reported above.")

    # Scaling performance section if data available
    if viz_scaling:
        report.append("\n### 2. Scaling Performance")
        
        if 'weak_scaling_efficiency' in viz_scaling:
            report.append("\n#### Weak Scaling")
            report.append("\n![Weak Scaling Efficiency](../figures/weak_scaling_efficiency.png)")
            report.append("\n**Figure 2a**: Weak scaling efficiency when increasing problem size and computational resources proportionally.")
            
            # Add pairwise comparisons for weak scaling if df_weak is available
            if df_weak is not None:
                report.append("\n**Pairwise Comparisons - Weak Scaling Efficiency:**")
                
                # Compare CTBN vs Legacy HH weak scaling
                max_scale = df_weak['scaling_factor'].max() if 'scaling_factor' in df_weak.columns else 4
                df_weak_last = df_weak[df_weak['scaling_factor'] == max_scale] if 'scaling_factor' in df_weak.columns else df_weak
                
                if len(df_weak_last) > 0:
                    ctbn_vs_hh_weak = perform_pairwise_comparison(df_weak_last, 'weak_scaling_efficiency', 'ctbn_stiff', 'legacy_hh')
                    if 'comparison' in ctbn_vs_hh_weak and 'error' not in ctbn_vs_hh_weak['comparison']:
                        p_value = ctbn_vs_hh_weak['comparison'].get('p-value', ctbn_vs_hh_weak['comparison'].get('p_value'))
                        test_name = ctbn_vs_hh_weak['comparison'].get('test', 'Statistical test')
                        significant = ctbn_vs_hh_weak['comparison'].get('significant', False)
                        sig_symbol = '*' if significant else 'not'
                        
                        # Get mean values
                        try:
                            mean_ctbn = ctbn_vs_hh_weak['descriptive'].loc['ctbn_stiff']['mean']
                            mean_hh = ctbn_vs_hh_weak['descriptive'].loc['legacy_hh']['mean']
                            pct_diff = ((mean_ctbn - mean_hh) / mean_hh) * 100
                            better_worse = "better" if pct_diff > 0 else "worse"
                            
                            report.append(f"\n- **CTBN Markov vs Legacy HH (Weak Scaling):** {test_name}, p={p_value:.4f} ({sig_symbol} significant).")
                            report.append(f"  * CTBN Markov has {abs(pct_diff):.1f}% {better_worse} weak scaling efficiency than Legacy HH at scaling factor {max_scale} ({mean_ctbn:.2f} vs {mean_hh:.2f})")
                        except (KeyError, AttributeError) as e:
                            report.append(f"\n- **CTBN Markov vs Legacy HH (Weak Scaling):** {test_name}, p={p_value:.4f} ({sig_symbol} significant). (Error calculating means: {str(e)})")
                    
                    # Compare CTBN vs Legacy Markov weak scaling
                    ctbn_vs_markov_weak = perform_pairwise_comparison(df_weak_last, 'weak_scaling_efficiency', 'ctbn_stiff', 'legacy_markov')
                    if 'comparison' in ctbn_vs_markov_weak and 'error' not in ctbn_vs_markov_weak['comparison']:
                        p_value = ctbn_vs_markov_weak['comparison'].get('p-value', ctbn_vs_markov_weak['comparison'].get('p_value'))
                        test_name = ctbn_vs_markov_weak['comparison'].get('test', 'Statistical test')
                        significant = ctbn_vs_markov_weak['comparison'].get('significant', False)
                        sig_symbol = '*' if significant else 'not'
                        
                        # Get mean values
                        try:
                            mean_ctbn = ctbn_vs_markov_weak['descriptive'].loc['ctbn_stiff']['mean']
                            mean_markov = ctbn_vs_markov_weak['descriptive'].loc['legacy_markov']['mean']
                            pct_diff = ((mean_ctbn - mean_markov) / mean_markov) * 100
                            better_worse = "better" if pct_diff > 0 else "worse"
                            
                            report.append(f"\n- **CTBN Markov vs Legacy Markov (Weak Scaling):** {test_name}, p={p_value:.4f} ({sig_symbol} significant).")
                            report.append(f"  * CTBN Markov has {abs(pct_diff):.1f}% {better_worse} weak scaling efficiency than Legacy Markov at scaling factor {max_scale} ({mean_ctbn:.2f} vs {mean_markov:.2f})")
                        except (KeyError, AttributeError) as e:
                            report.append(f"\n- **CTBN Markov vs Legacy Markov (Weak Scaling):** {test_name}, p={p_value:.4f} ({sig_symbol} significant). (Error calculating means: {str(e)})")

        
        if 'strong_scaling_speedup' in viz_scaling:
            report.append("\n#### Strong Scaling")
            report.append("\n![Strong Scaling Speedup](../figures/strong_scaling_speedup.png)")
            report.append("\n**Figure 2b**: Strong scaling speedup when increasing computational resources for a fixed problem size.")
            
            report.append("\n![Strong Scaling Efficiency](../figures/strong_scaling_efficiency.png)")
            report.append("\n**Figure 2c**: Strong scaling efficiency as a measure of parallelization effectiveness.")
            
            # Add pairwise comparisons for strong scaling if df_parallel is available
            if df_parallel is not None:
                report.append("\n**Pairwise Comparisons - Strong Scaling Performance:**")
                
                # Compare CTBN vs Legacy HH strong scaling efficiency
                max_proc = df_parallel['n_processes'].max() if 'n_processes' in df_parallel.columns else 4
                df_parallel_max = df_parallel[df_parallel['n_processes'] == max_proc] if 'n_processes' in df_parallel.columns else df_parallel
                
                if len(df_parallel_max) > 0:
                    # Strong scaling efficiency comparison
                    ctbn_vs_hh_strong = perform_pairwise_comparison(df_parallel_max, 'parallel_efficiency', 'ctbn_stiff', 'legacy_hh')
                    if 'comparison' in ctbn_vs_hh_strong and 'error' not in ctbn_vs_hh_strong['comparison']:
                        p_value = ctbn_vs_hh_strong['comparison'].get('p-value', ctbn_vs_hh_strong['comparison'].get('p_value'))
                        test_name = ctbn_vs_hh_strong['comparison'].get('test', 'Statistical test')
                        significant = ctbn_vs_hh_strong['comparison'].get('significant', False)
                        sig_symbol = '*' if significant else 'not'
                        
                        # Get mean values
                        try:
                            mean_ctbn = ctbn_vs_hh_strong['descriptive'].loc['ctbn_stiff']['mean']
                            mean_hh = ctbn_vs_hh_strong['descriptive'].loc['legacy_hh']['mean']
                            pct_diff = ((mean_ctbn - mean_hh) / mean_hh) * 100
                            better_worse = "better" if pct_diff > 0 else "worse"
                            
                            report.append(f"\n- **CTBN Markov vs Legacy HH (Strong Scaling Efficiency):** {test_name}, p={p_value:.4f} ({sig_symbol} significant).")
                            report.append(f"  * CTBN Markov has {abs(pct_diff):.1f}% {better_worse} strong scaling efficiency than Legacy HH at {max_proc} processes ({mean_ctbn:.2f} vs {mean_hh:.2f})")
                        except (KeyError, AttributeError) as e:
                            report.append(f"\n- **CTBN Markov vs Legacy HH (Strong Scaling Efficiency):** {test_name}, p={p_value:.4f} ({sig_symbol} significant). (Error calculating means: {str(e)})")
                    
                    # Strong scaling speedup comparison
                    ctbn_vs_hh_speedup = perform_pairwise_comparison(df_parallel_max, 'speedup_factor', 'ctbn_stiff', 'legacy_hh')
                    if 'comparison' in ctbn_vs_hh_speedup and 'error' not in ctbn_vs_hh_speedup['comparison']:
                        p_value = ctbn_vs_hh_speedup['comparison'].get('p-value', ctbn_vs_hh_speedup['comparison'].get('p_value'))
                        test_name = ctbn_vs_hh_speedup['comparison'].get('test', 'Statistical test')
                        significant = ctbn_vs_hh_speedup['comparison'].get('significant', False)
                        sig_symbol = '*' if significant else 'not'
                        
                        # Get mean values
                        try:
                            mean_ctbn = ctbn_vs_hh_speedup['descriptive'].loc['ctbn_stiff']['mean']
                            mean_hh = ctbn_vs_hh_speedup['descriptive'].loc['legacy_hh']['mean']
                            pct_diff = ((mean_ctbn - mean_hh) / mean_hh) * 100
                            better_worse = "better" if pct_diff > 0 else "worse"
                            
                            report.append(f"\n- **CTBN Markov vs Legacy HH (Strong Scaling Speedup):** {test_name}, p={p_value:.4f} ({sig_symbol} significant).")
                            report.append(f"  * CTBN Markov has {abs(pct_diff):.1f}% {better_worse} speedup than Legacy HH at {max_proc} processes ({mean_ctbn:.2f}x vs {mean_hh:.2f}x)")
                        except (KeyError, AttributeError) as e:
                            report.append(f"\n- **CTBN Markov vs Legacy HH (Strong Scaling Speedup):** {test_name}, p={p_value:.4f} ({sig_symbol} significant). (Error calculating means: {str(e)})")
                    
                    # Compare CTBN vs Legacy Markov strong scaling efficiency
                    ctbn_vs_markov_strong = perform_pairwise_comparison(df_parallel_max, 'parallel_efficiency', 'ctbn_stiff', 'legacy_markov')
                    if 'comparison' in ctbn_vs_markov_strong and 'error' not in ctbn_vs_markov_strong['comparison']:
                        p_value = ctbn_vs_markov_strong['comparison'].get('p-value', ctbn_vs_markov_strong['comparison'].get('p_value'))
                        test_name = ctbn_vs_markov_strong['comparison'].get('test', 'Statistical test')
                        significant = ctbn_vs_markov_strong['comparison'].get('significant', False)
                        sig_symbol = '*' if significant else 'not'
                        
                        # Get mean values
                        try:
                            mean_ctbn = ctbn_vs_markov_strong['descriptive'].loc['ctbn_stiff']['mean']
                            mean_markov = ctbn_vs_markov_strong['descriptive'].loc['legacy_markov']['mean']
                            pct_diff = ((mean_ctbn - mean_markov) / mean_markov) * 100
                            better_worse = "better" if pct_diff > 0 else "worse"
                            
                            report.append(f"\n- **CTBN Markov vs Legacy Markov (Strong Scaling Efficiency):** {test_name}, p={p_value:.4f} ({sig_symbol} significant).")
                            report.append(f"  * CTBN Markov has {abs(pct_diff):.1f}% {better_worse} strong scaling efficiency than Legacy Markov at {max_proc} processes ({mean_ctbn:.2f} vs {mean_markov:.2f})")
                        except (KeyError, AttributeError) as e:
                            report.append(f"\n- **CTBN Markov vs Legacy Markov (Strong Scaling Efficiency):** {test_name}, p={p_value:.4f} ({sig_symbol} significant). (Error calculating means: {str(e)})")
                    
                    # Strong scaling speedup comparison for CTBN vs Legacy Markov
                    ctbn_vs_markov_speedup = perform_pairwise_comparison(df_parallel_max, 'speedup_factor', 'ctbn_stiff', 'legacy_markov')
                    if 'comparison' in ctbn_vs_markov_speedup and 'error' not in ctbn_vs_markov_speedup['comparison']:
                        p_value = ctbn_vs_markov_speedup['comparison'].get('p-value', ctbn_vs_markov_speedup['comparison'].get('p_value'))
                        test_name = ctbn_vs_markov_speedup['comparison'].get('test', 'Statistical test')
                        significant = ctbn_vs_markov_speedup['comparison'].get('significant', False)
                        sig_symbol = '*' if significant else 'not'
                        
                        # Get mean values
                        try:
                            mean_ctbn = ctbn_vs_markov_speedup['descriptive'].loc['ctbn_stiff']['mean']
                            mean_markov = ctbn_vs_markov_speedup['descriptive'].loc['legacy_markov']['mean']
                            pct_diff = ((mean_ctbn - mean_markov) / mean_markov) * 100
                            better_worse = "better" if pct_diff > 0 else "worse"
                            
                            report.append(f"\n- **CTBN Markov vs Legacy Markov (Strong Scaling Speedup):** {test_name}, p={p_value:.4f} ({sig_symbol} significant).")
                            report.append(f"  * CTBN Markov has {abs(pct_diff):.1f}% {better_worse} speedup than Legacy Markov at {max_proc} processes ({mean_ctbn:.2f}x vs {mean_markov:.2f}x)")
                        except (KeyError, AttributeError) as e:
                            report.append(f"\n- **CTBN Markov vs Legacy Markov (Strong Scaling Speedup):** {test_name}, p={p_value:.4f} ({sig_symbol} significant). (Error calculating means: {str(e)})")
    # Write report to file
    report_path = 'data/benchmark/report/benchmark_results.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report generated: {report_path}")
    return '\n'.join(report)

def get_model_description(model_id):
    """
    Get detailed description of each model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        str: Detailed description of the model
    """
    descriptions = {
        'ctbn_stiff': "Continuous-Time Bayesian Network model with stiff equation solver. Implements advanced probabilistic modeling with direct representation of transition dynamics.",
        'legacy_markov': "Traditional Markov chain model using matrix exponential for state transitions. Well-established approach with broad literature support.",
        'legacy_hh': "Classical Hodgkin-Huxley formulation using differential equations for gating variables. Computationally efficient with minimal state representation."
    }
    
    return descriptions.get(model_id, "No description available")

def benchmark_all_models(models, output_dir='data/benchmark'):
    """
    Run all benchmarks for the given models and generate a comprehensive report.
    
    Args:
        models: Dictionary of model instances to benchmark
        output_dir: Directory to save benchmark results
        
    Returns:
        str: Path to generated report
    """
    print("Running comprehensive benchmarks for all models...")
    
    # Create output directories
    os.makedirs(f'{output_dir}/data', exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    os.makedirs(f'{output_dir}/report', exist_ok=True)
    
    # 1. Computational efficiency benchmarks
    print("\n1. Running computational efficiency benchmarks...")
    df_comp = benchmark_computational_efficiency(models)
    
    # 2. Weak scaling benchmarks (if models support it)
    print("\n2. Running weak scaling benchmarks...")
    try:
        df_weak = benchmark_weak_scaling(models)
    except Exception as e:
        print(f"  Weak scaling benchmarks failed: {e}")
        df_weak = None
    
    # 3. Parallel scaling benchmarks (if models support it)
    print("\n3. Running parallel scaling benchmarks...")
    try:
        df_parallel = benchmark_parallel_scaling(models)
    except Exception as e:
        print(f"  Parallel scaling benchmarks failed: {e}")
        df_parallel = None
    
    # Generate comprehensive report
    print("\nGenerating comprehensive benchmark report...")
    report = generate_results_report(df_comp, df_weak, df_parallel)
    
    report_path = f'{output_dir}/report/benchmark_results.md'
    print(f"\nBenchmarking complete! Report available at: {report_path}")
    
    return report_path


if __name__ == "__main__":
    """
    Main function to run benchmarks and generate reports
    """
    import sys
    import os
    from legacy_hh import HHModel
    from legacy_markov import AnticonvulsantMarkovModel
    from ctbn_markov import AnticonvulsantCTBNMarkovModel
    
    # Create output directory
    os.makedirs('data/benchmark', exist_ok=True)
    
    print("Sodium Channel Model Benchmarking Suite")
    print("====================================")
    
    # Initialize models
    print("Initializing models...")
    models = {
        'legacy_hh': HHModel(),
        'legacy_markov': AnticonvulsantMarkovModel(),
        'ctbn_stiff': AnticonvulsantCTBNMarkovModel()
    }
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--full' or sys.argv[1] == '-f':
            # Run comprehensive benchmarks
            print("Running comprehensive benchmark suite...")
            report_path = benchmark_all_models(models)
            print(f"\nComplete benchmark report generated at: {report_path}")
        elif sys.argv[1] == '--comp' or sys.argv[1] == '-c':
            # Run only computational efficiency benchmarks
            print("Running computational efficiency benchmarks...")
            df_comp = benchmark_computational_efficiency(models)
            figures = visualize_computational_efficiency(df_comp)
            print(f"Figures saved to data/benchmark/figures/")
        elif sys.argv[1] == '--scale' or sys.argv[1] == '-s':
            # Run only scaling benchmarks
            print("Running scaling benchmarks...")
            df_weak = benchmark_weak_scaling(models)
            df_parallel = benchmark_parallel_scaling(models)
            figures = visualize_scaling(df_weak, df_parallel)
            print(f"Figures saved to data/benchmark/figures/")
        elif sys.argv[1] == '--report-only' or sys.argv[1] == '-r':
            # Generate report from existing CSV files without running benchmarks
            print("Generating report from existing CSV files...")
            
            data_dir = 'data/benchmark/data'
            try:
                # Load existing CSV files
                print("Loading existing benchmark data...")
                
                # Computational efficiency data
                comp_path = os.path.join(data_dir, 'anticonvulsant_computational_efficiency.csv')
                if os.path.exists(comp_path):
                    df_comp = pd.read_csv(comp_path)
                    print(f"Loaded computational efficiency data from {comp_path}")
                    # Generate visualizations
                    print("DEBUG: Main - About to call visualize_computational_efficiency")
                    visualize_computational_efficiency(df_comp)
                    print("DEBUG: Main - Returned from visualize_computational_efficiency")
                else:
                    df_comp = None
                    print(f"Warning: Could not find computational efficiency data at {comp_path}")
                
                # Weak scaling data
                weak_path = os.path.join(data_dir, 'anticonvulsant_weak_scaling.csv')
                if os.path.exists(weak_path):
                    df_weak = pd.read_csv(weak_path)
                    print(f"Loaded weak scaling data from {weak_path}")
                else:
                    df_weak = None
                    print(f"Warning: Could not find weak scaling data at {weak_path}")
                
                # Parallel scaling data
                parallel_path = os.path.join(data_dir, 'parallel_scaling.csv')
                if os.path.exists(parallel_path):
                    df_parallel = pd.read_csv(parallel_path)
                    print(f"Loaded parallel scaling data from {parallel_path}")
                else:
                    df_parallel = None
                    print(f"Warning: Could not find parallel scaling data at {parallel_path}")
                
                # Generate visualizations
                if df_weak is not None or df_parallel is not None:
                    print("DEBUG: Main - About to call visualize_scaling")
                    visualize_scaling(df_weak, df_parallel)
                    print("DEBUG: Main - Returned from visualize_scaling")
                    

                # Generate report
                print("DEBUG: Main - About to call generate_full_report")
                report_path = generate_full_report(df_comp, df_weak, df_parallel)
                print("DEBUG: Main - Returned from generate_full_report")
                print(f"\nComplete benchmark report generated at: {report_path}")
                
            except Exception as e:
                print(f"Error generating report from existing data: {str(e)}")
                import traceback
                traceback.print_exc()
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            # Show help
            print("Usage: python comparison.py [option]")
            print("Options:")
            print("  --full, -f      Run all benchmarks and generate a comprehensive report")
            print("  --comp, -c      Run only computational efficiency benchmarks")
            print("  --scale, -s     Run only scaling benchmarks")
            print("  --report-only, -r  Generate report from existing CSV files without running benchmarks")
            print("  --help, -h      Show this help message")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Default: run a simple comparative benchmark
        print("Running a basic comparative benchmark (use --help for more options)...")
        df_comp = benchmark_computational_efficiency(models)
        figures = visualize_computational_efficiency(df_comp)
        print("\nBasic computational efficiency comparison:")
        
        # Show a simple summary of results
        # Print column names for debugging
        print("\nDataFrame columns:", df_comp.columns.tolist())
        
        summary = df_comp.groupby('model').agg({
            'execution_time_ms': ['mean', 'std'],
            'peak_memory_mb': ['mean', 'std']
        })
        
        print("\nExecution time (lower is better):")
        for model in summary.index:
            mean = summary.loc[model, ('execution_time_ms', 'mean')]
            std = summary.loc[model, ('execution_time_ms', 'std')]
            print(f"  {MODEL_NAMES.get(model, model)}: {mean:.2f} ms ± {std:.2f}")
            
        print("\nMemory usage (lower is better):")
        for model in summary.index:
            mean = summary.loc[model, ('peak_memory_mb', 'mean')]
            std = summary.loc[model, ('peak_memory_mb', 'std')]
            print(f"  {MODEL_NAMES.get(model, model)}: {mean:.2f} MB ± {std:.2f}")
            
        print(f"\nFigures saved to data/benchmark/figures/")