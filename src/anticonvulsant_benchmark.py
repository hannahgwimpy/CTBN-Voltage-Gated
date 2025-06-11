import sys
import os

# Ensure the project's root directory (parent of 'src') is in PYTHONPATH
# so that 'from src...' imports work correctly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import time
import numpy as np

# Assuming the script is run from the root of the CTBN-Voltage-Gated directory
# or that the src directory is in PYTHONPATH
from src.ctbn_markov import AnticonvulsantCTBNMarkovModel
from src.legacy_markov import AnticonvulsantMarkovModel

def run_benchmark(model_class, model_name, drug_concentration=10.0, drug_type='CBZ', repetitions=20):
    """Runs a benchmark for a given anticonvulsant model class."""
    print(f"Benchmarking {model_name}...")
    
    # Instantiate the model
    try:
        model = model_class(drug_concentration=drug_concentration, drug_type=drug_type)
        # Ensure a default protocol exists and is set up
        if not hasattr(model, 'SwpSeq') or model.SwpSeq is None:
            if hasattr(model, 'create_default_protocol'):
                model.create_default_protocol() # Standard default protocol
            elif hasattr(model, 'create_activation_protocol'): # Fallback if default is missing
                 model.create_activation_protocol()
            else:
                print(f"Could not find a suitable default protocol for {model_name}.")
                return None, None
        
        if model.NumSwps == 0:
            print(f"Model {model_name} has 0 sweeps in its default protocol.")
            return None, None
            
    except Exception as e:
        print(f"Error instantiating {model_name}: {e}")
        return None, None

    total_time = 0
    
    # Warm-up run (optional, but can help stabilize timings)
    try:
        model.Sweep(0)
    except Exception as e:
        print(f"Error during warm-up run for {model_name}, sweep 0: {e}")
        return None, None

    # Actual benchmark runs
    for i in range(repetitions):
        start_time = time.perf_counter()
        try:
            model.Sweep(0) # Run the first sweep of the current protocol
        except Exception as e:
            print(f"Error during {model_name} Sweep(0) on repetition {i+1}: {e}")
            return None, None # Abort benchmark for this model on error
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        
    avg_time_per_sweep = total_time / repetitions
    print(f"{model_name} - Average time per Sweep(0): {avg_time_per_sweep:.6f} seconds (over {repetitions} repetitions)")
    return avg_time_per_sweep, repetitions

if __name__ == "__main__":
    print("Starting Anticonvulsant Model Benchmark...")
    
    # Parameters for the benchmark
    repetitions = 20 # Number of times to run Sweep(0) for timing
    drug_conc = 25.0   # Example drug concentration in uM
    drug_t = 'LTG'    # Example drug type

    # Benchmark AnticonvulsantCTBNMarkovModel
    ctbn_avg_time, ctbn_reps = run_benchmark(
        AnticonvulsantCTBNMarkovModel, 
        "AnticonvulsantCTBNMarkovModel", 
        drug_concentration=drug_conc, 
        drug_type=drug_t, 
        repetitions=repetitions
    )

    # Benchmark AnticonvulsantMarkovModel (legacy)
    legacy_avg_time, legacy_reps = run_benchmark(
        AnticonvulsantMarkovModel, 
        "AnticonvulsantMarkovModel (Legacy)", 
        drug_concentration=drug_conc, 
        drug_type=drug_t, 
        repetitions=repetitions
    )
    
    print("\nBenchmark Summary:")
    if ctbn_avg_time is not None:
        print(f"- AnticonvulsantCTBNMarkovModel:      {ctbn_avg_time:.6f} s/sweep ({ctbn_reps} reps)")
    else:
        print("- AnticonvulsantCTBNMarkovModel:      Benchmark failed or was skipped.")
        
    if legacy_avg_time is not None:
        print(f"- AnticonvulsantMarkovModel (Legacy): {legacy_avg_time:.6f} s/sweep ({legacy_reps} reps)")
    else:
        print("- AnticonvulsantMarkovModel (Legacy): Benchmark failed or was skipped.")

    if ctbn_avg_time is not None and legacy_avg_time is not None:
        if legacy_avg_time > 0: # Avoid division by zero
            speedup_factor = legacy_avg_time / ctbn_avg_time
            print(f"\nSpeedup of CTBN over Legacy: {speedup_factor:.2f}x")
        else:
            print("\nCannot calculate speedup due to zero execution time for legacy model.")
    
    print("\nBenchmark complete.")
