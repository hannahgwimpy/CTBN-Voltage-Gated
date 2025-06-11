# worker.py
import gc
import numpy as np
from ctbn_markov import CTBNMarkovModel, AnticonvulsantCTBNMarkovModel
from legacy_markov import MarkovModel, AnticonvulsantMarkovModel
from legacy_hh import HHModel

def run_single_sweep(args):
    """
    Runs a single simulation sweep for a specified ion channel model.

    This function is designed to be executed in a separate process (e.g., via
    multiprocessing) to avoid blocking the main GUI thread. It takes all
    necessary information for a single sweep, performs the simulation, and
    returns the results.

    Args:
        args (tuple): A tuple containing the following three elements:
            - sweep_no (int): The index of the current sweep to be run.
            - parameters (dict): A dictionary containing the model parameters.
              Special keys 'is_hh_model' (bool) and 'use_ctbn' (bool) are
              used to determine which model class (HHModel, CTBNMarkovModel,
              or MarkovModel) to instantiate. These keys are removed before
              setting attributes on the model instance.
            - swp_seq (list): A list of dictionaries, where each dictionary
              defines a voltage step (epoch) for the sweep protocol. Each
              dictionary should contain keys like 'holding', 'holding_duration',
              'test', 'test_duration', 'tail', 'tail_duration'.

    Returns:
        dict: A dictionary containing the results of the simulation sweep:
            - 'sweep_no' (int): The original sweep number.
            - 'sim_swp' (numpy.ndarray): The simulated current trace (in pA).
              Returns an empty array if an error occurred.
            - 'step_volt' (float): The test voltage (in mV) applied during
              this specific sweep.
            - 'time' (numpy.ndarray): An array of time points (in ms)
              corresponding to the `sim_swp`. Returns an empty array if an
              error occurred.

    The function performs the following steps:
    1. Unpacks `args` into `sweep_no`, `parameters`, and `swp_seq`.
    2. Determines the model type based on 'is_hh_model' and 'use_ctbn' flags
       in `parameters` and instantiates the corresponding model.
    3. Sets the provided parameters on the model instance.
    4. Initializes model-specific properties (e.g., rate constants).
    5. Formats the `swp_seq` into a NumPy array (`model_swp`) compatible
       with the `Sweep` method of the selected model. This format typically
       includes the number of epochs and pairs of (start_time_samples, voltage)
       for each epoch.
    6. Calls the `Sweep(sweep_no)` method on the model instance.
    7. Extracts the simulation output (`SimSwp`) and generates a time array.
    8. Handles exceptions that may occur during any step, printing error
       information and returning a minimal result dictionary.
    9. Ensures garbage collection (`gc.collect()`) is called in a `finally`
       block to manage memory, especially when run in worker processes.
    """
    sweep_no, parameters, swp_seq = args
    try:
        # Determine model type based on parameters
        is_hh_model = parameters.get('is_hh_model', False)
        use_ctbn = parameters.get('use_ctbn', False)
        is_anticonvulsant_model = parameters.get('is_anticonvulsant_model', False)

        if is_hh_model:
            model = HHModel()
        elif is_anticonvulsant_model:
            if use_ctbn:
                model = AnticonvulsantCTBNMarkovModel()
            else:
                model = AnticonvulsantMarkovModel()
        elif use_ctbn:
            model = CTBNMarkovModel()
        else:
            model = MarkovModel()

        # Remove model type flags from parameters
        model_flags = ['is_hh_model', 'use_ctbn', 'is_anticonvulsant_model']
        parameters = {k: v for k, v in parameters.items() if k not in model_flags}

        # Set parameters
        for param, value in parameters.items():
            setattr(model, param, value)

        # Update model parameters based on model type
        if isinstance(model, HHModel):
            # Update rate constants for HH model
            model.initialize_rate_constants()
        elif isinstance(model, CTBNMarkovModel):
            # Update rate constants for CTBN model
            # The CTBN model computes rates on-the-fly during Sweep()
            pass
        else:  # Legacy MarkovModel
            # Update rate constants
            model.stRatesVolt()
            model.CurrVolt()

        # Set protocol and run sweep based on model type
        # Both models now use the same protocol format (numpy array)
        if isinstance(swp_seq, list) and len(swp_seq) > 0:
            num_swps = len(swp_seq)
            # Create array with correct shape for both models: 
            # First row is number of epochs (3 per sweep)
            # Following rows are voltage and duration points in pairs
            model_swp = np.zeros((8, num_swps))  # 8 rows for 3 epochs (2 parameters per epoch + 1 row for epoch count + 1 for total points)

            for i, sweep in enumerate(swp_seq):
                # Match MarkovModel.MkSwpSeqMultiStep structure exactly
                model_swp[0, i] = 3  # Number of epochs (holding, test, tail)

                # Holding period
                model_swp[1, i] = 0  # First epoch starts at t=0
                model_swp[2, i] = sweep.get('holding', -80)  # Holding voltage

                # Test period
                holding_samples = int(sweep.get('holding_duration', 5) / 0.005)
                model_swp[3, i] = holding_samples  # Holding duration in samples
                model_swp[4, i] = sweep.get('test', 0)  # Test voltage

                # Tail period
                test_samples = int(sweep.get('test_duration', 20) / 0.005)
                model_swp[5, i] = holding_samples + test_samples  # Test end in samples
                model_swp[6, i] = sweep.get('tail', -80)  # Tail voltage

                # Total trace length
                tail_samples = int(sweep.get('tail_duration', 5) / 0.005)
                model_swp[7, i] = holding_samples + test_samples + tail_samples  # Total samples

            # Set the properly formatted protocol
            model.SwpSeq = model_swp
            model.NumSwps = num_swps

            # Run the sweep and extract step voltage for results
            # The worker model only has one sweep, so we always get the step voltage from index 0.
            step_volt = model_swp[4, 0]

            try:
                # Run the sweep - CTBNMarkovModel Sweep method no longer takes use_direct_update parameter
                # The model instance in the worker only knows about a single sweep protocol,
                # so we must always call Sweep with index 0, regardless of model type.
                if isinstance(model, CTBNMarkovModel):
                    model.Sweep(0)
                else:
                    model.Sweep(0)

                # Extract the peak current from SimSwp (should be negative for sodium channels)
                if hasattr(model, 'SimSwp'):
                    # For sodium channels, we want the most negative value (peak inward current)
                    peak_current = np.min(model.SimSwp)
                    # If all currents are positive, use max instead
                    if peak_current == 0 and np.max(model.SimSwp) > 0:
                        peak_current = np.max(model.SimSwp)
                # Extract the current trace 
                if hasattr(model, 'SimSwp'):
                    # Verify we're getting actual current values, not command voltage
                    min_current = np.min(model.SimSwp)
                    max_current = np.max(model.SimSwp)

                # Make sure we're returning current, not command voltage
                return {
                    'sweep_no': sweep_no,
                    'sim_swp': model.SimSwp.copy() if hasattr(model, 'SimSwp') else np.array([]),
                    'step_volt': step_volt,
                    'time': np.arange(len(model.SimSwp)) * 0.005 if hasattr(model, 'SimSwp') else np.array([]),
                    'protocol': swp_seq[0]
                }
            except Exception as e:
                print(f"Error running sweep {sweep_no}: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    'sweep_no': sweep_no,
                    'sim_swp': np.array([]),
                    'step_volt': step_volt,
                    'time': np.array([])
                }

        # If we get here, something went wrong with protocol format
        print(f"Protocol format error in sweep {sweep_no}")
        return {
            'sweep_no': sweep_no,
            'sim_swp': np.array([]),
            'step_volt': 0,
            'time': np.array([])
        }

    except Exception as e:
        print(f"Worker error in sweep {sweep_no}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return minimal data to avoid breaking the main thread
        return {
            'sweep_no': sweep_no,
            'sim_swp': np.array([]),
            'step_volt': 0,
            'time': np.array([])
        }
    finally:
        del model
        gc.collect()
