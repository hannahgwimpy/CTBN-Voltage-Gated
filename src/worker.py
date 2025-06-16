"""
Defines the worker function for parallelized ion channel simulations.

This module contains `run_single_sweep`, a function designed to be executed
by individual processes in a multiprocessing pool. It handles the instantiation,
parameterization, and execution of a single voltage clamp simulation sweep
for various ion channel models (Hodgkin-Huxley, Markov, CTBN-based Markov,
and their anticonvulsant-sensitive variants).
"""
import gc
import numpy as np
from ctbn_markov import CTBNMarkovModel, AnticonvulsantCTBNMarkovModel
from legacy_markov import MarkovModel, AnticonvulsantMarkovModel
from legacy_hh import HHModel
def run_single_sweep(args):
    """
    Executes a single voltage clamp simulation sweep for a given model and protocol.

    This function is designed to be called by a multiprocessing Pool, typically
    from `IonChannelGUI.run_simulation_thread` in `main.py`. It handles the
    instantiation, parameterization, and simulation of one specific sweep.

    Args:
        args (tuple): A tuple containing:
            - sweep_no (int): The identifier/index of this sweep in a larger sequence.
            - parameters (dict): A dictionary containing model parameters.
                This includes flags like 'is_hh_model', 'use_ctbn',
                'is_anticonvulsant_model', 'drug_type', 'drug_concentration',
                as well as biophysical parameters for the model.
            - swp_seq (list): A list containing a single dictionary that defines
                the voltage clamp protocol for this specific sweep (e.g.,
                holding/test/tail voltages and durations).

    Returns:
        dict: A dictionary containing the simulation results for the sweep:
            - 'sweep_no' (int): The input sweep number.
            - 'sim_swp' (np.ndarray): The simulated current trace. Empty if error.
            - 'step_volt' (float): The main test voltage of the sweep. 0 if error.
            - 'time' (np.ndarray): The time vector for the simulation. Empty if error.
            - 'protocol' (dict): The input protocol dictionary for this sweep.
            Returns a dictionary with empty arrays and default values upon error.
    """
    (sweep_no, parameters, swp_seq) = args
    try:
        is_hh_model = parameters.get('is_hh_model', False)
        use_ctbn = parameters.get('use_ctbn', False)
        is_anticonvulsant_model = parameters.get('is_anticonvulsant_model', False)
        if is_hh_model:
            model = HHModel()
        elif is_anticonvulsant_model:
            drug_type = parameters.get('drug_type', 'mixed')
            drug_concentration = parameters.get('drug_concentration', 0.0)
            if use_ctbn:
                model = AnticonvulsantCTBNMarkovModel(drug_concentration=drug_concentration, drug_type=drug_type)
            else:
                model = AnticonvulsantMarkovModel(drug_concentration=drug_concentration, drug_type=drug_type)
        elif use_ctbn:
            model = CTBNMarkovModel()
        else:
            model = MarkovModel()
        model_flags = ['is_hh_model', 'use_ctbn', 'is_anticonvulsant_model', 'drug_type', 'drug_concentration']
        parameters = {k: v for (k, v) in parameters.items() if (k not in model_flags)}
        for (param, value) in parameters.items():
            setattr(model, param, value)
        if isinstance(model, HHModel):
            model.initialize_rate_constants()
        elif isinstance(model, CTBNMarkovModel):
            pass
        else:
            model.stRatesVolt()
            model.CurrVolt()
        if (isinstance(swp_seq, list) and (len(swp_seq) > 0)):
            num_swps = len(swp_seq)
            model_swp = np.zeros((8, num_swps))
            for (i, sweep) in enumerate(swp_seq):
                model_swp[(0, i)] = 3
                model_swp[(1, i)] = 0
                model_swp[(2, i)] = sweep.get('holding', (- 80))
                holding_samples = int((sweep.get('holding_duration', 5) / 0.005))
                model_swp[(3, i)] = holding_samples
                model_swp[(4, i)] = sweep.get('test', 0)
                test_samples = int((sweep.get('test_duration', 20) / 0.005))
                model_swp[(5, i)] = (holding_samples + test_samples)
                model_swp[(6, i)] = sweep.get('tail', (- 80))
                tail_samples = int((sweep.get('tail_duration', 5) / 0.005))
                model_swp[(7, i)] = ((holding_samples + test_samples) + tail_samples)
            model.SwpSeq = model_swp
            model.NumSwps = num_swps
            step_volt = model_swp[(4, 0)]
            try:
                if isinstance(model, CTBNMarkovModel):
                    model.Sweep(0)
                else:
                    model.Sweep(0)
                if hasattr(model, 'SimSwp'):
                    peak_current = np.min(model.SimSwp)
                    if ((peak_current == 0) and (np.max(model.SimSwp) > 0)):
                        peak_current = np.max(model.SimSwp)
                if hasattr(model, 'SimSwp'):
                    min_current = np.min(model.SimSwp)
                    max_current = np.max(model.SimSwp)
                return {'sweep_no': sweep_no, 'sim_swp': (model.SimSwp.copy() if hasattr(model, 'SimSwp') else np.array([])), 'step_volt': step_volt, 'time': ((np.arange(len(model.SimSwp)) * 0.005) if hasattr(model, 'SimSwp') else np.array([])), 'protocol': swp_seq[0]}
            except Exception as e:
                print(f'Error running sweep {sweep_no}: {str(e)}')
                import traceback
                traceback.print_exc()
                return {'sweep_no': sweep_no, 'sim_swp': np.array([]), 'step_volt': step_volt, 'time': np.array([])}
        print(f'Protocol format error in sweep {sweep_no}')
        return {'sweep_no': sweep_no, 'sim_swp': np.array([]), 'step_volt': 0, 'time': np.array([])}
    except Exception as e:
        print(f'Worker error in sweep {sweep_no}: {str(e)}')
        import traceback
        traceback.print_exc()
        return {'sweep_no': sweep_no, 'sim_swp': np.array([]), 'step_volt': 0, 'time': np.array([])}
    finally:
        del model
        gc.collect()
