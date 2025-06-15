import dearpygui.dearpygui as dpg
import numpy as np
import sys
import threading
import gc
from multiprocessing import Pool, freeze_support
from worker import run_single_sweep
from ctbn_markov import CTBNMarkovModel, AnticonvulsantCTBNMarkovModel
from legacy_markov import MarkovModel, AnticonvulsantMarkovModel
from legacy_hh import HHModel
class IonChannelGUI():
    def __init__(self):
        dpg.create_context()
        self.drug_types = ['CBZ', 'LTG', 'DPH']
        self.ctbn_markov_model = CTBNMarkovModel()
        self.legacy_markov_model = MarkovModel()
        self.legacy_hh_model = HHModel()
        self.anticonvulsant_markov_model = AnticonvulsantMarkovModel(drug_type='DPH')
        self.anticonvulsant_ctbn_markov_model = AnticonvulsantCTBNMarkovModel(drug_type='DPH')
        self.current_model = self.ctbn_markov_model
        self.current_model_name = 'CTBN Markov'
        self.markov_parameters = ['alcoeff', 'alslp', 'btcoeff', 'btslp', 'gmcoeff', 'gmslp', 'dlcoeff', 'dlslp', 'ConCoeff', 'CoffCoeff', 'OpOnCoeff', 'OpOffCoeff']
        self.hh_parameters = ['g_Na', 'E_Na', 'C_m', 'numchan']
        self.anticonvulsant_markov_parameters = (self.markov_parameters + ['drug_concentration'])
        self.parameter_names = self.markov_parameters
        self.markov_parameter_info = {'alcoeff': {'desc': 'alpha coefficient', 'bounds': (64, 96)}, 'alslp': {'desc': 'alpha voltage dependence', 'bounds': (6.4, 9.6)}, 'btcoeff': {'desc': 'beta coefficient', 'bounds': (0.64, 0.96)}, 'btslp': {'desc': 'beta voltage dependence', 'bounds': (12, 18)}, 'gmcoeff': {'desc': 'gamma coefficient', 'bounds': (120, 180)}, 'gmslp': {'desc': 'gamma voltage dependence', 'bounds': (6.4, 9.6)}, 'dlcoeff': {'desc': 'delta coefficient', 'bounds': (32, 48)}, 'dlslp': {'desc': 'delta voltage dependence', 'bounds': (12, 18)}, 'ConCoeff': {'desc': 'Base konlo', 'bounds': (0.016, 0.024)}, 'CoffCoeff': {'desc': 'Base kofflo', 'bounds': (0.16, 0.24)}, 'OpOnCoeff': {'desc': 'Base konOp', 'bounds': (0.6, 0.9)}, 'OpOffCoeff': {'desc': 'Base koffOp', 'bounds': (0.004, 0.006)}}
        self.hh_parameter_info = {'g_Na': {'desc': 'Max Na Conductance (mS/cm²)', 'bounds': (0.05, 0.25)}, 'E_Na': {'desc': 'Na Reversal Potential (mV)', 'bounds': (40, 60)}, 'C_m': {'desc': 'Membrane Capacitance (μF/cm²)', 'bounds': (0.8, 1.2)}, 'numchan': {'desc': 'Number of Channels', 'bounds': (1, 1000)}}
        self.anticonvulsant_markov_parameter_info = self.markov_parameter_info.copy()
        self.anticonvulsant_markov_parameter_info.update({'drug_concentration': {'desc': 'Drug Concentration (μM)', 'bounds': (0, 100)}})
        self.parameter_info = self.markov_parameter_info
        self.sim_results = []
        self.spont_ap_results = None
        self.evoked_ap_results = None
        self.temp_scaled_data = []
        self.temp_sim_scaled_data = []
        self.voltage_step_tags = []
        dpg.create_viewport(title='Ion Channel Simulator', width=1400, height=800, resizable=True, decorated=True, vsync=True)
        dpg.setup_dearpygui()
        with dpg.window(label='Main Window', autosize=True, no_resize=False, no_title_bar=True, no_move=True, tag='primary_window'):
            self.setup_gui()
        dpg.show_viewport()
        if (sys.platform == 'darwin'):
            import objc
            import AppKit
            window = AppKit.NSApp().mainWindow()
            if window:
                window.zoom_(None)
        else:
            dpg.maximize_viewport()
    def set_drug_type(self, drug_type):
        self.drug_type = drug_type.upper()
        self.init_parameters()
        self.stRatesVolt()
    def on_drug_type_change(self, sender, app_data, user_data):
        drug_type = app_data
        if hasattr(self.current_model, 'set_drug_type'):
            self.current_model.set_drug_type(drug_type)
            self.setup_parameters()
        print(f'Drug type changed to: {drug_type}')
    def save_plot_to_file(self, plot_type, plot_data):
        if ((plot_type != 'Current') and (plot_type != 'Current Traces')):
            print(f'Skipping auto-save for {plot_type} plot - should be saved via save_comparison_plots')
            return
        try:
            import subprocess
            import tempfile
            import json
            import os
            import sys
            temp_dir = tempfile.mkdtemp()
            data_file = os.path.join(temp_dir, 'plot_data.json')
            if ((plot_type == 'Current') and hasattr(self, 'current_model') and (self.current_model is not None)):
                if ('model_type' not in plot_data):
                    plot_data['model_type'] = self.current_model.__class__.__name__
                if ((('time_points' not in plot_data) or ('currents' not in plot_data)) and hasattr(self.current_model, 'SimTime') and hasattr(self.current_model, 'SimCur')):
                    try:
                        plot_data['time_points'] = (self.current_model.SimTime.tolist() if hasattr(self.current_model.SimTime, 'tolist') else list(self.current_model.SimTime))
                        plot_data['currents'] = [(self.current_model.SimCur.tolist() if hasattr(self.current_model.SimCur, 'tolist') else list(self.current_model.SimCur))]
                        if hasattr(self.current_model, 'SimCom'):
                            plot_data['voltages'] = [(self.current_model.SimCom.tolist() if hasattr(self.current_model.SimCom, 'tolist') else list(self.current_model.SimCom))]
                    except Exception as e:
                        print(f'Error accessing simulation data from model: {e}')
                        import traceback
                        traceback.print_exc()
            with open(data_file, 'w') as f:
                json.dump({'plot_type': plot_type, 'plot_data': plot_data}, f)
            script_file = os.path.join(temp_dir, 'save_plot.py')
            with open(script_file, 'w') as f:
                f.write('\nimport matplotlib\nmatplotlib.use(\'Agg\')  # Use non-interactive backend\nimport matplotlib.pyplot as plt\nimport json\nimport numpy as np\nimport gc\nimport os\nimport sys\nimport traceback\nfrom scipy.stats import pearsonr\n# Load the plot data\nwith open(sys.argv[1], \'r\') as f:\n    data = json.load(f)\nplot_type = data["plot_type"]\nplot_data = data["plot_data"]\n# Get the project root directory from command line arguments (passed from\n# the parent script)\nproject_root = sys.argv[2] if len(\n    sys.argv) > 2 else os.path.dirname(\n        os.path.dirname(\n            os.path.abspath(__file__)))\ndata_dir = os.path.join(project_root, "data", "currents")\nos.makedirs(data_dir, exist_ok=True)\n# Get model type for filename, defaulting to "unknown" if not provided\nmodel_type = plot_data.get(\'model_type\', \'unknown\').lower().replace(\' \', \'_\')\n# Create filename based on plot type and model type\nfilename = f"{plot_type.lower().replace(\' \', \'_\')}_{model_type}.png"\n# Full path to save the file\nfile_path = os.path.join(data_dir, filename)\n# Print where the file will be saved for user information\n# Create the appropriate plot based on the plot type\ntry:\n    if plot_type == "Current Traces":\n        # Extract data for current traces\n        time_points = plot_data.get(\'time_points\', [])\n        markov_current = plot_data.get(\'markov_current\', [])\n        hh_current = plot_data.get(\'hh_current\', [])\n        voltage = plot_data.get(\'voltage\', 0)\n        # Validate data before plotting\n        if len(time_points) == 0 or len(\n            markov_current) == 0 or len(hh_current) == 0:\n            sys.exit(1)\n        plt.figure(figsize=(10, 6))\n        plt.plot(\n    time_points,\n    markov_current,\n    label="Markov Model",\n     color="blue")\n        plt.plot(\n    time_points,\n    hh_current,\n    label="HH Model (Scaled)",\n     color="red")\n        plt.xlabel("Time (ms)")\n        plt.ylabel("Current (pA)")\n        plt.title(f"Ion Channel Current Traces at {voltage} mV")\n        plt.grid(True)\n        plt.legend()\n    elif plot_type == "Current":\n        # Extract current model data\n        time_points = plot_data.get(\'time_points\', [])\n        currents = plot_data.get(\'currents\', [])\n        voltages = plot_data.get(\'voltages\', [])\n        model_type = plot_data.get(\'model_type\', \'Unknown Model\')\n        # Validate data\n        if len(time_points) == 0 or len(currents) == 0 or len(voltages) == 0:\n            sys.exit(1)\n        # Create a figure with two subplots (voltage protocol and current\n        # responses)\n        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(\n            10, 8), gridspec_kw={\'height_ratios\': [1, 3]})\n        # Use a colormap to distinguish different voltage traces\n        import matplotlib.cm as cm\n        colors = cm.viridis(np.linspace(0, 1, len(currents)))\n        # 1. Top subplot: Command Voltage Protocol\n        ax1.set_title(\'Command Voltage Protocol\')\n        # Plot voltage protocols for each trace\n        for i, voltage in enumerate(voltages):\n            # Create simplified voltage protocol: hold at -80mV, step to test\n            # voltage, back to -80mV\n            holding_voltage = -80  # Default holding potential\n            holding_duration = 98  # ms\n            step_duration = 102    # ms\n            total_duration = 300   # ms\n            # Create voltage protocol time points and values\n            protocol_time = [\n    0,\n    holding_duration,\n    holding_duration,\n    holding_duration+\n    step_duration,\n    holding_duration+\n    step_duration,\n     total_duration]\n            protocol_voltage = [\n    holding_voltage,\n    holding_voltage,\n    voltage,\n    voltage,\n    holding_voltage,\n     holding_voltage]\n            # Plot with the same color as the corresponding current trace\n            ax1.plot(\n    protocol_time,\n    protocol_voltage,\n    color=colors[i],\n     label=f"{voltage} mV")\n        ax1.set_ylabel(\'Voltage (mV)\')\n        ax1.set_xlim(0, 300)\n        ax1.set_ylim(-120, 60)\n        ax1.grid(True)\n        # 2. Bottom subplot: Current Responses\n        ax2.set_title(f"{model_type} Current Responses")\n        # Plot each current trace with corresponding voltage label\n        for i, (current, voltage) in enumerate(zip(currents, voltages)):\n            # Ensure the time and current arrays have the same length\n            if len(time_points) != len(current):\n                # Use the shorter length\n                min_length = min(len(time_points), len(current))\n                adjusted_time = time_points[:min_length]\n                adjusted_current = current[:min_length]\n                ax2.plot(\n    adjusted_time,\n    adjusted_current,\n    color=colors[i],\n     label=f"{voltage} mV")\n            else:\n                # Use the arrays directly if they match\n                ax2.plot(\n    time_points,\n    current,\n    color=colors[i],\n     label=f"{voltage} mV")\n        ax2.set_xlabel("Time (ms)")\n        ax2.set_ylabel("Current (pA)")\n        ax2.grid(True, linestyle=\'--\', alpha=0.6)\n        ax2.set_xlim(0, 300)\n        # Add legend with voltage values to the bottom of the figure\n        handles, labels = ax2.get_legend_handles_labels()\n        fig.legend(handles, labels, loc=\'lower center\', bbox_to_anchor=(0.5, 0), ncol=7,\n                  title="Test Voltage", frameon=True)\n        # Add extra space at bottom for the legend\n        plt.subplots_adjust(bottom=0.15)\n        # Adjust spacing between subplots\n        plt.tight_layout(rect=[0, 0.05, 1, 0.95])\n    else:\n        sys.exit(1)\n    # Save the figure with high resolution\n    plt.savefig(file_path, dpi=300, bbox_inches=\'tight\', format=\'png\')\n    # Clean up\n    plt.close(\'all\')\n    gc.collect()\nexcept Exception as e:\n    traceback.print_exc()\n    plt.close(\'all\')\n    gc.collect()\n    sys.exit(1)\n    ')
            root_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(root_dir)
            data_dir = os.path.join(parent_dir, 'data', 'currents')
            os.makedirs(data_dir, exist_ok=True)
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            process = subprocess.Popen([sys.executable, script_file, data_file, project_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            try:
                (stdout, stderr) = process.communicate(timeout=5.0)
                if stdout:
                    print(f'Plot script output: {stdout}')
                if stderr:
                    print(f'Plot script error: {stderr}')
                if (hasattr(self, 'current_model') and (self.current_model is not None)):
                    model_type = self.current_model.__class__.__name__.lower().replace(' ', '_')
                else:
                    model_type = 'unknown'
                filename = f"{plot_type.lower().replace(' ', '_')}_{model_type}.png"
                plot_path = os.path.join(data_dir, filename)
                if os.path.exists(plot_path):
                    self.last_saved_plot = plot_path
                else:
                    print(f'Plot file was not created at: {plot_path}')
            except subprocess.TimeoutExpired:
                process.kill()
                print(f'Plot script timed out after 5 seconds, check logs for errors')
        except Exception as e:
            print(f'Error setting up save operation: {e}')
            import traceback
            traceback.print_exc()
    def setup_gui(self):
        with dpg.collapsing_header(label='Model Selection', default_open=True):
            dpg.add_radio_button(('CTBN Markov', 'Legacy Markov', 'Hodgkin-Huxley', 'Anticonvulsant Legacy Markov', 'Anticonvulsant CTBN Markov'), default_value='CTBN Markov', callback=self.on_model_change, tag='model_selector')
        with dpg.collapsing_header(label='Model Parameters', default_open=True, tag='parameters_header'):
            self.setup_parameters()
        with dpg.collapsing_header(label='Voltage Protocol', default_open=False, tag='protocol_header'):
            self.setup_protocol_widgets()
        with dpg.group(horizontal=True):
            dpg.add_button(label='Run Simulation', callback=self.run_simulation)
        with dpg.window(label='Plots', width=1100, height=800, pos=[300, 0]):
            with dpg.tab_bar():
                with dpg.tab(label='Current Traces'):
                    with dpg.plot(label='Command Voltage Protocol', height=120, width=(- 1), tag='command_voltage_plot'):
                        dpg.add_plot_legend(outside=True, tag='command_voltage_legend')
                        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='Time (ms)')
                        dpg.set_axis_limits(x_axis, 0, 300)
                        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label='Voltage (mV)')
                        dpg.set_axis_limits(y_axis, (- 120), 60)
                    with dpg.group(horizontal=True):
                        dpg.add_text('Current Responses', color=[0, 150, 255])
                    with dpg.plot(label='Current Responses', height=350, width=(- 1), tag='current_plot'):
                        dpg.add_plot_legend(outside=True, tag='current_plot_legend')
                        dpg.add_plot_axis(dpg.mvXAxis, label='Time (ms)', tag='current_plot_x_axis')
                        dpg.set_axis_limits('current_plot_x_axis', 0, 300)
                        dpg.add_plot_axis(dpg.mvYAxis, label='Current (pA)', tag='current_plot_y_axis')
                        dpg.set_axis_limits('current_plot_y_axis', (- 500), 50)
                        self.current_series = []
    def setup_parameters(self):
        if dpg.does_item_exist('param_group'):
            dpg.delete_item('param_group')
        with dpg.group(tag='param_group', parent='parameters_header'):
            model = self.current_model
            model_name = self.current_model_name
            is_markov = ('Markov' in model_name)
            is_anticonvulsant = ('Anticonvulsant' in model_name)
            is_hh = (model_name == 'Hodgkin-Huxley')
            param_width = 150
            if is_markov:
                dpg.add_text('Gate Parameters')
                gate_config_map = {'AL Gate': [('alcoeff', 'alpha coefficient'), ('alslp', 'alpha voltage depend')], 'BT Gate': [('btcoeff', 'beta coefficient'), ('btslp', 'beta voltage depend')], 'GM Gate': [('gmcoeff', 'gamma coefficient'), ('gmslp', 'gamma voltage depend')], 'DL Gate': [('dlcoeff', 'delta coefficient'), ('dlslp', 'delta voltage depend')]}
                for (display_name, params_info) in gate_config_map.items():
                    dpg.add_text(display_name)
                    for (attr_name, label_text) in params_info:
                        if hasattr(model, attr_name):
                            dpg.add_input_float(label=label_text, default_value=getattr(model, attr_name), callback=self.on_parameter_change, user_data=attr_name, tag=f'param_input_{attr_name}', width=param_width)
                dpg.add_separator()
                dpg.add_text('Transition Rate Parameters')
                transition_params_info = [('ConCoeff', 'Base konIo (ConCoeff)'), ('CoffCoeff', 'Base koffIo (CoffCoeff)'), ('OpOnCoeff', 'Base konOp (OpOnCoeff)'), ('OpOffCoeff', 'Base koffOp (OpOffCoeff)')]
                for (attr_name, label_text) in transition_params_info:
                    if hasattr(model, attr_name):
                        dpg.add_input_float(label=label_text, default_value=getattr(model, attr_name), callback=self.on_parameter_change, user_data=attr_name, tag=f'param_input_{attr_name}', width=param_width)
                dpg.add_separator()
                if hasattr(model, 'numchan'):
                    dpg.add_input_int(label='Number of Channels', default_value=int(getattr(model, 'numchan', 100)), callback=self.on_parameter_change, user_data='numchan', tag='param_input_numchan', width=param_width)
            if is_anticonvulsant:
                dpg.add_separator()
                dpg.add_text('Anticonvulsant Drug Parameters')
                current_drug_type = 'CBZ'
                if (hasattr(model, 'drug_type') and model.drug_type):
                    current_drug_type = model.drug_type
                elif hasattr(model, 'set_drug_type'):
                    model.set_drug_type(current_drug_type)
                    if hasattr(model, 'drug_type'):
                        current_drug_type = model.drug_type
                dpg.add_combo(items=['CBZ', 'LTG', 'DPH'], label='Drug Type', default_value=current_drug_type, callback=self.on_drug_type_change, tag='drug_type_combo')
                drug_params_display_to_attr = {'Drug Conc. (µM)': 'drug_concentration'}
                for (display_label, attr_name) in drug_params_display_to_attr.items():
                    if hasattr(model, attr_name):
                        default_val = getattr(model, attr_name)
                        if (default_val is None):
                            default_val = 0.0
                        dpg.add_input_float(label=display_label, default_value=float(default_val), callback=self.on_parameter_change, user_data=attr_name, tag=f'param_input_{attr_name}', width=param_width)
                    else:
                        print(f"Debug: Anticonvulsant model '{model_name}' is missing attribute '{attr_name}' for drug type '{current_drug_type}'")
            if (is_hh and (not is_markov)):
                dpg.add_text('Hodgkin-Huxley Parameters')
                for attr_name in self.parameter_names:
                    if hasattr(model, attr_name):
                        info = self.parameter_info.get(attr_name, {})
                        label_text = info.get('desc', attr_name)
                        if (attr_name == 'numchan'):
                            dpg.add_input_int(label=label_text, default_value=int(getattr(model, attr_name)), callback=self.on_parameter_change, user_data=attr_name, tag=f'param_input_{attr_name}', width=param_width)
                        else:
                            dpg.add_input_float(label=label_text, default_value=getattr(model, attr_name), callback=self.on_parameter_change, user_data=attr_name, tag=f'param_input_{attr_name}', width=param_width)
    def on_protocol_type_change(self, sender, app_data, user_data):
        protocol_type = app_data
        is_custom = (protocol_type == 'Custom')
        if dpg.does_item_exist('custom_protocol_widgets'):
            dpg.configure_item('custom_protocol_widgets', show=is_custom)
        if (not is_custom):
            if (protocol_type == 'Default'):
                self.current_model.create_default_protocol()
            elif (protocol_type == 'Inactivation'):
                self.current_model.create_inactivation_protocol()
            elif (protocol_type == 'Recovery'):
                self.current_model.create_recovery_protocol()
            elif (protocol_type == 'Steady-State Inactivation'):
                if hasattr(self.current_model, 'create_steady_state_inactivation_protocol'):
                    self.current_model.create_steady_state_inactivation_protocol()
                else:
                    print(f"Warning: {self.current_model_name} does not have 'create_steady_state_inactivation_protocol'.")
            print(f'{protocol_type} protocol applied.')
            self.update_plots()
    def on_parameter_change(self, sender, app_data, user_data):
        model = self.current_model
        param_key = user_data
        rates_updated_by_handler = False
        try:
            if (param_key == 'numchan'):
                setattr(model, param_key, int(app_data))
            elif ((param_key == 'drug_concentration') and ('Anticonvulsant' in self.current_model_name)):
                if hasattr(model, 'set_drug_concentration'):
                    model.set_drug_concentration(float(app_data)) # This calls update_rates internally
                    rates_updated_by_handler = True
                else:
                    setattr(model, param_key, float(app_data))
            else:
                setattr(model, param_key, float(app_data))
            self.update_plots(rates_already_updated=rates_updated_by_handler)
        except ValueError:
            print(f"Error: Invalid input value '{app_data}' for parameter '{param_key}'. Please enter a valid number.")
        except Exception as e:
            print(f'An error occurred in on_parameter_change for {param_key}: {e}')
    def setup_protocol_widgets(self):
        self.voltage_step_tags = []
        if dpg.does_item_exist('voltage_protocol_group'):
            dpg.delete_item('voltage_protocol_group')
        with dpg.group(parent='protocol_header', tag='voltage_protocol_group'):
            dpg.add_radio_button(['Default', 'Inactivation', 'Recovery', 'Steady-State Inactivation', 'Custom'], label='Protocol Type', callback=self.on_protocol_type_change, default_value='Default', tag='protocol_type_radio')
            with dpg.group(tag='custom_protocol_widgets', show=False):
                dpg.add_input_int(label='Holding Potential (mV)', default_value=(- 120), width=150, callback=self.on_protocol_change, tag='holding_potential')
                dpg.add_input_int(label='Prepulse Duration (ms)', default_value=100, width=150, callback=self.on_protocol_change, tag='prepulse_duration')
                dpg.add_input_int(label='Pulse Duration (ms)', default_value=50, width=150, callback=self.on_protocol_change, tag='pulse_duration')
                dpg.add_input_int(label='Postpulse Duration (ms)', default_value=50, width=150, callback=self.on_protocol_change, tag='postpulse_duration')
                dpg.add_separator()
                dpg.add_text('Voltage Steps')
                with dpg.group(tag='voltage_steps_group'):
                    tag = 'voltage_step_0'
                    dpg.add_input_int(label='Step 1', default_value=0, width=150, callback=self.on_protocol_change, tag=tag)
                    self.voltage_step_tags.append(tag)
                with dpg.group(horizontal=True):
                    dpg.add_button(label='Add Voltage Step', callback=self.add_voltage_step)
                    dpg.add_button(label='Remove Last Step', callback=self.remove_voltage_step)
                dpg.add_button(label='Apply Custom Protocol', callback=self.apply_voltage_protocol)
    def on_protocol_change(self, sender, value):
        print("Custom protocol modified. Click 'Apply Custom Protocol' to update.")
    def add_voltage_step(self):
        step_num = len(self.voltage_step_tags)
        tag = f'voltage_step_{step_num}'
        dpg.add_input_int(label=f'Step {(step_num + 1)}', default_value=0, width=150, callback=self.on_protocol_change, tag=tag, parent='voltage_steps_group')
        self.voltage_step_tags.append(tag)
    def remove_voltage_step(self):
        if (len(self.voltage_step_tags) > 1):
            tag_to_remove = self.voltage_step_tags.pop()
            if dpg.does_item_exist(tag_to_remove):
                dpg.delete_item(tag_to_remove)
        else:
            print('Cannot remove the last voltage step.')
    def apply_voltage_protocol(self):
        try:
            holding_potential = dpg.get_value('holding_potential')
            prepulse_duration = dpg.get_value('prepulse_duration')
            pulse_duration = dpg.get_value('pulse_duration')
            postpulse_duration = dpg.get_value('postpulse_duration')
            voltage_steps = [dpg.get_value(tag) for tag in self.voltage_step_tags]
            self.current_model.V_hold = holding_potential
            self.current_model.prepulse_duration = prepulse_duration
            self.current_model.pulse_duration = pulse_duration
            self.current_model.postpulse_duration = postpulse_duration
            self.current_model.voltages = voltage_steps
            self.current_model.makeprotocol()
            print('Custom protocol applied.')
            self.update_plots()
        except Exception as e:
            print(f'Error applying custom protocol: {e}')
    def update_plots(self, rates_already_updated=False):
        self._clear_all_plots()
        if hasattr(self, 'current_series'):
            self.current_series = []
        if hasattr(self, 'voltage_series'):
            self.voltage_series = []
        self.update_voltage_plot()
        if not rates_already_updated and hasattr(self.current_model, 'update_rates'):
            self.current_model.update_rates()
        self.update_current_plot()
    def update_voltage_plot(self):
        if ((not hasattr(self, 'sim_results')) or (not self.sim_results)):
            return
        if (not dpg.does_item_exist('voltage_plot_y_axis')):
            print('Could not find voltage plot y-axis')
            return
        sorted_results = sorted(self.sim_results, key=(lambda x: x.get('step_volt', 0)), reverse=True)
        for res in sorted_results:
            if (('protocol' not in res) or (not res['protocol'])):
                continue
            protocol = res['protocol']
            step_volt = res.get('step_volt', 0)
            holding_v = protocol.get('holding', (- 120))
            holding_dur = protocol.get('holding_duration', 100)
            test_v = protocol.get('test', 0)
            test_dur = protocol.get('test_duration', 200)
            tail_v = protocol.get('tail', (- 120))
            tail_dur = protocol.get('tail_duration', 0)
            time_points = [0]
            voltage_points = [holding_v]
            time_points.extend([holding_dur, holding_dur])
            voltage_points.extend([holding_v, test_v])
            time_points.extend([(holding_dur + test_dur), (holding_dur + test_dur)])
            voltage_points.extend([test_v, tail_v])
            total_duration = ((holding_dur + test_dur) + tail_dur)
            time_points.append(total_duration)
            voltage_points.append(tail_v)
            if (total_duration < 300):
                time_points.append(300)
                voltage_points.append(tail_v)
            label = f'{int(step_volt)} mV'
            dpg.add_line_series(x=time_points, y=voltage_points, label=label, parent='voltage_plot_y_axis')
    def _clear_all_plots(self):
        voltage_plot_tag = 'command_voltage_plot'
        voltage_y_axis_tag = 'voltage_plot_y_axis'
        voltage_legend_tag = 'command_voltage_legend'
        voltage_x_axis_tag = 'command_voltage_plot_x_axis'
        try:
            if dpg.does_item_exist(voltage_plot_tag):
                parent_item = dpg.get_item_parent(voltage_plot_tag)
                if parent_item:
                    dpg.delete_item(voltage_plot_tag)
                    with dpg.plot(label='Command Voltage Protocol', height=150, width=(- 1), tag=voltage_plot_tag, parent=parent_item):
                        dpg.add_plot_legend(outside=True, tag=voltage_legend_tag)
                        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='Time (ms)', tag=voltage_x_axis_tag)
                        dpg.set_axis_limits(x_axis, 0, 300)
                        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label='Voltage (mV)', tag=voltage_y_axis_tag)
                        dpg.set_axis_limits(y_axis, (- 140), 60)
                        self.voltage_y_axis = y_axis
                        self.voltage_series = []
                else:
                    print(f"Warning: Parent for plot '{voltage_plot_tag}' not found. Plot not cleared or recreated.")
                    self.voltage_y_axis = 0
                    self.voltage_series = []
            else:
                print(f"Info: Plot '{voltage_plot_tag}' did not exist. Not recreated by _clear_all_plots.")
                self.voltage_y_axis = 0
                self.voltage_series = []
        except Exception as e:
            print(f"Error processing voltage plot ('{voltage_plot_tag}') in _clear_all_plots: {e}")
            self.voltage_y_axis = 0
            self.voltage_series = []
            if dpg.does_item_exist(voltage_plot_tag):
                try:
                    dpg.delete_item(voltage_plot_tag)
                except Exception as del_e:
                    print(f"Error during cleanup of '{voltage_plot_tag}': {del_e}")
        current_plot_tag = 'current_plot'
        current_y_axis_tag = 'current_plot_y_axis'
        current_legend_tag = 'current_plot_legend'
        current_x_axis_tag = 'current_plot_x_axis'
        try:
            if dpg.does_item_exist(current_plot_tag):
                parent_item = dpg.get_item_parent(current_plot_tag)
                if parent_item:
                    dpg.delete_item(current_plot_tag)
                    with dpg.plot(label='Current Responses', height=350, width=(- 1), tag=current_plot_tag, parent=parent_item):
                        dpg.add_plot_legend(outside=True, tag=current_legend_tag)
                        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='Time (ms)', tag=current_x_axis_tag)
                        dpg.set_axis_limits(x_axis, 0, 300)
                        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label='Current (pA)', tag=current_y_axis_tag)
                        dpg.set_axis_limits(y_axis, (- 500), 50)
                        self.current_y_axis = y_axis
                        self.current_series = []
                else:
                    print(f"Warning: Parent for plot '{current_plot_tag}' not found. Plot not cleared or recreated.")
                    self.current_y_axis = 0
                    self.current_series = []
            else:
                print(f"Info: Plot '{current_plot_tag}' did not exist. Not recreated by _clear_all_plots.")
                self.current_y_axis = 0
                self.current_series = []
        except Exception as e:
            print(f"Error processing current plot ('{current_plot_tag}') in _clear_all_plots: {e}")
            self.current_y_axis = 0
            self.current_series = []
            if dpg.does_item_exist(current_plot_tag):
                try:
                    dpg.delete_item(current_plot_tag)
                except Exception as del_e:
                    print(f"Error during cleanup of '{current_plot_tag}': {del_e}")
    def update_current_plot(self):
        self.current_series = []
        plot_data = {'time_points': [], 'currents': [], 'voltages': [], 'model_type': (str(self.current_model.__class__.__name__) if self.current_model else 'Unknown')}
        current_y_axis_tag = 'current_plot_y_axis'
        if (not dpg.does_item_exist(current_y_axis_tag)):
            print(f"Error: Y-axis with tag '{current_y_axis_tag}' not found. Cannot update current plot.")
            if dpg.does_item_exist('current_plot'):
                children_dict = dpg.get_item_children('current_plot')
                if children_dict:
                    for slot_items in children_dict.values():
                        for item_in_slot in slot_items:
                            if (dpg.get_item_type(item_in_slot) == 'mvAppItemType::mvLineSeries'):
                                dpg.delete_item(item_in_slot)
            return
        children_dict = dpg.get_item_children(current_y_axis_tag)
        if children_dict:
            for slot_items in children_dict.values():
                for item in slot_items:
                    if (dpg.does_item_exist(item) and (dpg.get_item_type(item) == 'mvAppItemType::mvLineSeries')):
                        dpg.delete_item(item)
        if (not self.sim_results):
            if dpg.does_item_exist(current_y_axis_tag):
                dpg.set_axis_limits(current_y_axis_tag, (- 1.0), 1.0)
            return
        sorted_results = sorted(self.sim_results, key=(lambda x: x.get('step_volt', 0)), reverse=True)
        min_current_val = 0.0
        max_current_val = 0.0
        self.temp_scaled_data = []
        for (res_idx, res) in enumerate(sorted_results):
            try:
                time_data = res.get('time')
                current_data = res.get('sim_swp')
                volt = res.get('step_volt', 'N/A')
                if ((time_data is None) or (current_data is None) or (len(current_data) == 0)):
                    continue
                if (not isinstance(time_data, np.ndarray)):
                    time_data = np.array(time_data)
                if (not isinstance(current_data, np.ndarray)):
                    current_data = np.array(current_data)
                if (current_data.ndim > 1):
                    current_data = current_data.flatten()
                if (len(time_data) != len(current_data)):
                    time_data = np.linspace(0, (len(current_data) * 0.005), len(current_data))
                if (len(current_data) < 3):
                    self.temp_scaled_data.append({'voltage': volt, 'time': time_data, 'current': current_data, 'original_min': (np.min(current_data) if (len(current_data) > 0) else 0)})
                    continue
                window_size = 101
                if (window_size > len(current_data)):
                    window_size = max(3, (((len(current_data) // 2) * 2) - 1))
                if ((window_size >= 3) and (len(current_data) >= window_size)):
                    pad_width = (window_size // 2)
                    padded_current = np.pad(current_data, pad_width, mode='edge')
                    smoothed_current_list = [np.mean(padded_current[i:(i + window_size)]) for i in range(len(current_data))]
                    current_data = np.array(smoothed_current_list)
                peak_idx = np.argmin(current_data)
                current_peak_time = time_data[peak_idx]
                target_peak_time = 98.0
                time_shift = (target_peak_time - current_peak_time)
                time_data = (time_data + time_shift)
                if (time_data[0] > 0):
                    prepend_time = np.array([0.0])
                    prepend_current = np.array([current_data[0]])
                    time_data = np.concatenate((prepend_time, time_data))
                    current_data = np.concatenate((prepend_current, current_data))
                if (time_data[(- 1)] < 300.0):
                    last_time = time_data[(- 1)]
                    num_extra_points = max(2, int(((len(time_data) * (300.0 - last_time)) / (last_time if (last_time > 0) else 1.0))))
                    extra_times = np.linspace(last_time, 300.0, (num_extra_points + 1))[1:]
                    extra_currents = np.full_like(extra_times, current_data[(- 1)])
                    time_data = np.concatenate((time_data, extra_times))
                    current_data = np.concatenate((current_data, extra_currents))
                idx_97ms = np.argmin(np.abs((time_data - 97.0)))
                value_at_97ms = current_data[idx_97ms]
                current_data[:idx_97ms] = value_at_97ms
                idx_105ms = np.argmin(np.abs((time_data - 105.0)))
                flatline_value = current_data[idx_105ms]
                current_data[idx_105ms:] = flatline_value
                current_limit_val = 0.25
                current_data[(current_data > current_limit_val)] = current_limit_val
                self.temp_scaled_data.append({'voltage': volt, 'time': time_data, 'current': current_data, 'original_min': (np.min(current_data) if (len(current_data) > 0) else 0.0)})
            except IndexError as ie:
                print(f"IndexError processing sweep for current plot at {res.get('step_volt', 'N/A')}mV: {str(ie)}. Data length: T={len(res.get('time', []))}, C={len(res.get('sim_swp', []))}")
            except Exception as e:
                print(f"Error processing sweep for current plot at {res.get('step_volt', 'N/A')}mV: {str(e)}")
                import traceback
                traceback.print_exc()
        if self.temp_scaled_data:
            if ((self.temp_scaled_data[0]['time'] is not None) and (len(self.temp_scaled_data[0]['time']) > 0)):
                plot_data['time_points'] = self.temp_scaled_data[0]['time'].tolist()
            for (data_item_idx, data_item) in enumerate(self.temp_scaled_data):
                current_to_plot = data_item['current']
                time_to_plot = data_item['time']
                if ((len(current_to_plot) == 0) or (len(time_to_plot) == 0)):
                    continue
                min_current_val = min(min_current_val, np.min(current_to_plot))
                max_current_val = max(max_current_val, np.max(current_to_plot))
                time_list = time_to_plot.tolist()
                current_list = current_to_plot.tolist()
                series = dpg.add_line_series(time_list, current_list, label=f"{int(data_item['voltage'])}mV", parent=current_y_axis_tag)
                self.current_series.append(series)
                plot_data['currents'].append(current_list)
                plot_data['voltages'].append(int(data_item['voltage']))
        if dpg.does_item_exist(current_y_axis_tag):
            (y_min_limit, y_max_limit) = ((- 1.0), 1.0)
            if self.temp_scaled_data:
                if (min_current_val < 0):
                    y_min_limit = (min_current_val * 1.1)
                    y_max_limit = max((max_current_val * 1.1), ((- min_current_val) * 0.1), 0.1)
                else:
                    y_min_limit = 0.0
                    y_max_limit = max((max_current_val * 1.1), 0.1)
                if (abs((y_max_limit - y_min_limit)) < 1e-09):
                    y_max_limit = (y_min_limit + 0.1)
            dpg.set_axis_limits(current_y_axis_tag, y_min_limit, y_max_limit)
        if plot_data['currents']:
            if (hasattr(self, 'current_model') and (self.current_model is not None)):
                model_name = self.current_model.__class__.__name__
                plot_data['model_type'] = model_name
                if ('CTBN' in model_name):
                    model_suffix = 'ctbnmodel'
                elif ('Markov' in model_name):
                    model_suffix = 'markovmodel'
                elif ('HH' in model_name):
                    model_suffix = 'hhmodel'
                else:
                    model_suffix = 'model'
                plot_data['model_suffix'] = model_suffix
            if ((not plot_data['time_points']) and self.temp_scaled_data and (self.temp_scaled_data[0]['time'] is not None)):
                plot_data['time_points'] = self.temp_scaled_data[0]['time'].tolist()
            if plot_data['time_points']:
                self.save_plot_to_file('Current', plot_data)
    def run_simulation(self):
        self.sim_results = []
        self._clear_all_plots()
        for param_name in self.parameter_names:
            widget_tag = f'param_input_{param_name}'
            if dpg.does_item_exist(widget_tag):
                current_gui_value = dpg.get_value(widget_tag)
                setattr(self.current_model, param_name, current_gui_value)
        parameters = {}
        for param in self.parameter_names:
            value = getattr(self.current_model, param)
            parameters[param] = value
        parameters['is_hh_model'] = isinstance(self.current_model, HHModel)
        parameters['use_ctbn'] = isinstance(self.current_model, (CTBNMarkovModel, AnticonvulsantCTBNMarkovModel))
        parameters['is_anticonvulsant_model'] = isinstance(self.current_model, (AnticonvulsantMarkovModel, AnticonvulsantCTBNMarkovModel))
        if parameters['is_anticonvulsant_model']:
            if hasattr(self.current_model, 'drug_type'):
                parameters['drug_type'] = self.current_model.drug_type
        swp_seq = []
        if (hasattr(self.current_model, 'SwpSeq') and hasattr(self.current_model, 'NumSwps')):
            swp_array = self.current_model.SwpSeq
            num_swps = self.current_model.NumSwps
            for sweep_no in range(num_swps):
                if (sweep_no < swp_array.shape[1]):
                    holding_potential = swp_array[(2, sweep_no)]
                    holding_end_samples = swp_array[(3, sweep_no)]
                    target_voltage = swp_array[(4, sweep_no)]
                    test_end_samples = swp_array[(5, sweep_no)]
                    tail_potential = swp_array[(6, sweep_no)]
                    tail_end_samples = swp_array[(7, sweep_no)]
                    sampling_interval_ms = 0.005
                    holding_duration_ms = (holding_end_samples * sampling_interval_ms)
                    test_duration_ms = ((test_end_samples - holding_end_samples) * sampling_interval_ms)
                    tail_duration_ms = ((tail_end_samples - test_end_samples) * sampling_interval_ms)
                    sweep_dict = {'holding': holding_potential, 'conditioning': holding_potential, 'test': target_voltage, 'tail': tail_potential, 'holding_duration': holding_duration_ms, 'conditioning_duration': 0, 'test_duration': test_duration_ms, 'tail_duration': tail_duration_ms, 'holding_clamp': 0, 'conditioning_clamp': 0, 'test_clamp': 0, 'tail_clamp': 0}
                    swp_seq.append(sweep_dict)
        elif (hasattr(self.current_model, 'SwpSeq') and isinstance(self.current_model.SwpSeq, list)):
            for sweep_dict in self.current_model.SwpSeq:
                swp_seq.append(sweep_dict.copy())
        if (not swp_seq):
            self.show_message_dialog('Error', 'No voltage protocol defined for the current model.')
            return
        simulation_thread = threading.Thread(target=self.run_simulation_thread, args=(parameters, swp_seq))
        simulation_thread.start()
    def run_simulation_thread(self, parameters, swp_seq):
        try:
            num_swps = len(swp_seq)
            sweep_args = [(i, parameters, [swp_seq[i]]) for i in range(num_swps)]
            with Pool() as pool:
                results = pool.map(run_single_sweep, sweep_args)
            successful_results = [res for res in results if (res and ('sim_swp' in res) and (len(res['sim_swp']) > 0))]
            if successful_results:
                self.sim_results = sorted(successful_results, key=(lambda x: x['sweep_no']))
                dpg.split_frame()
                self.update_plots()
            else:
                self.show_message_dialog('Error', 'Simulation failed for all sweeps. Check console for details.')
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.show_message_dialog('Error', f'An unexpected error occurred during simulation: {e}')
        finally:
            gc.collect()
    def show_message_dialog(self, title, message):
        with dpg.window(label=title, modal=True, no_close=False, width=400) as modal_id:
            dpg.add_text(message)
            dpg.add_button(label='OK', width=75, callback=(lambda : dpg.delete_item(modal_id)))
    def on_model_change(self, sender, app_data):
        if (app_data == 'CTBN Markov'):
            self.current_model = self.ctbn_markov_model
            self.current_model_name = app_data
            self.parameter_names = self.markov_parameters
            self.parameter_info = self.markov_parameter_info
        elif (app_data == 'Legacy Markov'):
            self.current_model = self.legacy_markov_model
            self.current_model_name = app_data
            self.parameter_names = self.markov_parameters
            self.parameter_info = self.markov_parameter_info
        elif (app_data == 'Hodgkin-Huxley'):
            self.current_model = self.legacy_hh_model
            self.current_model_name = app_data
            self.parameter_names = self.hh_parameters
            self.parameter_info = self.hh_parameter_info
        elif (app_data == 'Anticonvulsant Legacy Markov'):
            self.current_model = self.anticonvulsant_markov_model
            self.current_model_name = app_data
            self.parameter_names = self.anticonvulsant_markov_parameters
            self.parameter_info = self.anticonvulsant_markov_parameter_info
        elif (app_data == 'Anticonvulsant CTBN Markov'):
            self.current_model = self.anticonvulsant_ctbn_markov_model
            self.current_model_name = app_data
            self.parameter_names = self.anticonvulsant_markov_parameters
            self.parameter_info = self.anticonvulsant_markov_parameter_info
        self.sim_results = []
        self.last_plot_data = {}
        self.setup_parameters()
        self.setup_protocol_widgets()
        self.update_plots()
    def start(self):
        dpg.show_viewport()
        dpg.set_primary_window('primary_window', True)
        dpg.start_dearpygui()
        dpg.destroy_context()
if (__name__ == '__main__'):
    freeze_support()
    app = IonChannelGUI()
    app.start()