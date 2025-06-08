
import dearpygui.dearpygui as dpg
import numpy as np
import sys
import threading
import gc
from multiprocessing import freeze_support

from worker import run_single_sweep
from ctbn_markov import CTBNMarkovModel
from legacy_markov import MarkovModel
from legacy_hh import HHModel

class IonChannelGUI:
    """
    Manages the main application GUI for simulating and comparing ion channel models.

    This class initializes the DearPyGui context, sets up the different simulation
    models (CTBN Markov, Legacy Markov, Hodgkin-Huxley), defines their parameters,
    and builds the user interface for model selection, parameter adjustment,
    voltage protocol definition, simulation execution, and results visualization.

    Attributes:
        ctbn_stiff_markov_model (CTBNMarkovModel): Instance of the CTBN Markov model.
        legacy_markov_model (MarkovModel): Instance of the Legacy Markov model.
        legacy_hh_model (HHModel): Instance of the Hodgkin-Huxley model.
        current_model (object): The currently selected simulation model.
        markov_parameters (list): List of parameter names for Markov-based models.
        hh_parameters (list): List of parameter names for the Hodgkin-Huxley model.
        parameter_names (list): List of parameter names for the current_model.
        markov_parameter_info (dict): Descriptions and bounds for Markov parameters.
        hh_parameter_info (dict): Descriptions and bounds for HH parameters.
        parameter_info (dict): Descriptions and bounds for current_model parameters.
        sim_results (list): Stores results from the latest simulation sweeps.
        last_saved_plot (str): Path to the last saved plot image.
    """
    def __init__(self):
        """
        Initializes the IonChannelGUI application.

        Sets up the DearPyGui context, initializes the simulation models,
        defines model parameters and their default values/bounds,
        configures the main viewport, and calls the GUI setup method.
        It also attempts to maximize the window, with specific handling for macOS.
        """
        dpg.create_context()

        # Initialize models
        self.ctbn_markov_model = CTBNMarkovModel()  # Renamed from ctbn_stiff_markov_model
        self.legacy_markov_model = MarkovModel()
        self.legacy_hh_model = HHModel()
        self.current_model = self.ctbn_markov_model

        # Model parameters
        self.markov_parameters = [
            'alcoeff', 'alslp', 'btcoeff', 'btslp',
            'gmcoeff', 'gmslp', 'dlcoeff', 'dlslp',
            'epcoeff', 'epslp', 'ztcoeff', 'ztslp',
            'ConCoeff', 'CoffCoeff', 'OpOnCoeff', 'OpOffCoeff'
        ]

        self.hh_parameters = ['g_Na', 'E_Na', 'C_m']
        self.parameter_names = self.markov_parameters  # Default

        # Parameter descriptions and bounds
        self.markov_parameter_info = {
            # default 80 ±20%
            'alcoeff': {'desc': 'alpha coefficient', 'bounds': (64, 96)},
            # default 8 ±20%
            'alslp': {'desc': 'alpha voltage dependence', 'bounds': (6.4, 9.6)},
            # default 0.8 ±20%
            'btcoeff': {'desc': 'beta coefficient',
                       'bounds': (0.64, 0.96)},
            # default 15 ±20%
            'btslp': {'desc': 'beta voltage dependence', 'bounds': (12, 18)},
            # default 150 ±20%
            'gmcoeff': {'desc': 'gamma coefficient', 'bounds': (120, 180)},
            # default 8 ±20%
            'gmslp': {'desc': 'gamma voltage dependence', 'bounds': (6.4, 9.6)},
            # default 40 ±20%
            'dlcoeff': {'desc': 'delta coefficient',
                       'bounds': (32, 48)},
            # default 15 ±20%
            'dlslp': {'desc': 'delta voltage dependence', 'bounds': (12, 18)},
            # default 1.75 ±20%
            'epcoeff': {'desc': 'epsilon coefficient', 'bounds': (1.4, 2.1)},
            # default 20 ±20%
            'epslp': {'desc': 'epsilon voltage dependence', 'bounds': (16, 24)},
            # default 0.03 ±20%
            'ztcoeff': {'desc': 'zeta coefficient',
                       'bounds': (0.024, 0.036)},
            # default 20 ±20%
            'ztslp': {'desc': 'zeta voltage dependence', 'bounds': (16, 24)},
            # default 0.02 ±20%
            'ConCoeff': {'desc': 'Base konlo', 'bounds': (0.016, 0.024)},
            # default 0.2 ±20%
            'CoffCoeff': {'desc': 'Base kofflo', 'bounds': (0.16, 0.24)},
            # default 0.75 ±20%
            'OpOnCoeff': {'desc': 'Base konOp', 'bounds': (0.6, 0.9)},
            # default 0.005 ±20%
            'OpOffCoeff': {'desc': 'Base koffOp', 'bounds': (0.004, 0.006)}
        }

        self.hh_parameter_info = {
            # default 120 ±20%
            'g_Na': {'desc': 'Sodium conductance', 'bounds': (100, 150)},
            # default 50 ±20%
            'E_Na': {'desc': 'Sodium reversal potential', 'bounds': (40, 60)},
            # default 1 ±20%
            'C_m': {'desc': 'Membrane capacitance', 'bounds': (0.8, 1.2)}
        }

        self.parameter_info = self.markov_parameter_info  # Default

        self.sim_results = []
        self.spont_ap_results = None
        self.evoked_ap_results = None

        # Temporary storage for scaled data
        self.temp_scaled_data = []
        self.temp_sim_scaled_data = []

        # Create themes for different data types

        # Create viewport with specific settings
        dpg.create_viewport(
            title="Ion Channel Simulator",
            width=1400,
            height=800,
            resizable=True,
            decorated=True,
            vsync=True
        )

        dpg.setup_dearpygui()

        # Create primary window that fills viewport
        with dpg.window(label="Main Window", autosize=True, no_resize=False, no_title_bar=True, no_move=True, tag="primary_window"):
            self.setup_gui()  # Move GUI setup inside the primary window

        dpg.show_viewport()

        # Try platform-specific maximization for macOS
        if sys.platform == 'darwin':
            import objc
            import AppKit

            # Get the window
            window = AppKit.NSApp().mainWindow()
            if window:
                # Maximize the window
                window.zoom_(None)
        else:
            # For other platforms
            dpg.maximize_viewport()

    def save_plot_to_file(self, plot_type, plot_data):
        """
        Saves the specified plot data to a PNG file in a separate process.

        This method is designed to auto-save "Current" or "Current Traces" plots.
        Other plot types (e.g., comparison plots) are expected to be saved
        via a different mechanism and will be skipped by this method.

        To avoid potential GUI backend conflicts with Matplotlib, this method
        works by:
        1. Creating a temporary directory.
        2. Serializing the `plot_data` (along with `plot_type` and model
           information if available) into a JSON file within the temp directory.
        3. Generating a small Python script (`save_plot.py`) in the temp directory.
           This script:
           a. Uses Matplotlib with the 'Agg' non-interactive backend.
           b. Loads the plot data from the JSON file.
           c. Determines the save path within the project's 'data/currents'
              directory, creating it if necessary. The filename is based on
              `plot_type` and `model_type`.
           d. Generates the plot using Matplotlib based on `plot_type`:
              - For "Current Traces": Plots Markov vs. HH currents.
              - For "Current": Plots multiple current sweeps for a single model,
                along with the command voltage protocol if available.
           e. Saves the figure to the determined PNG file path.
        4. Executes this `save_plot.py` script as a subprocess, passing the
           path to the JSON data file and the project root directory as arguments.
        5. Waits for the subprocess to complete and then cleans up the temporary
           directory and its contents.
        6. Stores the path of the saved file in `self.last_saved_plot` and
           displays a success or error message.

        Args:
            plot_type (str): The type of plot to save (e.g., "Current",
                             "Current Traces").
            plot_data (dict): A dictionary containing the data required to
                              reconstruct and save the plot. This typically
                              includes 'time_points', 'currents', 'voltages',
                              and 'model_type'.
        """
        # Only auto-save current plots, not comparison plots
        if plot_type != "Current" and plot_type != "Current Traces":
            print(
    f"Skipping auto-save for {plot_type} plot - should be saved via save_comparison_plots")
            return

        # Create a separate process to handle the save operation to avoid GUI
        # conflicts
        try:
            # Import all dependencies here to avoid import errors
            import subprocess
            import tempfile
            import json
            import os
            import sys

            # Create a temporary directory to store data
            temp_dir = tempfile.mkdtemp()
            data_file = os.path.join(temp_dir, "plot_data.json")

            # For Current plot type, ensure we have model information
            if plot_type == "Current" and hasattr(
                self, 'current_model') and self.current_model is not None:
                # Add model type to plot data if not already present
                if 'model_type' not in plot_data:
                    plot_data['model_type'] = self.current_model.__class__.__name__

                # Make sure we're using the data already in plot_data rather than regenerating
                # Only try to get data from model if we don't already have it
                # in plot_data
                if ('time_points' not in plot_data or 'currents' not in plot_data) and hasattr(
                    self.current_model, 'SimTime') and hasattr(self.current_model, 'SimCur'):
                    try:
                        plot_data['time_points'] = self.current_model.SimTime.tolist() if hasattr(
                            self.current_model.SimTime, 'tolist') else list(self.current_model.SimTime)
                        plot_data['currents'] = [
    self.current_model.SimCur.tolist() if hasattr(
        self.current_model.SimCur, 'tolist') else list(
            self.current_model.SimCur)]

                        # Add voltage protocol if available
                        if hasattr(self.current_model, 'SimCom'):
                            plot_data['voltages'] = [
    self.current_model.SimCom.tolist() if hasattr(
        self.current_model.SimCom, 'tolist') else list(
            self.current_model.SimCom)]
                    except Exception as e:
                        print(
    f"Error accessing simulation data from model: {e}")
                        import traceback
                        traceback.print_exc()

            # Save the plot data to a temporary file
            with open(data_file, 'w') as f:
                json.dump({"plot_type": plot_type, "plot_data": plot_data}, f)

            # Create a Python script to handle the plotting in a separate
            # process
            script_file = os.path.join(temp_dir, "save_plot.py")

            with open(script_file, 'w') as f:
                f.write("""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
import numpy as np
import gc
import os
import sys
import traceback
from scipy.stats import pearsonr

# Load the plot data
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

plot_type = data["plot_type"]
plot_data = data["plot_data"]

# Get the project root directory from command line arguments (passed from
# the parent script)
project_root = sys.argv[2] if len(
    sys.argv) > 2 else os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data", "currents")
os.makedirs(data_dir, exist_ok=True)

# Get model type for filename, defaulting to "unknown" if not provided
model_type = plot_data.get('model_type', 'unknown').lower().replace(' ', '_')

# Create filename based on plot type and model type
filename = f"{plot_type.lower().replace(' ', '_')}_{model_type}.png"

# Full path to save the file
file_path = os.path.join(data_dir, filename)

# Print where the file will be saved for user information

# Create the appropriate plot based on the plot type
try:
    if plot_type == "Current Traces":
        # Extract data for current traces
        time_points = plot_data.get('time_points', [])
        markov_current = plot_data.get('markov_current', [])
        hh_current = plot_data.get('hh_current', [])
        voltage = plot_data.get('voltage', 0)

        # Validate data before plotting
        if len(time_points) == 0 or len(
            markov_current) == 0 or len(hh_current) == 0:
            sys.exit(1)

        plt.figure(figsize=(10, 6))
        plt.plot(
    time_points,
    markov_current,
    label="Markov Model",
     color="blue")
        plt.plot(
    time_points,
    hh_current,
    label="HH Model (Scaled)",
     color="red")
        plt.xlabel("Time (ms)")
        plt.ylabel("Current (pA)")
        plt.title(f"Ion Channel Current Traces at {voltage} mV")
        plt.grid(True)
        plt.legend()

    elif plot_type == "Current":
        # Extract current model data
        time_points = plot_data.get('time_points', [])
        currents = plot_data.get('currents', [])
        voltages = plot_data.get('voltages', [])
        model_type = plot_data.get('model_type', 'Unknown Model')

        # Validate data
        if len(time_points) == 0 or len(currents) == 0 or len(voltages) == 0:
            sys.exit(1)

        # Create a figure with two subplots (voltage protocol and current
        # responses)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(
            10, 8), gridspec_kw={'height_ratios': [1, 3]})

        # Use a colormap to distinguish different voltage traces
        import matplotlib.cm as cm
        colors = cm.viridis(np.linspace(0, 1, len(currents)))

        # 1. Top subplot: Command Voltage Protocol
        ax1.set_title('Command Voltage Protocol')

        # Plot voltage protocols for each trace
        for i, voltage in enumerate(voltages):
            # Create simplified voltage protocol: hold at -80mV, step to test
            # voltage, back to -80mV
            holding_voltage = -80  # Default holding potential
            holding_duration = 98  # ms
            step_duration = 102    # ms
            total_duration = 300   # ms

            # Create voltage protocol time points and values
            protocol_time = [
    0,
    holding_duration,
    holding_duration,
    holding_duration+
    step_duration,
    holding_duration+
    step_duration,
     total_duration]
            protocol_voltage = [
    holding_voltage,
    holding_voltage,
    voltage,
    voltage,
    holding_voltage,
     holding_voltage]

            # Plot with the same color as the corresponding current trace
            ax1.plot(
    protocol_time,
    protocol_voltage,
    color=colors[i],
     label=f"{voltage} mV")

        ax1.set_ylabel('Voltage (mV)')
        ax1.set_xlim(0, 300)
        ax1.set_ylim(-120, 60)
        ax1.grid(True)

        # 2. Bottom subplot: Current Responses
        ax2.set_title(f"{model_type} Current Responses")

        # Plot each current trace with corresponding voltage label
        for i, (current, voltage) in enumerate(zip(currents, voltages)):
            # Ensure the time and current arrays have the same length
            if len(time_points) != len(current):
                # Use the shorter length
                min_length = min(len(time_points), len(current))
                adjusted_time = time_points[:min_length]
                adjusted_current = current[:min_length]
                ax2.plot(
    adjusted_time,
    adjusted_current,
    color=colors[i],
     label=f"{voltage} mV")
            else:
                # Use the arrays directly if they match
                ax2.plot(
    time_points,
    current,
    color=colors[i],
     label=f"{voltage} mV")

        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Current (pA)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_xlim(0, 300)

        # Add legend with voltage values to the bottom of the figure
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=7,
                  title="Test Voltage", frameon=True)

        # Add extra space at bottom for the legend
        plt.subplots_adjust(bottom=0.15)

        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    else:
        sys.exit(1)

    # Save the figure with high resolution
    plt.savefig(file_path, dpi=300, bbox_inches='tight', format='png')

    # Clean up
    plt.close('all')
    gc.collect()

except Exception as e:
    traceback.print_exc()
    plt.close('all')
    gc.collect()
    sys.exit(1)
    """)

            # Ensure data/comparison directory exists
            root_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(root_dir)
            data_dir = os.path.join(parent_dir, "data", "currents")
            os.makedirs(data_dir, exist_ok=True)

            # Get the project root directory
            project_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))

            # Run the script with output capture for debugging - passing the
            # project root directory
            process = subprocess.Popen(
                [sys.executable, script_file, data_file, project_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True)

            # Get the output and wait for process to complete with a reasonable
            # timeout
            try:
                # Wait for the process to complete (up to 5 seconds should be
                # plenty for plotting)
                stdout, stderr = process.communicate(timeout=5.0)
                if stdout:
                    print(f"Plot script output: {stdout}")
                if stderr:
                    print(f"Plot script error: {stderr}")

                # Check if the plot file was actually created
                if hasattr(
                    self, 'current_model') and self.current_model is not None:
                    model_type = self.current_model.__class__.__name__.lower().replace(' ', '_')
                else:
                    model_type = 'unknown'

                filename = f"{plot_type.lower().replace(' ', '_')}_{model_type}.png"
                plot_path = os.path.join(data_dir, filename)
                if os.path.exists(plot_path):
                    # Store the path to the most recently saved plot
                    self.last_saved_plot = plot_path
                else:
                    print(f"Plot file was not created at: {plot_path}")
            except subprocess.TimeoutExpired:
                process.kill()  # Kill the process if it takes too long
                print(
    f"Plot script timed out after 5 seconds, check logs for errors")

        except Exception as e:
            print(f"Error setting up save operation: {e}")
            import traceback
            traceback.print_exc()

    def setup_gui(self):
        """
        Creates and arranges the main graphical user interface elements.

        This includes sections for model selection, parameter configuration,
        voltage protocol definition, control buttons (e.g., "Run Simulation"),
        and the plotting areas for command voltage and current responses.
        """
        # Model selection
        with dpg.collapsing_header(label="Model Selection", default_open=True):
            dpg.add_combo(
            ["CTBN Markov", "Legacy Markov", "Hodgkin-Huxley"],
            default_value="CTBN Markov",
            callback=self.on_model_change,
            width=250,
            tag="model_selector"
        )

        # Parameters section
        with dpg.collapsing_header(label="Model Parameters", default_open=True,
                                  tag="parameters_header"):
            self.setup_parameters()

        # Voltage protocol
        with dpg.collapsing_header(label="Voltage Protocol", default_open=False, tag="protocol_header"):
            self.setup_voltage_protocol()

        # Control buttons
        with dpg.group(horizontal=True):
            dpg.add_button(
    label="Run Simulation",
     callback=self.run_simulation)

        # Main plot window
        with dpg.window(label="Plots", width=1100, height=800, pos=[300, 0]):
            with dpg.tab_bar():
                # Current Traces
                with dpg.tab(label="Current Traces"):
                    # Command voltage protocol plot above current traces
                    with dpg.plot(label="Command Voltage Protocol", height=120, width=-1, tag="command_voltage_plot"):
                        dpg.add_plot_legend(
    outside=True, tag="command_voltage_legend")
                        x_axis = dpg.add_plot_axis(
                            dpg.mvXAxis, label="Time (ms)")
                        dpg.set_axis_limits(x_axis, 0, 300)
                        y_axis = dpg.add_plot_axis(
                            dpg.mvYAxis, label="Voltage (mV)")
                        dpg.set_axis_limits(y_axis, -120, 60)

                    # Current traces plot
                    with dpg.group(horizontal=True):
                        dpg.add_text("Current Responses", color=[0, 150, 255])

                    with dpg.plot(label="Current Responses", height=350, width=-1, tag="current_plot"):
                        dpg.add_plot_legend(
    outside=True, tag="current_plot_legend")
                        x_axis = dpg.add_plot_axis(
                            dpg.mvXAxis, label="Time (ms)")
                        dpg.set_axis_limits(x_axis, 0, 300)
                        y_axis = dpg.add_plot_axis(
                            dpg.mvYAxis, label="Current (pA)")
                        dpg.set_axis_limits(y_axis, -500, 50)
                        self.current_y_axis = y_axis
                        self.current_series = []

    def setup_parameters(self):
        """
        Dynamically sets up the GUI input fields for the current model's parameters.

        This method first clears any existing parameter input fields within the
        'param_group'. It then inspects the type of the `current_model`
        (CTBNMarkovModel, MarkovModel, or HHModel) and populates the
        'Model Parameters' section of the GUI with appropriate input fields
        for each parameter.

        For Markov-based models, it groups parameters into 'Gate Parameters'
        (e.g., 'alcoeff', 'alslp') and 'Transition Rate Parameters'
        (e.g., 'ConCoeff'). For the Hodgkin-Huxley model, it lists its
        specific parameters (e.g., 'g_Na').

        Each input field is configured with its description, default value from the
        model, bounds, and is linked to the `on_parameter_change` callback
        to update the model when the user changes a value.
        """
        if dpg.does_item_exist("param_group"):
            dpg.delete_item("param_group")

        with dpg.group(tag="param_group", parent="parameters_header"):
            if isinstance(self.current_model, CTBNMarkovModel) or isinstance(
                self.current_model, MarkovModel):
                # Voltage-dependent parameters
                dpg.add_text("Gate Parameters")
                for i, param_base in enumerate(
                    ['al', 'bt', 'gm', 'dl', 'ep', 'zt']):
                    param_coeff = f"{param_base}coeff"
                    param_slp = f"{param_base}slp"

                    # Add a text label for each gate
                    dpg.add_text(f"{param_base.upper()} Gate")

                    # Coefficient - on its own line
                    value = getattr(self.current_model, param_coeff, 0.0)
                    bounds = self.parameter_info.get(
    param_coeff, {
        'bounds': [
            0.0, 1000.0], 'desc': f"{param_base} coefficient"})
                    dpg.add_input_float(
                        label=bounds.get('desc', f"{param_base} coefficient"),
                        default_value=value,
                        callback=self.on_parameter_change,
                        min_value=bounds.get('bounds', [0.0, 1000.0])[0],
                        max_value=bounds.get('bounds', [0.0, 1000.0])[1],
                        tag=f"param_{param_coeff}",
                        width=150
                    )

                    # Slope - on its own line
                    value = getattr(self.current_model, param_slp, 0.0)
                    bounds = self.parameter_info.get(
    param_slp, {
        'bounds': [
            0.0, 100.0], 'desc': f"{param_base} voltage dependence"})
                    dpg.add_input_float(
                        label=bounds.get(
    'desc', f"{param_base} voltage dependence"),
                        default_value=value,
                        callback=self.on_parameter_change,
                        min_value=bounds.get('bounds', [0.0, 100.0])[0],
                        max_value=bounds.get('bounds', [0.0, 100.0])[1],
                        tag=f"param_{param_slp}",
                        width=150
                    )
                    # Add a small separator between gate groups
                    if i < 5:  # Don't add after the last gate
                        dpg.add_spacer(height=5)

                # Transition rate parameters
                dpg.add_separator()
                dpg.add_text("Transition Rate Parameters")

                # Get appropriate parameter list based on model type
                # ConCoeff, CoffCoeff, OpOnCoeff, OpOffCoeff
                param_list = self.parameter_names[12:]

                for param in param_list:
                    value = getattr(self.current_model, param, 0.0)
                    bounds = self.parameter_info.get(
                        param, {'bounds': [0.0, 5.0], 'desc': f"Base {param}"})
                    dpg.add_input_float(
                        label=bounds.get('desc', f"Base {param}"),
                        default_value=value,
                        callback=self.on_parameter_change,
                        min_value=bounds.get('bounds', [0.0, 5.0])[0],
                        max_value=bounds.get('bounds', [0.0, 5.0])[1],
                        tag=f"param_{param}",
                        width=150
                    )
            else:  # HH model parameters
                for param in self.parameter_names:
                    value = getattr(self.current_model, param, 0.0)
                    bounds = self.parameter_info.get(
                        param, {'bounds': [0.0, 1000.0], 'desc': param})
                    dpg.add_input_float(
                        label=bounds.get('desc', param),
                        default_value=value,
                        callback=self.on_parameter_change,
                        min_value=bounds.get('bounds', [0.0, 1000.0])[0],
                        max_value=bounds.get('bounds', [0.0, 1000.0])[1],
                        tag=f"param_{param}",
                        width=150
                    )

    def setup_voltage_protocol(self):
        """
        Sets up the GUI elements for defining the voltage clamp protocol.

        This method clears any existing protocol controls and then creates new
        input fields for:
        - Holding potential (mV)
        - Holding duration (ms)
        - Test duration (ms)
        - Tail duration (ms)

        It also initializes a section for defining a series of test voltage steps,
        starting with a default set of voltages. Users can add or remove voltage
        steps dynamically.

        Input fields are linked to the `on_protocol_change` callback. Buttons for
        adding/removing steps and applying the protocol are also created, linked
        to `add_voltage_step`, `remove_voltage_step`, and
        `apply_voltage_protocol` respectively.
        """
        if dpg.does_item_exist("protocol_group"):
            dpg.delete_item("protocol_group")

        with dpg.group(tag="protocol_group", parent="protocol_header"):
            # Protocol settings
            dpg.add_text("Voltage Protocol Settings")

            # Holding potential
            dpg.add_input_int(
                label="Holding Potential (mV)",
                default_value=-80,
                callback=self.on_protocol_change,
                min_value=-120,
                max_value=50,
                tag="protocol_holding_potential",
                width=150
            )

            # Holding duration
            dpg.add_input_int(
                label="Holding Duration (ms)",
                default_value=98,
                callback=self.on_protocol_change,
                min_value=1,
                max_value=500,
                tag="protocol_holding_duration",
                width=150
            )

            # Test duration
            dpg.add_input_int(
                label="Test Duration (ms)",
                default_value=200,
                callback=self.on_protocol_change,
                min_value=1,
                max_value=500,
                tag="protocol_test_duration",
                width=150
            )

            # Tail duration
            dpg.add_input_int(
                label="Tail Duration (ms)",
                default_value=2,
                callback=self.on_protocol_change,
                min_value=0,
                max_value=100,
                tag="protocol_tail_duration",
                width=150
            )

            dpg.add_separator()
            dpg.add_text("Test Voltages")

            # Container for voltage steps
            with dpg.group(tag="voltage_steps_group"):
                # Start with default voltages
                self.voltage_step_tags = []
                default_voltages = [30, 0, -20, -30, -40, -50, -60]

                for i, voltage in enumerate(default_voltages):
                    with dpg.group(horizontal=True):
                        tag = f"voltage_step_{i}"
                        dpg.add_input_int(
                        label=f"Step {i + 1}",
                        default_value=voltage,
                        width=150,
                        callback=self.on_protocol_change,
                        tag=tag
                    )
                    self.voltage_step_tags.append(tag)

            # Add/remove voltage step buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Add Voltage Step",
                callback=self.add_voltage_step
            )
                dpg.add_button(
                    label="Remove Last Step",
                    callback=self.remove_voltage_step
                )

            # Apply protocol button
            dpg.add_button(
                label="Apply Protocol",
                callback=self.apply_voltage_protocol
            )

    def on_parameter_change(self, sender, value):
        """
        Callback function invoked when a model parameter is changed in the GUI.

        It extracts the parameter name from the sender's tag (by removing the
        'param_' prefix) and updates the corresponding attribute in the
        `current_model` with the new `value`.

        Args:
            sender (str or int): The tag of the GUI element that triggered the callback.
                                 Expected to be in the format 'param_<parameter_name>'.
            value (float): The new value of the parameter from the input field.
        """
        param_name = sender.replace("param_", "")
        setattr(self.current_model, param_name, value)

    def add_voltage_step(self):
        """
        Adds a new voltage step input field to the GUI.

        This method determines the number of existing voltage steps, then
        creates a new integer input field (defaulting to 0 mV) for the next
        step. The new input field is added to the 'voltage_steps_group'
        and is linked to the `on_protocol_change` callback. The tag for the
        new step is stored in `self.voltage_step_tags`.
        """
        # Get the number of existing steps
        step_num = len(self.voltage_step_tags)

        # Create a new step with default value of 0 mV
        with dpg.group(horizontal=True, parent="voltage_steps_group"):
            tag = f"voltage_step_{step_num}"
            dpg.add_input_int(
                label=f"Step {step_num + 1}",
                default_value=0,
                width=150,
                callback=self.on_protocol_change,
                tag=tag
            )
            self.voltage_step_tags.append(tag)

    def remove_voltage_step(self):
        """
        Removes the last added voltage step input field from the GUI.

        If there is more than one voltage step present, this method removes
        the last step's input field (and its associated label group) from the
        'voltage_steps_group'. It also removes the step's tag from
        `self.voltage_step_tags`. If only one step remains, it prints a
        message and does not remove the step, ensuring at least one step is
        always present.
        """
        if len(self.voltage_step_tags) > 1:  # Keep at least one step
            # Get the tag of the last step
            tag = self.voltage_step_tags.pop()

            # Get the parent group of the input widget
            parent = dpg.get_item_parent(tag)

            # Delete the parent group (which contains the label and input)
            dpg.delete_item(parent)

        else:
            print("Cannot remove the last step")

    def on_protocol_change(self, sender, value):
        """
        Callback function invoked when a voltage protocol parameter changes.

        Currently, this method prints a debug message indicating which protocol
        parameter was changed and its new value. In a future implementation,
        this would typically update the internal representation of the voltage
        protocol.

        Args:
            sender (str or int): The tag of the GUI element that triggered the
                                 callback (e.g., 'protocol_holding_potential',
                                 'voltage_step_0').
            value (int or float): The new value of the protocol parameter.
        """
        print(f"Protocol parameter {sender} changed to {value}")

    def apply_voltage_protocol(self):
        """
        Applies the currently defined voltage protocol to the selected model.

        This method retrieves all protocol parameters (holding potential,
        holding duration, test duration, tail duration, and all test voltage
        steps) from their respective GUI input fields. It then calls the
        `create_default_protocol` method of the `current_model` instance,
        passing these parameters to configure the model's simulation protocol.

        Finally, it displays a confirmation message summarizing the applied
        protocol settings.
        """
        # Get protocol parameters
        holding_potential = dpg.get_value("protocol_holding_potential")
        holding_duration = dpg.get_value("protocol_holding_duration")
        test_duration = dpg.get_value("protocol_test_duration")
        tail_duration = dpg.get_value("protocol_tail_duration")

        # Get voltage steps
        target_voltages = []
        for tag in self.voltage_step_tags:
            voltage = dpg.get_value(tag)
            target_voltages.append(voltage)

        # Apply the protocol to the current model
        if isinstance(self.current_model, CTBNMarkovModel):
            self.current_model.create_default_protocol(
                target_voltages=target_voltages,
                holding_potential=holding_potential,
                holding_duration=holding_duration,
                test_duration=test_duration,
                tail_duration=tail_duration
            )
        else:  # HH model
            self.current_model.create_default_protocol(
                target_voltages=target_voltages,
                holding_potential=holding_potential,
                holding_duration=holding_duration,
                test_duration=test_duration,
                tail_duration=tail_duration
            )

        # Show confirmation message
        self.show_message(
            f"Applied protocol with {len(target_voltages)} voltage steps:\n"
            f"Holding: {holding_potential} mV for {holding_duration} ms\n"
            f"Test: {target_voltages} mV for {test_duration} ms\n"
            f"Tail: {holding_potential} mV for {tail_duration} ms",
            "Protocol Applied"
        )

    def update_plots(self):
        """
        Refreshes all plots in the GUI, primarily the 'Current Responses' plot.

        This method first calls `_clear_all_plots()` to ensure a clean slate
        by removing and recreating the plot area. It also explicitly resets
        `self.current_series` to an empty list. Finally, it calls
        `update_current_plot()` to draw the new simulation data.
        """
        # Clear all existing plots and legends
        self._clear_all_plots()

        # Reset any stored plot data
        if hasattr(self, 'current_series'):
            self.current_series = []

        # Update all plots
        self.update_current_plot()

    def _clear_all_plots(self):
        """
        Clears and recreates the main 'Current Responses' plot.

        This method attempts to completely delete the existing 'current_plot'
        and then recreates it with its legend, axes (Time (ms) and Current (pA)),
        and default axis limits. The `self.current_y_axis` and
        `self.current_series` attributes are reset.

        If an exception occurs during this process (e.g., if the plot or its
        parent cannot be found or manipulated as expected), it falls back to
        calling `_clear_plot_series()` to at least remove the data series
        from the existing plot.
        """
        try:
            # Delete and recreate the current plot
            if dpg.does_item_exist("current_plot"):
                # Get the parent of the current plot
                current_plot_parent = dpg.get_item_parent("current_plot")
                # Delete the plot
                dpg.delete_item("current_plot")
                # Recreate the plot
                with dpg.plot(label="Current Responses", height=350, width=-1, tag="current_plot", parent=current_plot_parent):
                    dpg.add_plot_legend(
    outside=True, tag="current_plot_legend")
                    x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Time (ms)")
                    dpg.set_axis_limits(x_axis, 0, 300)
                    y_axis = dpg.add_plot_axis(
    dpg.mvYAxis, label="Current (pA)")
                    dpg.set_axis_limits(y_axis, -500, 50)
                    self.current_y_axis = y_axis
                    self.current_series = []

        except Exception as e:
            print(f"Error recreating plots: {e}")
            # Fallback to just clearing the series
            self._clear_plot_series()

    def _clear_plot_series(self):
        """
        Clears all data series from the 'Current Responses' plot.

        This method iterates through the `self.current_series` attribute,
        which can be a list of series tags or a single series tag. It attempts
        to delete each DearPyGui item (plot series) associated with these tags.
        Any exceptions during deletion (e.g., if a series was already deleted
        or never existed) are caught and ignored. Finally, `self.current_series`
        is reset to an empty list.
        """
        # Clear current plot series
        if hasattr(self, 'current_series'):
            if isinstance(self.current_series, list):
                for series in self.current_series:
                    try:
                        dpg.delete_item(series)
                    except:
                        pass
            else:
                try:
                    dpg.delete_item(self.current_series)
                except:
                    pass
            self.current_series = []

    def update_current_plot(self):
        """
        Updates the 'Current Responses' plot with data from `self.sim_results`.

        This method performs several steps to display the simulation results:
        1. Resets `self.current_series`.
        2. Initializes `plot_data` for potential auto-saving.
        3. Sorts simulation results by step voltage for consistent plot coloring.
        4. Finds the Y-axis of the 'current_plot'. If not found, it exits.
        5. Clears any existing line series from the Y-axis and recreates the legend.
        6. Iterates through the sorted simulation results:
           a. Extracts time, current, and voltage for each sweep.
           b. Skips empty results.
           c. Ensures current data is 1D and time/current arrays have compatible lengths.
           d. Applies Savitzky-Golay smoothing to the current trace if `use_ctbn`
              is true (CTBN model) and sufficient data points exist.
           e. Aligns the peak of the current trace to a target time (98ms) by
              shifting the time array.
           f. Ensures the time array starts at 0ms and extends to at least 300ms,
              padding with the first/last current values if necessary.
           g. Updates `min_current` and `max_current` for Y-axis auto-scaling.
           h. Adds the processed time and current data as a new line series to
              the plot, labeled with the step voltage.
           i. Stores the time, current, and voltage data in `plot_data`.
        7. If data was plotted, it attempts to autoscale the Y-axis based on
           `min_current` and `max_current`, with some padding.
        8. Stores the `plot_data` in `self.last_plot_data` for saving.
        """
        # Make sure current_series is empty
        self.current_series = []

        # Check if we have all expected sweeps
        expected_voltages = [30, 0, -20, -30, -40, -50, -60]
        found_voltages = [res['step_volt'] for res in self.sim_results]

        # Initialize plot data for auto-save
        plot_data = {
            'time_points': [],
            'currents': [],
            'voltages': [],
            'model_type': str(self.current_model.__class__.__name__)
        }

        # Sort results by voltage for consistent coloring
        sorted_results = sorted(
    self.sim_results,
    key=lambda x: x['step_volt'],
     reverse=True)

        # Track min/max current values for axis scaling
        min_current = 0
        max_current = 0

        # Temporary storage for scaled data
        self.temp_scaled_data = []

        # Find the current plot y-axis
        current_y_axis = None
        if dpg.does_item_exist("current_plot"):
            plot_children = dpg.get_item_children("current_plot")[1]
            for child in plot_children:
                if dpg.get_item_type(
                    child) == "mvAppItemType::mvPlotAxis" and "Current" in dpg.get_item_label(child):
                    current_y_axis = child
                    break

        if not current_y_axis:
            print("Could not find current plot y-axis")
            return

        # Clear existing traces
        if dpg.does_item_exist(current_y_axis):
            axis_children = dpg.get_item_children(current_y_axis)[1] if dpg.get_item_children(
                current_y_axis) and len(dpg.get_item_children(current_y_axis)) > 1 else []
            # Delete all series
            for child in axis_children:
                if dpg.get_item_type(child) == "mvAppItemType::mvLineSeries":
                    dpg.delete_item(child)

        # Completely recreate the legend to ensure it's empty
        if dpg.does_item_exist("current_plot_legend"):
            dpg.delete_item("current_plot_legend")

        # Add a new legend
        if dpg.does_item_exist("current_plot"):
            dpg.add_plot_legend(
    outside=True,
    tag="current_plot_legend",
     parent="current_plot")

        for res in sorted_results:
            try:
                time = res['time']
                current = res['sim_swp']
                volt = res['step_volt']

                # Skip empty results
                if len(current) == 0:
                    print(f"Skipping empty result for {volt}mV")
                    continue

                # Ensure arrays are 1D and of compatible length
                if isinstance(current, np.ndarray) and current.ndim > 1:
                    current = current.flatten()

                if len(time) != len(current):
                    # Adjust time array to match current length if needed
                    time = np.arange(len(current)) * 0.005

                # Apply smoothing for both models to make traces have rounded peaks
                # Use a window size that preserves important features but
                # smooths out noise
                window_size = 101  # Must be odd

                # Ensure window size is not larger than the data
                if window_size > len(current):
                    window_size = min(
    len(current) // 2 * 2 - 1,
     11)  # Ensure it's odd

                if window_size >= 3:  # Only smooth if we have enough points
                    # Create a padded version of the current to handle edges
                    # properly
                    padded_current = np.pad(
    current, (window_size // 2, window_size // 2), mode='edge')
                    smoothed_current = np.zeros_like(current)
                    for i in range(len(current)):
                        smoothed_current[i] = np.mean(
                            padded_current[i:i + window_size])
                    current = smoothed_current

            # 1. First align the peak to 98ms
            # Find the peak current (minimum for inward currents)
                peak_idx = np.argmin(current)

            # Calculate the time shift needed to align peak to 98ms
                current_peak_time = time[peak_idx]
                target_peak_time = 98.0  # ms
                time_shift = target_peak_time - current_peak_time

                # Apply the time shift by adjusting the time array
                time = time + time_shift

            # Ensure time array starts at 0ms and extends to at least 300ms for
            # proper display
                if time[0] > 0:
                    # Extend time array to start at 0ms
                    prepend_time = np.array([0])
                    # Use first value for prepended point
                    prepend_current = np.array([current[0]])
                    time = np.concatenate((prepend_time, time))
                    current = np.concatenate((prepend_current, current))

                if time[-1] < 300:
                    # Always ensure the time array extends to exactly 300ms
                    # Find the last time point
                    last_time = time[-1]
                    if last_time != 300:
                        # Create a new time array that goes from the current points to 300ms
                        # with enough points to ensure a smooth continuation
                        num_extra_points = max(
                            10, int(len(time) * (300 - last_time) / last_time))
                        extra_times = np.linspace(last_time, 300, num_extra_points)[
                                                  1:]  # Skip first point to avoid duplication
                        extra_currents = np.full_like(
                            extra_times, current[-1])  # Use last current value

                        # Concatenate with existing arrays
                        time = np.concatenate((time, extra_times))
                        current = np.concatenate((current, extra_currents))

                # 2. Then apply pre-97ms flatlining
                # Find the index corresponding to 97ms
                idx_97ms = np.argmin(np.abs(time - 97.0))

                # Get the exact value at 97ms for this sweep
                value_at_97ms = current[idx_97ms]

                # Set all values before 97ms to this value
                current[:idx_97ms] = value_at_97ms

                # 3. Then apply post-105ms flatlining
                # Find the index corresponding to 105ms
                idx_105ms = np.argmin(np.abs(time - 105.0))

                # Get the value at exactly 105ms for this sweep
                flatline_value = current[idx_105ms]

                # Set all points after 105ms to this value
                current[idx_105ms:] = flatline_value

                # 4. Finally limit extreme current values above 0.25 pA
                current_limit = 0.25  # pA

                # Find values above the limit
                above_limit = current > current_limit
                if np.any(above_limit):
                    # Apply the limit
                    current[above_limit] = current_limit
                    print(f"Limited {np.sum(above_limit)} points above {current_limit} pA")

                # Store processed data for later scaling
                self.temp_scaled_data.append({
                    'voltage': volt,
                    'time': time,
                    'current': current,
                    'original_min': np.min(current)
                })
            except Exception as e:
                print(f"Error processing sweep at {res['step_volt']}mV: {str(e)}")
                import traceback
                traceback.print_exc()

        # After processing all simulation sweeps, apply uniform scaling
        if self.temp_scaled_data:
            # Find the deepest current across all sweeps
            deepest_current = 0
            for data in self.temp_scaled_data:
                if data['original_min'] < deepest_current:
                    deepest_current = data['original_min']

            # Calculate a single scaling factor for all sweeps
            if deepest_current < 0:  # Only scale if there's a negative peak
                # Apply uniform scaling to all models
                scale_factor = 1.0

                # Apply the same scaling factor to all sweeps
                for data in self.temp_scaled_data:
                    # Track min/max for axis scaling
                    min_current = min(min_current, np.min(data['current']))
                    max_current = max(max_current, np.max(data['current']))

                    # Convert to Python lists for DearPyGUI
                    time_list = data['time'].tolist() if isinstance(
                        data['time'], np.ndarray) else list(data['time'])
                    current_list = data['current'].tolist() if isinstance(
                        data['current'], np.ndarray) else list(data['current'])

                    # Ensure the trace extends to 300ms by explicitly adding a
                    # final point
                    if time_list[-1] < 300:
                        time_list.append(300)
                        current_list.append(current_list[-1])

                    # Add simulation trace
                    series = dpg.add_line_series(
                        time_list, current_list,
                        label=f"{int(data['voltage'])}mV",
                        parent=current_y_axis
                    )
                    self.current_series.append(series)

                    # Collect data for auto-saving
                    # All sweeps use the same time points
                    plot_data['time_points'] = time_list
                    plot_data['currents'].append(current_list)
                    plot_data['voltages'].append(int(data['voltage']))
            else:
                # For Markov model, don't scale currents
                scale_factor = 1.0

                # Apply no scaling to sweeps
                for data in self.temp_scaled_data:
                    # Scale the current (no scaling, factor = 1.0)
                    # data['current'] = data['current'] * scale_factor

                    # Track min/max for axis scaling
                    min_current = min(min_current, np.min(data['current']))
                    max_current = max(max_current, np.max(data['current']))

                    # Convert to Python lists for DearPyGUI
                    time_list = data['time'].tolist() if isinstance(
                        data['time'], np.ndarray) else list(data['time'])
                    current_list = data['current'].tolist() if isinstance(
                        data['current'], np.ndarray) else list(data['current'])

                    # Ensure the trace extends to 300ms by explicitly adding a
                    # final point
                    if time_list[-1] < 300:
                        time_list.append(300)
                        current_list.append(current_list[-1])

                    # Add simulation trace
                    series = dpg.add_line_series(
                        time_list, current_list,
                        label=f"{int(data['voltage'])}mV",
                        parent=current_y_axis
                    )
                    self.current_series.append(series)
                    print(f"Plotted Markov simulation data for {data['voltage']}mV with {len(data['current'])} points")

        # Set y-axis limits based on data, ensuring max depth is visible
        if min_current < 0:
            # For negative currents (inward), ensure we show the full depth
            y_min = min_current * 1.1
            # Ensure some positive space
            y_max = max(max_current * 1.1, -min_current * 0.1)
            dpg.set_axis_limits(current_y_axis, y_min, y_max)
        else:
            # For positive currents, ensure we show from 0
            y_min = 0
            # Ensure y_max is at least 1.0 to avoid invisible traces when
            # max_current is very small
            y_max = max(max_current * 1.1, 1.0)
            dpg.set_axis_limits(current_y_axis, y_min, y_max)

        # Plot command voltage protocol
        if dpg.does_item_exist("command_voltage_plot"):
            # Find existing axes
            x_axis = None
            y_axis = None
            children = dpg.get_item_children("command_voltage_plot")[1] if dpg.get_item_children(
                "command_voltage_plot") and len(dpg.get_item_children("command_voltage_plot")) > 1 else []

            for child in children:
                if dpg.get_item_type(child) == "mvAppItemType::mvPlotAxis":
                    if "Time" in dpg.get_item_label(child):
                        x_axis = child
                    elif "Voltage" in dpg.get_item_label(child):
                        y_axis = child

        # Only delete line series, not axes
        for child in children:
            if dpg.get_item_type(child) == "mvAppItemType::mvLineSeries":
                dpg.delete_item(child)

        # If axes don't exist, create them
        if not x_axis or not y_axis:
            # Clear the entire plot and recreate
            dpg.delete_item("command_voltage_plot", children_only=True)
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Time (ms)")
            dpg.set_axis_limits(x_axis, 0, 300)
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Voltage (mV)")
            dpg.set_axis_limits(y_axis, -120, 60)

        # Auto-save the current plot if we have data
        if plot_data['currents'] and len(plot_data['currents']) > 0:

            # Add current model type to plot data
            if hasattr(self, 'current_model') and self.current_model is not None:
                model_name = self.current_model.__class__.__name__
                plot_data['model_type'] = model_name

                # Generate appropriate filename based on model type
                if 'CTBN' in model_name:
                    model_suffix = 'ctbnmodel'
                elif 'Markov' in model_name:
                    model_suffix = 'markovmodel'
                elif 'HH' in model_name:
                    model_suffix = 'hhmodel'
                else:
                    model_suffix = 'model'

                plot_data['model_suffix'] = model_suffix

            self.save_plot_to_file("Current", plot_data)

        # Determine voltage protocols for loaded and simulated data
        holding_duration = 98
        test_duration = 202
        total_duration = holding_duration + test_duration

        # Simulated data
        for idx, res in enumerate(getattr(self, 'sim_results', [])):
            hold = getattr(self, 'loaded_holding_potential', -80)
            step = res.get('step_volt', 0)
            # Protocol: hold, step, tail
            time = [
    0,
    holding_duration,
    holding_duration,
    holding_duration +
    test_duration,
    holding_duration +
    test_duration,
     total_duration]
            volt = [hold, hold, step, step, hold, hold]
            series = dpg.add_line_series(
                time, volt, parent=y_axis, label=f"Sim {step}")

    def run_simulation(self):
        """
        Executes the simulation based on the current model, parameters, and protocol.

        This method performs the following steps:
        1. Clears previous simulation results and plots.
        2. Collects the current model parameters.
        3. Retrieves the voltage protocol (SwpSeq) from the current model.
           - It handles different protocol formats:
             - For models with `SwpSeq` as a NumPy array (like CTBNMarkovModel and
               potentially MarkovModel if adapted), it converts the array data
               (voltages and epoch end times in samples) into a list of
               dictionaries, each representing a sweep with durations in ms.
               A sampling interval of 0.005 ms/sample is assumed for conversion.
             - For models where `SwpSeq` is already a list of dictionaries (legacy
               format), it creates a clean copy of each sweep dictionary.
        4. Adds flags to the parameters to identify the model type (HH, CTBN)
           for the worker process.
        5. Initiates a background task (`run_simulation_thread` or a direct call
           to `run_single_sweep` for each sweep if threading is disabled/simplified)
           to perform the simulation sweeps.
        6. The background task calls `run_single_sweep` for each sweep, passing
           the sweep number, model parameters, and the processed `swp_seq`.
        7. Results from each sweep are collected. If a sweep is successful and
           contains 'sim_swp' data, its results are stored.
        8. After all sweeps complete (or if an error occurs), it calls
           `update_plots` to display the new data and `save_plot_to_file`
           to automatically save the generated plot.
        9. Displays a success or error message to the user.
        """


        # Clear previous results and plots before starting new simulation
        self.sim_results = []
        self._clear_all_plots()

        # Create parameter dictionary
        parameters = {}

        # Add model-specific parameters
        for param in self.parameter_names:
            value = getattr(self.current_model, param)
            parameters[param] = value

        # Get protocol from current model - handle different formats for each model
        swp_seq = []
        if hasattr(self.current_model, 'SwpSeq') and hasattr(self.current_model, 'NumSwps'):
            # For both models - convert NumPy array protocol to list of dicts for worker
            swp_array = self.current_model.SwpSeq
            num_swps = self.current_model.NumSwps

            for sweep_no in range(num_swps):
                # Convert protocol format to dictionary format
                if sweep_no < swp_array.shape[1]:  # Make sure sweep_no is valid
                    # Extract protocol parameters from swp_array
                    # SwpSeq stores durations as end times in samples (0.005 ms per sample)
                    # SwpSeq[0, sweep_no] = number of epochs (typically 3)
                    # SwpSeq[1, sweep_no] = (not used here, seems to be start sample of epoch 1, often 0)
                    # SwpSeq[2, sweep_no] = voltage of epoch 1 (holding_potential)
                    # SwpSeq[3, sweep_no] = end sample of epoch 1 (holding_duration_end_samples)
                    # SwpSeq[4, sweep_no] = voltage of epoch 2 (test_potential/target_voltage)
                    # SwpSeq[5, sweep_no] = end sample of epoch 2 (test_duration_end_samples)
                    # SwpSeq[6, sweep_no] = voltage of epoch 3 (tail_potential)
                    # SwpSeq[7, sweep_no] = end sample of epoch 3 (tail_duration_end_samples)

                    holding_potential = swp_array[2, sweep_no]
                    holding_end_samples = swp_array[3, sweep_no]
                    target_voltage = swp_array[4, sweep_no]
                    test_end_samples = swp_array[5, sweep_no]
                    tail_potential = swp_array[6, sweep_no]
                    tail_end_samples = swp_array[7, sweep_no]

                    # Calculate durations in ms (assuming 0.005 ms per sample)
                    sampling_interval_ms = 0.005
                    holding_duration_ms = holding_end_samples * sampling_interval_ms
                    test_duration_ms = (test_end_samples - holding_end_samples) * sampling_interval_ms
                    tail_duration_ms = (tail_end_samples - test_end_samples) * sampling_interval_ms
                    
                    sweep_dict = {
                        'holding': holding_potential,
                        'conditioning': holding_potential,  # Assuming conditioning is same as holding for this protocol type
                        'test': target_voltage,
                        'tail': tail_potential,
                        'holding_duration': holding_duration_ms,
                        'conditioning_duration': 0,  # No explicit conditioning in this protocol structure
                        'test_duration': test_duration_ms,
                        'tail_duration': tail_duration_ms,
                        'holding_clamp': 0, # Clamps are not typically defined in SwpSeq, defaults to 0 (voltage clamp)
                        'conditioning_clamp': 0,
                        'test_clamp': 0,
                        'tail_clamp': 0
                    }
                    swp_seq.append(sweep_dict)
        else:
            # For legacy format - make clean copy of each sweep dictionary
            for sweep in self.current_model.SwpSeq:
                if isinstance(sweep, dict):
                    sweep_copy = {
                        'holding': sweep.get('holding', 0),
                        'conditioning': sweep.get('conditioning', 0),
                        'test': sweep.get('test', 0),
                        'tail': sweep.get('tail', 0),
                        'holding_duration': sweep.get('holding_duration', 0),
                        'conditioning_duration': sweep.get('conditioning_duration', 0),
                        'test_duration': sweep.get('test_duration', 0),
                        'tail_duration': sweep.get('tail_duration', 0),
                        'holding_clamp': sweep.get('holding_clamp', 0),
                        'conditioning_clamp': sweep.get('conditioning_clamp', 0),
                        'test_clamp': sweep.get('test_clamp', 0),
                        'tail_clamp': sweep.get('tail_clamp', 0)
                    }
                    swp_seq.append(sweep_copy)

        num_swps = len(swp_seq)

        # Add model type flags to parameters to identify model type in worker
        parameters['is_hh_model'] = isinstance(self.current_model, HHModel)
        parameters['use_ctbn'] = isinstance(self.current_model, CTBNMarkovModel)

        # Clean up memory
        gc.collect()

        def background_task():
            try:
                results = []
                for i in range(num_swps):
                    # Run each sweep directly in the current process
                    result = run_single_sweep((i, parameters, swp_seq))
                    if result and 'sim_swp' in result and len(result['sim_swp']) > 0:
                        results.append(result)
                    else:
                        import traceback
                        traceback.print_exc()

                if len(results) > 0:
                    # Sort results by sweep number
                    self.sim_results = sorted(results, key=lambda x: x['sweep_no'])

                    # Update plots in the main thread
                    dpg.split_frame()
                    self.update_plots()
                else:
                    print("No simulation results to plot")

            except Exception as e:
                print(f"Simulation error: {str(e)}")
                traceback.print_exc()

        # Start background task
        thread = threading.Thread(target=background_task, daemon=True)
        thread.start()




    def show_message(self, message, title="Message", is_error=False):
        """
        Displays a modal message dialog to the user.

        Args:
            message (str): The text message to display in the dialog.
            title (str, optional): The title of the modal dialog window.
                                   Defaults to "Message".
            is_error (bool, optional): If True, indicates the message is an error.
                                       (Currently not used to change dialog appearance,
                                       but could be used for future styling).
                                       Defaults to False.
        """
        with dpg.window(label=title, modal=True, no_close=False, width=400) as modal_id:
            dpg.add_text(message)
            dpg.add_button(label="OK", width=75, callback=lambda: dpg.delete_item(modal_id))

    def on_model_change(self, sender, value):
        """
        Callback function invoked when the selected simulation model changes.

        Based on the `value` from the model selection dropdown:
        - Sets `self.current_model` to the appropriate model instance
          (CTBNMarkovModel, MarkovModel, or HHModel).
        - Updates `self.parameter_names` and `self.parameter_info` to reflect
          the parameters of the newly selected model.
        - Calls `setup_parameters()` to refresh the parameter input fields in the GUI.
        - Calls `setup_voltage_protocol()` to refresh the voltage protocol section.
        - Calls `update_plots()` to clear and prepare plots for the new model.

        Args:
            sender (str or int): The tag of the GUI combo box that triggered the callback.
            value (str): The string value of the selected model
                         (e.g., "CTBN Markov", "Legacy Markov", "Hodgkin-Huxley").
        """
        if value == "CTBN Markov":
            self.current_model = self.ctbn_markov_model
            self.parameter_names = self.markov_parameters
            self.parameter_info = self.markov_parameter_info
        elif value == "Legacy Markov":
            self.current_model = self.legacy_markov_model
            self.parameter_names = self.markov_parameters
            self.parameter_info = self.markov_parameter_info
        else:  # Hodgkin-Huxley
            self.current_model = self.legacy_hh_model
            self.parameter_names = self.hh_parameters
            self.parameter_info = self.hh_parameter_info

        # Update parameter display
        self.setup_parameters()

        # Update voltage protocol
        self.setup_voltage_protocol()

        # Update all plots with new model
        self.update_plots()
    
    def start(self):
        """
        Initializes and starts the DearPyGui application.

        This method performs the final steps to launch the GUI:
        - Makes the viewport visible.
        - Sets the primary window.
        - Starts the DearPyGui event loop (blocking).
        - Destroys the DearPyGui context when the loop exits.
        """
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        dpg.start_dearpygui()
        dpg.destroy_context()

if __name__ == '__main__':
    # Main entry point for the application.
    # Initializes and starts the IonChannelGUI.
    freeze_support()
    app = IonChannelGUI()
    app.start()