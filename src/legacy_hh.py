"""
Implements a Hodgkin-Huxley (HH) type ion channel model.

This module defines the `HHModel` class, which represents a classical
Hodgkin-Huxley model, primarily configured for simulating sodium channel
dynamics. 
"""
import numpy as np
from scipy.integrate import solve_ivp
class HHModel():
    """
    Implements a Hodgkin-Huxley (HH) model for simulating ion channel dynamics.

    This class provides a framework for a canonical Hodgkin-Huxley style model,
    specifically configured to simulate sodium (Na+) channel behavior. It uses
    voltage-dependent rate constants (alpha and beta) to describe the
    transitions of activation (m) and inactivation (h) gates.

    Key features:
    -   Models channel gating using three independent activation gates (m1, m2, m3)
        and one inactivation gate (h).
    -   Calculates macroscopic ionic current using the Goldman-Hodgkin-Katz (GHK)
        current equation, considering Na+ permeability and concentrations.
    -   Simulates the channel's response to voltage clamp protocols by numerically
        integrating the system of ordinary differential equations (ODEs) that
        describe the time evolution of the gating variables.
    -   Includes methods to pre-calculate rate constants and GHK factors across
        a voltage range for computational efficiency.
    -   Provides utilities to generate standard experimental voltage protocols,
        such as those for determining activation curves, steady-state
        inactivation, and recovery from inactivation.

    The model is initialized with default biophysical parameters typical for
    neuronal sodium channels but can be customized by modifying its attributes.
    """
    def __init__(self):
        """Initializes the Hodgkin-Huxley model for a sodium channel.

        Sets up default parameters for a canonical Hodgkin-Huxley style model,
        specifically tailored for a sodium channel. This includes:
        - Biophysical constants: Max sodium conductance (`g_Na`), sodium reversal
          potential (`E_Na`), membrane capacitance (`C_m`), resting potential (`V_rest`).
        - Simulation parameters: Sampling interval (`sampint`), number of channels (`numchan`).
        - Ionic concentrations: Intracellular (`Na_in`) and extracellular (`Na_out`)
          sodium concentrations, also stored as `Nai` and `Nao`.
        - Gating variables: Initial values for m1, m2, m3 (activation) and h (inactivation)
          gates, all set to 0.0 initially.
        - Voltage range for rate calculations: `v_range` from -100 mV to 100 mV.
        - Placeholders for simulation results: `SimSwp` (currents), `SimCom` (voltages),
          `SimOp` (open probability), `SwpSeq` (sweep protocols), `time`.
        - Current membrane potential: `vm` initialized to -80 mV.
        - Physical constants: Faraday's constant (`F`), gas constant (`Rgc`),
          temperature in Kelvin (`Tkel`).
        - Channel permeability: `PNasc` for sodium.

        The initialization process also involves:
        1. Calling `self.initialize_rate_constants()` to pre-calculate voltage-dependent
           rate constants (alpha and beta values for m and h gates) across `v_range`.
        2. Preparing a reusable array `_reusable_y0` for ODE solver initial conditions.
        3. Calling `self.create_default_protocol()` to set up a default voltage
           clamp protocol.
        """
        self.g_Na = 0.12
        self.E_Na = 50.0
        self.C_m = 1.0
        self.V_rest = (- 65.0)
        self.sampint = 0.005
        self.numchan = 1
        self.Na_in = 15.0
        self.Na_out = 155.0
        self.m1 = 0.0
        self.m2 = 0.0
        self.m3 = 0.0
        self.h = 0.0
        self.v_range = np.linspace((- 100), 100, 1000)
        self.SimSwp = []
        self.SimCom = []
        self.SimOp = []
        self.SwpSeq = []
        self.NumSwps = 0
        self.vm = (- 80)
        self.time = []
        self.F = 96480
        self.Rgc = 8314
        self.Tkel = (273.15 + 22.0)
        self.Nao = 155
        self.Nai = 15
        self.PNasc = 1e-05
        self.initialize_rate_constants()
        self._reusable_y0 = np.zeros(4)
        self.create_default_protocol()
    def initialize_rate_constants(self):
        """Pre-calculates voltage-dependent rate constants and GHK current factor.

        This method computes the alpha and beta rate constants for the m1, m2, m3
        (activation) and h (inactivation) gates across the pre-defined
        `self.v_range` (voltage range). These rates are characteristic of
        Hodgkin-Huxley type models and describe the transition rates between
        permissive and non-permissive states of the channel gates.

        The calculations for alpha rates often involve expressions of the form
        `A * (V - V_half) / (1 - exp(-(V - V_half) / k))`, which can be
        indeterminate when `V - V_half` is close to zero. The method handles
        this by using L'Hôpital's rule or a limiting value for these specific cases
        (e.g., `alpha_m1_vec[mask_m1] = 1.5`).

        The calculated rate vectors are stored as attributes:
        - `self.alpha_m1_vec`, `self.beta_m1_vec`
        - `self.alpha_m2_vec`, `self.beta_m2_vec`
        - `self.alpha_m3_vec`, `self.beta_m3_vec`
        - `self.alpha_h_vec`, `self.beta_h_vec`

        Additionally, it pre-computes a factor related to the Goldman-Hodgkin-Katz
        (GHK) current equation across `self.v_range` by calling
        `self._compute_ghk_current(self.v_range)` and stores it in `self.iscft`.
        This factor likely represents the current per unit of open probability,
        simplifying current calculations later.
        """
        v = self.v_range
        mask_m1 = (np.abs((v + 40)) < 1e-06)
        self.alpha_m1_vec = np.zeros_like(v)
        self.alpha_m1_vec[(~ mask_m1)] = ((0.15 * (v[(~ mask_m1)] + 40)) / (1 - np.exp(((- (v[(~ mask_m1)] + 40)) / 10))))
        self.alpha_m1_vec[mask_m1] = 1.5
        self.beta_m1_vec = (6.0 * np.exp(((- (v + 65)) / 18)))
        mask_m2 = (np.abs((v + 35)) < 1e-06)
        self.alpha_m2_vec = np.zeros_like(v)
        self.alpha_m2_vec[(~ mask_m2)] = ((0.08 * (v[(~ mask_m2)] + 35)) / (1 - np.exp(((- (v[(~ mask_m2)] + 35)) / 12))))
        self.alpha_m2_vec[mask_m2] = 0.667
        self.beta_m2_vec = (2.5 * np.exp(((- (v + 60)) / 20)))
        mask_m3 = (np.abs((v + 30)) < 1e-06)
        self.alpha_m3_vec = np.zeros_like(v)
        self.alpha_m3_vec[(~ mask_m3)] = ((0.04 * (v[(~ mask_m3)] + 30)) / (1 - np.exp(((- (v[(~ mask_m3)] + 30)) / 15))))
        self.alpha_m3_vec[mask_m3] = 0.267
        self.beta_m3_vec = (1.0 * np.exp(((- (v + 55)) / 25)))
        self.alpha_h_vec = (0.07 * np.exp(((- (v + 65)) / 20)))
        self.beta_h_vec = (1.0 / (np.exp(((- (v + 35)) / 10)) + 1.0))
        self.iscft = self._compute_ghk_current(self.v_range)
    def _compute_ghk_current(self, V):
        """Computes the Goldman-Hodgkin-Katz (GHK) current factor for sodium.

        This method calculates a component of the GHK current equation for sodium
        ions, given a membrane potential or array of potentials. The GHK equation
        is used to describe the ionic current across a cell membrane, considering
        the permeability of the membrane to specific ions and their concentration
        gradients.

        The calculation handles the singularity at V=0 by applying L'Hôpital's
        rule or using the limiting form of the GHK equation.

        Args:
            V (float or np.ndarray): Membrane potential(s) in millivolts (mV).

        Returns:
            np.ndarray: An array of current values (or a component of current,
                e.g., current per unit open probability or permeability)
                corresponding to the input voltage(s). The units would depend
                on the exact formulation and the units of `PNasc`, `F`, `Rgc`, `Tkel`,
                `Nai`, and `Nao`.
        
        Note:
            The term 'current' in the return value might represent a factor that,
            when multiplied by the open probability of sodium channels, gives the
            total sodium current. The variable `PNasc` (sodium permeability coefficient)
            is used, suggesting this calculation is specific to sodium ions.
        """
        V = np.atleast_1d(V)
        current = np.zeros_like(V, dtype=float)
        v_volts = (V * 0.001)
        near_zero = (np.abs(v_volts) < 1e-06)
        not_zero = (~ near_zero)
        if np.any(near_zero):
            du2_zero = ((self.F * self.F) / (self.Rgc * self.Tkel))
            current[near_zero] = ((self.PNasc * du2_zero) * (self.Nai - self.Nao))
        if np.any(not_zero):
            v_nz = v_volts[not_zero]
            du1 = ((v_nz * self.F) / (self.Rgc * self.Tkel))
            du2 = (self.F * du1)
            du3 = np.exp((- du1))
            current[not_zero] = (((self.PNasc * du2) * (self.Nai - (self.Nao * du3))) / (1 - du3))
        return current
    def get_rate_constants(self, V):
        """Retrieves pre-calculated rate constants for a given voltage(s).

        This method looks up the alpha and beta rate constants for the m1, m2, m3
        (activation) and h (inactivation) gates at a specific membrane
        potential `V`. It uses the `self.v_range` and the pre-calculated
        rate vectors (e.g., `self.alpha_m1_vec`) that were initialized by
        `initialize_rate_constants`.

        The method finds the closest index in `self.v_range` corresponding to
        the input `V` (or each voltage in `V` if it's an array) and returns
        the rates stored at that index.

        Args:
            V (float or np.ndarray): The membrane potential(s) in millivolts (mV)
                for which to retrieve the rate constants.

        Returns:
            dict: A dictionary containing the alpha and beta rate constants for
                each gate (m1, m2, m3, h) at the specified voltage(s).
                For example:
                {
                    'alpha_m1': ..., 'beta_m1': ...,
                    'alpha_m2': ..., 'beta_m2': ...,
                    'alpha_m3': ..., 'beta_m3': ...,
                    'alpha_h': ..., 'beta_h': ...
                }
                If `V` is an array, the values in the dictionary will also be arrays.
        """
        V = np.atleast_1d(V)
        v_idx = np.searchsorted(self.v_range, V)
        v_idx = np.clip(v_idx, 0, (len(self.v_range) - 1))
        return {'alpha_m1': self.alpha_m1_vec[v_idx], 'beta_m1': self.beta_m1_vec[v_idx], 'alpha_m2': self.alpha_m2_vec[v_idx], 'beta_m2': self.beta_m2_vec[v_idx], 'alpha_m3': self.alpha_m3_vec[v_idx], 'beta_m3': self.beta_m3_vec[v_idx], 'alpha_h': self.alpha_h_vec[v_idx], 'beta_h': self.beta_h_vec[v_idx]}
    def steady_state_values(self, V):
        """Calculates steady-state values for gating variables at a given voltage.

        This method computes the steady-state activation (m1_inf, m2_inf, m3_inf)
        and inactivation (h_inf) values for the Hodgkin-Huxley model gates
        at a specified membrane potential `V`.

        The steady-state value for a gate `x` (where x can be m1, m2, m3, or h)
        is calculated as `alpha_x / (alpha_x + beta_x)`, where `alpha_x` and
        `beta_x` are the voltage-dependent opening and closing rate constants
        for that gate, respectively. These rates are obtained by calling
        `self.get_rate_constants(V)`.

        Args:
            V (float or np.ndarray): The membrane potential(s) in millivolts (mV)
                at which to calculate the steady-state values.

        Returns:
            tuple: A tuple containing the steady-state values for m1, m2, m3, and h
                gates, in that order: (m1_inf, m2_inf, m3_inf, h_inf).
                If `V` is an array, each element in the tuple will also be an array.
        """
        rates = self.get_rate_constants(V)
        m1_inf = (rates['alpha_m1'] / (rates['alpha_m1'] + rates['beta_m1']))
        m2_inf = (rates['alpha_m2'] / (rates['alpha_m2'] + rates['beta_m2']))
        m3_inf = (rates['alpha_m3'] / (rates['alpha_m3'] + rates['beta_m3']))
        h_inf = (rates['alpha_h'] / (rates['alpha_h'] + rates['beta_h']))
        return (m1_inf, m2_inf, m3_inf, h_inf)
    def time_constants(self, V):
        """Calculates time constants for gating variables at a given voltage.

        This method computes the time constants (tau_m1, tau_m2, tau_m3, tau_h)
        for the activation (m1, m2, m3) and inactivation (h) gates of the
        Hodgkin-Huxley model at a specified membrane potential `V`.

        The time constant for a gate `x` (where x can be m1, m2, m3, or h)
        is calculated as `1.0 / (alpha_x + beta_x)`, where `alpha_x` and
        `beta_x` are the voltage-dependent opening and closing rate constants
        for that gate, respectively. These rates are obtained by calling
        `self.get_rate_constants(V)`.

        Args:
            V (float or np.ndarray): The membrane potential(s) in millivolts (mV)
                at which to calculate the time constants.

        Returns:
            tuple: A tuple containing the time constants for m1, m2, m3, and h
                gates, in that order: (tau_m1, tau_m2, tau_m3, tau_h).
                The units of the time constants will be the inverse of the units
                of the rate constants (e.g., ms if rates are in ms^-1).
                If `V` is an array, each element in the tuple will also be an array.
        """
        rates = self.get_rate_constants(V)
        tau_m1 = (1.0 / (rates['alpha_m1'] + rates['beta_m1']))
        tau_m2 = (1.0 / (rates['alpha_m2'] + rates['beta_m2']))
        tau_m3 = (1.0 / (rates['alpha_m3'] + rates['beta_m3']))
        tau_h = (1.0 / (rates['alpha_h'] + rates['beta_h']))
        return (tau_m1, tau_m2, tau_m3, tau_h)
    def compute_sodium_current(self, V, m1, m2, m3, h):
        """Computes the total sodium current using the GHK formulation.

        This method calculates the macroscopic sodium current (I_Na) based on
        the membrane potential (V) and the probabilities of the activation
        (m1, m2, m3) and inactivation (h) gates being in their permissive states.

        The calculation involves:
        1. Determining the open probability of a channel: `open_prob = m1 * m2 * m3 * h`.
           This assumes that the three 'm' gates and one 'h' gate operate independently.
        2. Retrieving a pre-calculated Goldman-Hodgkin-Katz (GHK) current factor
           (`ghk_current`) from `self.iscft` based on the voltage `V`. This factor,
           derived from `_compute_ghk_current`, incorporates sodium permeability
           (`self.PNasc`) and ionic concentrations.
        3. Calculating the total current:
           `current = open_prob * ghk_current * self.numchan * 0.0105`.
           - `self.numchan` is the number of channels.
           - `0.0105` is an additional scaling factor.

        This method uses the GHK current equation rather than an Ohmic driving force
        (i.e., it does not directly use `g_Na` or `E_Na` in this specific calculation,
        relying instead on `self.PNasc` and ionic concentrations embedded in `ghk_current`).

        Args:
            V (float or np.ndarray): Membrane potential(s) in millivolts (mV).
            m1 (float or np.ndarray): Probability of the first activation gate (m1) being open.
            m2 (float or np.ndarray): Probability of the second activation gate (m2) being open.
            m3 (float or np.ndarray): Probability of the third activation gate (m3) being open.
            h (float or np.ndarray): Probability of the inactivation gate (h) being open.

        Returns:
            np.ndarray: The total sodium current. The units depend on the units of
                the GHK factor and the scaling constant. If `V` and gating variables
                are arrays, the output will be an array.
        """
        V = np.atleast_1d(V)
        m1 = np.atleast_1d(m1)
        m2 = np.atleast_1d(m2)
        m3 = np.atleast_1d(m3)
        h = np.atleast_1d(h)
        if ((V.shape[0] == 1) and (m1.shape[0] > 1)):
            V = np.full_like(m1, V[0])
        v_idx = np.searchsorted(self.v_range, V)
        v_idx = np.clip(v_idx, 0, (len(self.v_range) - 1))
        ghk_current = self.iscft[v_idx]
        open_prob = (((m1 * m2) * m3) * h)
        current = (((open_prob * ghk_current) * self.numchan) * 0.0105)
        return current
    def Sweep(self, sweep_no):
        """Simulates a single voltage clamp sweep from the defined protocol.

        This method executes one sweep (trial) of a voltage clamp experiment,
        as defined by the `sweep_no` in the `self.SwpSeq` protocol array.
        A sweep can consist of multiple 'epochs', where each epoch is a period
        of constant voltage.

        The simulation proceeds epoch by epoch:
        1. Initial state: For the first epoch, gating variables (m1, m2, m3, h)
           are initialized to their steady-state values at the initial voltage.
           For subsequent epochs, they start from their values at the end of the
           previous epoch.
        2. ODE solving: Within each epoch, the Hodgkin-Huxley differential
           equations for the gating variables are solved using `scipy.integrate.solve_ivp`
           with the 'LSODA' method. The `derivatives` nested function defines these ODEs.
        3. Current calculation: At each time point evaluated by the ODE solver,
           the sodium current is calculated using `self.compute_sodium_current`.
        4. Storage: The calculated currents (`self.SimSwp`), command voltages
           (`self.SimCom`), and open probabilities (`self.SimOp`) are stored
           for each time point of the sweep. The time vector is stored in `self.time`.

        The `self.SwpSeq` array defines the protocol:
        - `SwpSeq[0, sweep_no]`: Number of epochs in the sweep.
        - `SwpSeq[2, sweep_no]`: Voltage of the first (holding) epoch.
        - `SwpSeq[2*e, sweep_no]`: Voltage of epoch `e` (for e > 0).
        - `SwpSeq[2*e + 1, sweep_no]`: End time point (index) of epoch `e`.

        Args:
            sweep_no (int): The index of the sweep to simulate from `self.SwpSeq`.

        Returns:
            float: The minimum (most negative) current value observed during the sweep.

        Raises:
            ValueError: If `sweep_no` is invalid or if the protocol definition
                in `self.SwpSeq` is malformed for the given `sweep_no`.
        
        Note:
            This method populates `self.SimSwp`, `self.SimCom`, `self.SimOp`, and
            `self.time` with the results of the current sweep.
        """
        if ((sweep_no >= self.SwpSeq.shape[1]) or (sweep_no < 0)):
            raise ValueError(f'Invalid sweep number {sweep_no}')
        SwpSeq = self.SwpSeq
        NumEpchs = int(SwpSeq[(0, sweep_no)])
        if ((NumEpchs <= 0) or (((2 * NumEpchs) + 1) >= SwpSeq.shape[0])):
            raise ValueError('Invalid number of epochs in protocol')
        total_points = (int(SwpSeq[(((2 * NumEpchs) + 1), sweep_no)]) + 1)
        self.SimSwp = np.zeros(total_points)
        self.SimCom = np.zeros(total_points)
        self.SimOp = np.zeros(total_points)
        epoch_voltages = np.zeros((NumEpchs + 1))
        epoch_end_times = np.zeros((NumEpchs + 1), dtype=int)
        epoch_voltages[0] = SwpSeq[(2, sweep_no)]
        epoch_end_times[0] = 0
        for e in range(1, (NumEpchs + 1)):
            epoch_voltages[e] = SwpSeq[((2 * e), sweep_no)]
            epoch_end_times[e] = int(SwpSeq[(((2 * e) + 1), sweep_no)])
        (m1_inf, m2_inf, m3_inf, h_inf) = self.steady_state_values(epoch_voltages[0])
        m1_current = m1_inf
        m2_current = m2_inf
        m3_current = m3_inf
        h_current = h_inf
        self.SimSwp[0] = self.compute_sodium_current(epoch_voltages[0], m1_current, m2_current, m3_current, h_current)
        self.SimCom[0] = epoch_voltages[0]
        self.SimOp[0] = (((m1_current * m2_current) * m3_current) * h_current)
        store_idx = 0
        for epoch in range(1, (NumEpchs + 1)):
            epoch_voltage = epoch_voltages[epoch]
            epoch_end_idx = epoch_end_times[epoch]
            rates = self.get_rate_constants(epoch_voltage)
            epoch_start_time = (store_idx * self.sampint)
            epoch_end_time = (epoch_end_idx * self.sampint)
            num_points = max(2, (int(((epoch_end_time - epoch_start_time) / self.sampint)) + 1))
            t_eval = np.linspace(epoch_start_time, epoch_end_time, num_points)
            if (len(t_eval) <= 1):
                continue
            def derivatives(t, y):
                """Defines the system of ODEs for Hodgkin-Huxley gating variables.

                This nested function calculates the time derivatives (dm1/dt, dm2/dt,
                dm3/dt, dh/dt) for the activation (m1, m2, m3) and inactivation (h)
                gates of the sodium channel model.

                The derivative for each gate `x` is given by:
                `dx/dt = alpha_x * (1 - x) - beta_x * x`
                where `alpha_x` and `beta_x` are the voltage-dependent rate constants
                for the current epoch's voltage, obtained from the `rates` dictionary
                (which is in the scope of the parent `Sweep` function).

                Args:

                Returns:
                    np.ndarray: A 1D array containing the calculated derivatives
                                [dm1/dt, dm2/dt, dm3/dt, dh/dt].
                """
                (m1, m2, m3, h) = (y.flatten() if hasattr(y, 'flatten') else y)
                dm1dt = ((rates['alpha_m1'] * (1 - m1)) - (rates['beta_m1'] * m1))
                dm2dt = ((rates['alpha_m2'] * (1 - m2)) - (rates['beta_m2'] * m2))
                dm3dt = ((rates['alpha_m3'] * (1 - m3)) - (rates['beta_m3'] * m3))
                dhdt = ((rates['alpha_h'] * (1 - h)) - (rates['beta_h'] * h))
                result = np.array([dm1dt, dm2dt, dm3dt, dhdt], dtype=float).flatten()
                return result
            self._reusable_y0[0] = m1_current
            self._reusable_y0[1] = m2_current
            self._reusable_y0[2] = m3_current
            self._reusable_y0[3] = h_current
            sol = solve_ivp(derivatives, [epoch_start_time, epoch_end_time], self._reusable_y0, method='LSODA', t_eval=t_eval, rtol=1e-06, atol=1e-08)
            if (sol.success and (len(sol.t) > 0)):
                start_idx = (store_idx + 1)
                end_idx = min((start_idx + len(sol.t)), total_points)
                actual_end = (end_idx - start_idx)
                if (actual_end > 0):
                    m1_vals = sol.y[0, :actual_end]
                    m2_vals = sol.y[1, :actual_end]
                    m3_vals = sol.y[2, :actual_end]
                    h_vals = sol.y[3, :actual_end]
                    indices = np.arange(start_idx, end_idx)
                    self.SimSwp[indices] = self.compute_sodium_current(epoch_voltage, m1_vals, m2_vals, m3_vals, h_vals)
                    self.SimCom[indices] = epoch_voltage
                    self.SimOp[indices] = (((m1_vals * m2_vals) * m3_vals) * h_vals)
                    if (epoch_end_idx < total_points):
                        m1_current = m1_vals[(- 1)]
                        m2_current = m2_vals[(- 1)]
                        m3_current = m3_vals[(- 1)]
                        h_current = h_vals[(- 1)]
            store_idx = epoch_end_idx
        self.time = (np.arange(total_points) * self.sampint)
        return np.min(self.SimSwp)
    def EquilOccup(self, voltage=None):
        """Calculates equilibrium state occupancies for a simplified state model.

        This method computes the steady-state probability distribution across a
        pre-defined 20-state array, representing a simplified mapping of the
        Hodgkin-Huxley model's states. It is likely designed for compatibility
        with other models that use a larger, explicit state space.

        The calculation proceeds as follows:
        1. Determines the steady-state values (m1_inf, m2_inf, m3_inf, h_inf)
           for the given voltage using `self.steady_state_values`.
        2. Calculates the open probability (`Po`) as the product of the individual
           gate probabilities: `m1_inf * m2_inf * m3_inf * h_inf`. This value is
           placed in index 6 of the output array.
        3. The total probability of the channel being inactivated (i.e., the `h`
           gate is not permissive) is `1 - h_inf`. This probability is
           distributed evenly across 6 states, from index 7 to 12.
        4. The remaining states in the 20-element array are left at zero.

        Args:
            voltage (float, optional): The membrane potential (in mV) at which to
                calculate equilibrium occupancies. If None, `self.vm` is used.
                Defaults to None.

        Returns:
            np.ndarray: A 20-element numpy array representing the equilibrium
                        probabilities of the model's states. `pop[6]` contains
                        the open probability.
        """
        V = (voltage if (voltage is not None) else self.vm)
        (m1_inf, m2_inf, m3_inf, h_inf) = self.steady_state_values(V)
        pop = np.zeros(20)
        pop[6] = (((m1_inf * m2_inf) * m3_inf) * h_inf)
        inact_prob = ((1 - h_inf) / 6)
        pop[7:13] = inact_prob
        return pop
    def create_default_protocol(self, target_voltages=None, holding_potential=(- 80), holding_duration=98, test_duration=200, tail_duration=2):
        """Creates a default multi-step voltage clamp protocol for activation.

        This method generates a standard voltage-activation protocol, which is
        often used to measure the current-voltage (I-V) relationship of the channel.
        The protocol consists of multiple sweeps, where each sweep has three epochs:
        1. A holding period at a set `holding_potential`.
        2. A test pulse to one of the specified `target_voltages`.
        3. A brief tail pulse, returning to the `holding_potential`.

        The method populates `self.SwpSeq` with the details of this protocol,
        where each column represents a sweep for a different test voltage.
        Durations are converted from milliseconds to sample points based on a
        hardcoded sampling interval of 0.005 ms.

        Args:
            target_voltages (list or np.ndarray, optional): A list of voltages (mV)
                to be used for the test pulse in each sweep. Defaults to
                [30, 0, -20, -30, -40, -50, -60].
            holding_potential (float, optional): The membrane potential (mV) during
                the holding and tail periods. Defaults to -80 mV.
            holding_duration (float, optional): The duration (ms) of the initial
                holding period. Defaults to 98 ms.
            test_duration (float, optional): The duration (ms) of the test pulse.
                Defaults to 200 ms.
            tail_duration (float, optional): The duration (ms) of the final tail
                pulse. Defaults to 2 ms.
        """
        self.BsNm = 'MultiStepKeyVoltages'
        if (target_voltages is None):
            target_voltages = [30, 0, (- 20), (- 30), (- 40), (- 50), (- 60)]
        target_voltages = np.array(target_voltages)
        self.NumSwps = len(target_voltages)
        self.SwpSeq = np.zeros((8, self.NumSwps))
        holding_samples = int((holding_duration / 0.005))
        test_samples = int((test_duration / 0.005))
        tail_samples = int((tail_duration / 0.005))
        total_samples = ((holding_samples + test_samples) + tail_samples)
        self.SwpSeq[0, :] = 3
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        self.SwpSeq[4, :] = target_voltages
        self.SwpSeq[5, :] = (holding_samples + test_samples)
        self.SwpSeq[6, :] = holding_potential
        self.SwpSeq[7, :] = total_samples
        setattr(self, f'SwpSeq{self.BsNm}', self.SwpSeq.copy())
    def create_inactivation_protocol(self, inactivating_voltage=(- 20), test_voltage=0, inactivating_duration=2000, recovery_duration=100):
        """Creates a single-sweep protocol to measure channel inactivation.

        This method generates a four-epoch voltage clamp protocol designed to
        characterize the voltage-dependent inactivation of the channel. The
        protocol consists of a single sweep with the following sequence:
        1. A holding period at -80 mV to establish a baseline state.
        2. A long inactivating pre-pulse at a specified `inactivating_voltage`
           to allow channels to enter the inactivated state.
        3. A brief test pulse to a `test_voltage` to measure the fraction of
           channels that remain available (i.e., not inactivated).
        4. A final recovery period at -80 mV.

        The method populates `self.SwpSeq` with the details for this single sweep.
        Durations are converted from milliseconds to sample points based on a
        hardcoded sampling interval of 0.005 ms.

        Args:
            inactivating_voltage (float, optional): The membrane potential (mV) of
                the inactivating pre-pulse. Defaults to -20 mV.
            test_voltage (float, optional): The membrane potential (mV) of the
                subsequent test pulse. Defaults to 0 mV.
            inactivating_duration (float, optional): The duration (ms) of the
                inactivating pre-pulse. Defaults to 2000 ms.
            recovery_duration (float, optional): The duration (ms) of the final
                recovery period. Defaults to 100 ms.
        """
        self.BsNm = 'InactivationProtocol'
        self.NumSwps = 1
        self.SwpSeq = np.zeros((10, 1))
        sampint = 0.005
        holding_duration = 200
        holding_samples = int((holding_duration / sampint))
        inactivating_samples = int((inactivating_duration / sampint))
        test_samples = int((5 / sampint))
        recovery_samples = int((recovery_duration / sampint))
        self.SwpSeq[(0, 0)] = 4
        self.SwpSeq[(2, 0)] = (- 80)
        self.SwpSeq[(3, 0)] = holding_samples
        self.SwpSeq[(4, 0)] = inactivating_voltage
        self.SwpSeq[(5, 0)] = (holding_samples + inactivating_samples)
        self.SwpSeq[(6, 0)] = test_voltage
        self.SwpSeq[(7, 0)] = ((holding_samples + inactivating_samples) + test_samples)
        self.SwpSeq[(8, 0)] = (- 80)
        self.SwpSeq[(9, 0)] = (((holding_samples + inactivating_samples) + test_samples) + recovery_samples)
        setattr(self, f'SwpSeq{self.BsNm}', self.SwpSeq.copy())
    def create_recovery_protocol(self, target_recovery_times=None, holding_potential=(- 80), inactivating_voltage=(- 20), test_voltage=0, holding_duration=200, inactivating_duration=2000, test_duration=20, tail_duration=100):
        """Creates a protocol to measure the time course of recovery from inactivation.

        This method generates a multi-sweep protocol where each sweep measures the
        channel availability after a different recovery duration. This is a classic
        two-pulse experiment to determine the rate of recovery from inactivation.

        Each sweep consists of five epochs:
        1. A holding period at `holding_potential`.
        2. An inactivating pulse at `inactivating_voltage` to inactivate the channels.
        3. A variable-duration recovery period at `holding_potential`.
        4. A test pulse to `test_voltage` to measure the fraction of recovered channels.
        5. A final tail pulse at `holding_potential`.

        The key variable across sweeps is the duration of the recovery period (Epoch 3),
        which is determined by the values in `target_recovery_times`.

        Args:
            target_recovery_times (list or np.ndarray, optional): A list of recovery
                durations (ms) to test. Each time will correspond to a separate sweep.
                Defaults to [1, 3, 10, 30, 100, 300, 1000].
            holding_potential (float, optional): The potential (mV) for holding,
                recovery, and tail epochs. Defaults to -80 mV.
            inactivating_voltage (float, optional): The potential (mV) of the
                inactivating pulse. Defaults to -20 mV.
            test_voltage (float, optional): The potential (mV) of the test pulse.
                Defaults to 0 mV.
            holding_duration (float, optional): The duration (ms) of the initial
                holding period. Defaults to 200 ms.
            inactivating_duration (float, optional): The duration (ms) of the
                inactivating pulse. Defaults to 2000 ms.
            test_duration (float, optional): The duration (ms) of the test pulse.
                Defaults to 20 ms.
            tail_duration (float, optional): The duration (ms) of the final tail
                pulse. Defaults to 100 ms.
        """
        self.BsNm = 'RecoveryFromInactivation'
        if (target_recovery_times is None):
            target_recovery_times = [1, 3, 10, 30, 100, 300, 1000]
        target_recovery_times = np.array(target_recovery_times)
        self.NumSwps = len(target_recovery_times)
        self.SwpSeq = np.zeros((12, self.NumSwps))
        sampint = 0.005
        holding_samples = int((holding_duration / sampint))
        inactivating_samples = int((inactivating_duration / sampint))
        test_samples = int((test_duration / sampint))
        tail_samples = int((tail_duration / sampint))
        recovery_samples = (target_recovery_times / sampint).astype(int)
        self.SwpSeq[:, 0] = 5
        self.SwpSeq[:, 2] = holding_potential
        self.SwpSeq[:, 3] = holding_samples
        self.SwpSeq[:, 4] = inactivating_voltage
        self.SwpSeq[:, 5] = (holding_samples + inactivating_samples)
        self.SwpSeq[:, 6] = holding_potential
        self.SwpSeq[:, 7] = ((holding_samples + inactivating_samples) + recovery_samples)
        self.SwpSeq[:, 8] = test_voltage
        self.SwpSeq[:, 9] = (((holding_samples + inactivating_samples) + recovery_samples) + test_samples)
        self.SwpSeq[:, 10] = holding_potential
        self.SwpSeq[:, 11] = ((((holding_samples + inactivating_samples) + recovery_samples) + test_samples) + tail_samples)
        setattr(self, f'SwpSeq{self.BsNm}', self.SwpSeq.copy())
    def create_steady_state_inactivation_protocol(self, test_voltages=None, holding_potential=(- 120), prepulse_duration=2000, test_pulse_voltage=0, test_pulse_duration=5, recovery_duration=100):
        """Creates a protocol to measure steady-state inactivation (availability).

        This method generates a multi-sweep voltage clamp protocol designed to
        determine the channel's steady-state inactivation curve (also known as the
        availability curve). This curve plots the fraction of available channels
        as a function of the pre-pulse potential.

        Each sweep consists of four epochs:
        1. A holding period at `holding_potential`.
        2. A long conditioning pre-pulse to one of the `test_voltages` to allow
           the channel population to reach equilibrium between open, closed, and
           inactivated states at that voltage.
        3. A brief, strong test pulse at a fixed `test_pulse_voltage` to measure
           the peak current from the fraction of channels that were not inactivated
           during the pre-pulse.
        4. A final recovery period at `holding_potential`.

        The key variable across sweeps is the voltage of the pre-pulse (Epoch 2),
        which is iterated through the provided `test_voltages`.

        Args:
            test_voltages (list or np.ndarray, optional): A list of pre-pulse
                potentials (mV) to test. Each voltage will correspond to a
                separate sweep. Defaults to a range from -120 mV to -15 mV.
            holding_potential (float, optional): The potential (mV) for the initial
                holding and final recovery periods. Defaults to -120 mV.
            prepulse_duration (float, optional): The duration (ms) of the conditioning
                pre-pulse. Should be long enough to reach steady state. Defaults to 2000 ms.
            test_pulse_voltage (float, optional): The fixed potential (mV) of the
                test pulse. Defaults to 0 mV.
            test_pulse_duration (float, optional): The duration (ms) of the test
                pulse. Defaults to 5 ms.
            recovery_duration (float, optional): The duration (ms) of the final
                recovery period. Defaults to 100 ms.
        """
        self.BsNm = 'SteadyStateInactivation'
        if (test_voltages is None):
            test_voltages = np.arange((- 120), (- 15), 5)
        test_voltages = np.array(test_voltages)
        self.NumSwps = len(test_voltages)
        self.SwpSeq = np.zeros((10, self.NumSwps))
        sampint = 0.005
        holding_samples = int((200 / sampint))
        prepulse_samples = int((prepulse_duration / sampint))
        test_samples = int((test_pulse_duration / sampint))
        recovery_samples = int((recovery_duration / sampint))
        self.SwpSeq[:, 0] = 4
        self.SwpSeq[:, 2] = holding_potential
        self.SwpSeq[:, 3] = holding_samples
        self.SwpSeq[:, 4] = test_voltages
        self.SwpSeq[:, 5] = (holding_samples + prepulse_samples)
        self.SwpSeq[:, 6] = test_pulse_voltage
        self.SwpSeq[:, 7] = ((holding_samples + prepulse_samples) + test_samples)
        self.SwpSeq[:, 8] = holding_potential
        self.SwpSeq[:, 9] = (((holding_samples + prepulse_samples) + test_samples) + recovery_samples)
        setattr(self, f'SwpSeq{self.BsNm}', self.SwpSeq.copy())