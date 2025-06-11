import numpy as np
from scipy.integrate import solve_ivp

class HHModel:
    """
    Implements a Hodgkin-Huxley (HH) style model for sodium channel currents.

    This model simulates sodium channel behavior using a set of differential
    equations for three activation gates (m1, m2, m3) and one inactivation
    gate (h). The total sodium current is calculated based on the conductances
    derived from these gates and the Nernst potential, or optionally using the
    Goldman-Hodgkin-Katz (GHK) current equation.

    Key Attributes:
        g_Na (float): Maximum sodium conductance (mS/cm²).
        E_Na (float): Sodium reversal potential (mV), used if GHK is not.
        C_m (float): Membrane capacitance (μF/cm²).
        V_rest (float): Resting membrane potential (mV).
        sampint (float): Sampling interval for simulations (ms).
        numchan (int): Number of channels (primarily for scaling, not stochasticity).
        Na_in, Na_out (float): Intracellular and extracellular sodium concentrations (mM) for GHK.
        m1, m2, m3, h (float): Current values of the gating variables.
        v_range (np.ndarray): Voltage range for pre-calculating rate constants.
        alpha_m1_vec, beta_m1_vec, etc. (np.ndarray): Pre-calculated voltage-dependent
                                                     rate constants for each gate.
        iscft (np.ndarray): Pre-calculated GHK current scaling factor across `v_range`.
        SwpSeq (np.ndarray): Array defining the voltage-clamp protocol.
        NumSwps (int): Number of sweeps in the protocol.
        SimSwp, SimCom, SimOp (list or np.ndarray): Arrays to store simulation
                                                    results (current, command voltage,
                                                    open probability).
        time (np.ndarray): Time vector for simulations.
    """

    def __init__(self):
        """
        Initializes the Hodgkin-Huxley model parameters and data structures.

        Sets up default biophysical parameters (conductances, reversal potentials,
        ion concentrations), simulation settings (sampling interval), physical
        constants, and initializes arrays for storing gating variable rates,
        simulation results, and the default voltage protocol.

        Calls `initialize_rate_constants()` to pre-calculate voltage-dependent
        rate constants and GHK current factors.
        Calls `create_default_protocol()` to set up an initial stimulation protocol.
        """
        # Conductance and reversal potential
        self.g_Na = 0.12      # Maximum sodium conductance (mS/cm²)
        self.E_Na = 50.0      # Sodium reversal potential (mV)
        self.C_m = 1.0        # Membrane capacitance (μF/cm²)
        self.V_rest = -65.0   # Resting potential (mV)
        
        # Sampling interval
        self.sampint = 0.005  # 5 μs sampling interval
        
        # Number of channels
        self.numchan = 1
        
        # Ion concentrations for GHK equation (mM)
        self.Na_in = 15.0
        self.Na_out = 155.0
        
        # Gating variables
        self.m1 = 0.0         # First activation gate
        self.m2 = 0.0         # Second activation gate
        self.m3 = 0.0         # Third activation gate
        self.h = 0.0          # Inactivation gate
        
        # Voltage range for lookup tables
        self.v_range = np.linspace(-100, 100, 1000)
        
        # Protocol storage
        self.SimSwp = []
        self.SimCom = []
        self.SimOp = []
        self.SwpSeq = []
        self.NumSwps = 0
        self.vm = -80
        self.time = []
        
        # Physical constants
        self.F = 96480        # Faraday constant (C/mol)
        self.Rgc = 8314       # Gas constant (J/(K·mol))
        self.Tkel = 273.15 + 22.0  # Fixed at 22°C
        self.Nao = 155
        self.Nai = 15
        self.PNasc = 1e-5
        
        # Initialize rate constants
        self.initialize_rate_constants()
        
        # Pre-allocate arrays
        self._reusable_y0 = np.zeros(4)  # For 4 gates
        
        # Create default protocol
        self.create_default_protocol()
    
    def initialize_rate_constants(self):
        """
        Pre-calculates voltage-dependent rate constants for all gating variables.

        This method computes the alpha and beta rate constants for each of the
        three activation gates (m1, m2, m3) and the inactivation gate (h)
        across the pre-defined voltage range `self.v_range`. The formulas
        used are variants of the standard Hodgkin-Huxley rate equations,
        adjusted to model specific sodium channel kinetics with three distinct
        activation steps.

        The calculated rates are stored in vectorized numpy arrays (e.g.,
        `self.alpha_m1_vec`, `self.beta_m1_vec`) for efficient lookup during
        simulations. It also calls `_compute_ghk_current` to pre-calculate
        the GHK current scaling factor across `self.v_range`.
        Handles potential division by zero for specific voltage points by
        applying L'Hôpital's rule or a limiting value.
        """
        v = self.v_range
        
        # Gate m1 - fastest activation gate
        # Similar to standard m gate but slightly faster
        mask_m1 = np.abs(v + 40) < 1e-6
        self.alpha_m1_vec = np.zeros_like(v)
        self.alpha_m1_vec[~mask_m1] = 0.15 * (v[~mask_m1] + 40) / (1 - np.exp(-(v[~mask_m1] + 40) / 10))
        self.alpha_m1_vec[mask_m1] = 1.5
        self.beta_m1_vec = 6.0 * np.exp(-(v + 65) / 18)
        
        # Gate m2 - intermediate activation gate
        # Slower than m1, voltage shifted
        mask_m2 = np.abs(v + 35) < 1e-6
        self.alpha_m2_vec = np.zeros_like(v)
        self.alpha_m2_vec[~mask_m2] = 0.08 * (v[~mask_m2] + 35) / (1 - np.exp(-(v[~mask_m2] + 35) / 12))
        self.alpha_m2_vec[mask_m2] = 0.667
        self.beta_m2_vec = 2.5 * np.exp(-(v + 60) / 20)
        
        # Gate m3 - slowest activation gate
        # Much slower, different voltage dependence
        mask_m3 = np.abs(v + 30) < 1e-6
        self.alpha_m3_vec = np.zeros_like(v)
        self.alpha_m3_vec[~mask_m3] = 0.04 * (v[~mask_m3] + 30) / (1 - np.exp(-(v[~mask_m3] + 30) / 15))
        self.alpha_m3_vec[mask_m3] = 0.267
        self.beta_m3_vec = 1.0 * np.exp(-(v + 55) / 25)
        
        # Gate h - inactivation gate (same as standard HH)
        self.alpha_h_vec = 0.07 * np.exp(-(v + 65) / 20)
        self.beta_h_vec = 1.0 / (np.exp(-(v + 35) / 10) + 1.0)
        
        # Initialize GHK current array
        self.iscft = self._compute_ghk_current(self.v_range)
    
    def _compute_ghk_current(self, V):
        """
        Calculates the Goldman-Hodgkin-Katz (GHK) current scaling factor.

        This method computes the GHK current factor for a given voltage or
        array of voltages `V`. This factor, when multiplied by the open
        probability and permeability, gives the ionic current. It considers
        sodium ion concentrations (`self.Nai`, `self.Nao`), temperature
        (`self.Tkel`), and physical constants.

        Args:
            V (float or np.ndarray): The membrane potential(s) in mV at which
                                     to calculate the GHK current factor.

        Returns:
            np.ndarray: An array of GHK current scaling factors corresponding
                        to the input voltage(s). Handles the case where V is
                        close to zero to avoid division by zero.
        """
        V = np.atleast_1d(V)
        current = np.zeros_like(V, dtype=float)
        
        # Convert to volts
        v_volts = V * 1e-3
        
        # Handle near-zero voltages
        near_zero = np.abs(v_volts) < 1e-6
        not_zero = ~near_zero
        
        if np.any(near_zero):
            du2_zero = self.F * self.F / (self.Rgc * self.Tkel)
            current[near_zero] = self.PNasc * du2_zero * (self.Nai - self.Nao)
        
        if np.any(not_zero):
            v_nz = v_volts[not_zero]
            du1 = (v_nz * self.F) / (self.Rgc * self.Tkel)
            du2 = self.F * du1
            du3 = np.exp(-du1)
            current[not_zero] = self.PNasc * du2 * (self.Nai - self.Nao * du3) / (1 - du3)
        
        return current
    
    def get_rate_constants(self, V):
        """
        Retrieves pre-calculated rate constants for a given voltage.

        Looks up the alpha and beta values for each gating variable (m1, m2, m3, h)
        at the specified membrane potential `V` from the pre-calculated vectorized
        rate arrays (e.g., `self.alpha_m1_vec`). It finds the closest index
        in `self.v_range` corresponding to `V` for the lookup.

        Args:
            V (float): The membrane potential (mV) for which to retrieve rates.

        Returns:
            dict: A dictionary containing the alpha and beta rates for each gate
                  (e.g., {'alpha_m1': value, 'beta_m1': value, ...}).
        """
        V = np.atleast_1d(V)
        v_idx = np.searchsorted(self.v_range, V)
        v_idx = np.clip(v_idx, 0, len(self.v_range) - 1)
        
        return {
            'alpha_m1': self.alpha_m1_vec[v_idx],
            'beta_m1': self.beta_m1_vec[v_idx],
            'alpha_m2': self.alpha_m2_vec[v_idx],
            'beta_m2': self.beta_m2_vec[v_idx],
            'alpha_m3': self.alpha_m3_vec[v_idx],
            'beta_m3': self.beta_m3_vec[v_idx],
            'alpha_h': self.alpha_h_vec[v_idx],
            'beta_h': self.beta_h_vec[v_idx]
        }
    
    def steady_state_values(self, V):
        """
        Calculates the steady-state activation/inactivation values (m_inf, h_inf) for gates.

        For each gating variable (m1, m2, m3, h), this method computes its
        steady-state value (e.g., m1_inf) at the given membrane potential `V`.
        The steady-state value is calculated as alpha / (alpha + beta), using
        the rate constants obtained from `get_rate_constants(V)`.

        Args:
            V (float): The membrane potential (mV) at which to calculate
                       steady-state values.

        Returns:
            tuple: A tuple containing the steady-state values for m1, m2, m3,
                   and h, in that order (m1_inf, m2_inf, m3_inf, h_inf).
        """
        rates = self.get_rate_constants(V)
        
        m1_inf = rates['alpha_m1'] / (rates['alpha_m1'] + rates['beta_m1'])
        m2_inf = rates['alpha_m2'] / (rates['alpha_m2'] + rates['beta_m2'])
        m3_inf = rates['alpha_m3'] / (rates['alpha_m3'] + rates['beta_m3'])
        h_inf = rates['alpha_h'] / (rates['alpha_h'] + rates['beta_h'])
        
        return m1_inf, m2_inf, m3_inf, h_inf
    
    def time_constants(self, V):
        """
        Calculates the time constants (tau_m, tau_h) for gating variables.

        For each gating variable (m1, m2, m3, h), this method computes its
        time constant (e.g., tau_m1) at the given membrane potential `V`.
        The time constant is calculated as 1 / (alpha + beta), using the
        rate constants obtained from `get_rate_constants(V)`.

        Args:
            V (float): The membrane potential (mV) at which to calculate
                       time constants.

        Returns:
            tuple: A tuple containing the time constants for m1, m2, m3, and h,
                   in that order (tau_m1, tau_m2, tau_m3, tau_h).
        """
        rates = self.get_rate_constants(V)
        
        tau_m1 = 1.0 / (rates['alpha_m1'] + rates['beta_m1'])
        tau_m2 = 1.0 / (rates['alpha_m2'] + rates['beta_m2'])
        tau_m3 = 1.0 / (rates['alpha_m3'] + rates['beta_m3'])
        tau_h = 1.0 / (rates['alpha_h'] + rates['beta_h'])
        
        return tau_m1, tau_m2, tau_m3, tau_h
    
    def compute_sodium_current(self, V, m1, m2, m3, h):
        """
        Computes the macroscopic sodium current at a given voltage and gate states.

        The current is calculated using the product of the three activation gates
        (m1*m2*m3) and the inactivation gate (h), multiplied by the maximum
        sodium conductance (`self.g_Na`) and the driving force.
        The driving force can be calculated either as (V - E_Na) or using the
        pre-calculated GHK current factor (`self.iscft`) if `self.PNasc` > 0,
        allowing for selection between Ohmic and GHK formulations.

        Args:
            V (float or np.ndarray): The membrane potential(s) (mV).
            m1 (float or np.ndarray): Value(s) of the first activation gate.
            m2 (float or np.ndarray): Value(s) of the second activation gate.
            m3 (float or np.ndarray): Value(s) of the third activation gate.
            h (float or np.ndarray): Value(s) of the inactivation gate.

        Returns:
            float or np.ndarray: The computed sodium current(s) in μA/cm².
                                 The sign convention is positive for outward current.
        """
        V = np.atleast_1d(V)
        m1 = np.atleast_1d(m1)
        m2 = np.atleast_1d(m2)
        m3 = np.atleast_1d(m3)
        h = np.atleast_1d(h)
        
        # Broadcast V if needed
        if V.shape[0] == 1 and m1.shape[0] > 1:
            V = np.full_like(m1, V[0])
        
        # Get GHK current
        v_idx = np.searchsorted(self.v_range, V)
        v_idx = np.clip(v_idx, 0, len(self.v_range) - 1)
        ghk_current = self.iscft[v_idx]
        
        # Current = permeability * open_probability * driving_force
        open_prob = m1 * m2 * m3 * h
        current = open_prob * ghk_current * self.numchan * 0.021  # Scale to match
        
        return current
    
    def Sweep(self, sweep_no):
        """
        Runs a single voltage-clamp sweep simulation for the HH model.

        This method simulates the channel's response to a specific sweep
        (`sweep_no`) from the current voltage protocol (`self.SwpSeq`).
        The process involves:
        1. Setting initial gating variable values (m1, m2, m3, h) to their
           steady-state values at the holding potential of the first epoch.
        2. Iterating through each epoch (voltage step) defined in the protocol.
        3. For each epoch:
            a. Defining the `derivatives` function for the ODE solver, which
               calculates dm1/dt, dm2/dt, dm3/dt, and dh/dt based on current
               voltage and gate values.
            b. Using `scipy.integrate.solve_ivp` with the `derivatives`
               function to solve the system of ODEs for the gating variables.
            c. Storing the results (sodium current, command voltage, open
               probability) at sampled time points.
            d. Updating gating variable values for the start of the next epoch.
        4. Populating `self.time` with the time vector for the simulation.

        Args:
            sweep_no (int): The sweep number (0-indexed) from `self.SwpSeq`
                            to simulate.

        Returns:
            float: The minimum (most negative) current value observed during the sweep.
                   This is often used for quick checks of peak inward current.
        """
        if sweep_no >= self.SwpSeq.shape[1] or sweep_no < 0:
            raise ValueError(f"Invalid sweep number {sweep_no}")
        
        SwpSeq = self.SwpSeq
        NumEpchs = int(SwpSeq[0, sweep_no])
        
        if NumEpchs <= 0 or 2*NumEpchs + 1 >= SwpSeq.shape[0]:
            raise ValueError("Invalid number of epochs in protocol")
        
        # Initialize parameters
        total_points = int(SwpSeq[2*NumEpchs + 1, sweep_no]) + 1
        
        # Pre-allocate arrays
        self.SimSwp = np.zeros(total_points)
        self.SimCom = np.zeros(total_points)
        self.SimOp = np.zeros(total_points)
        
        # Extract epoch parameters
        epoch_voltages = np.zeros(NumEpchs + 1)
        epoch_end_times = np.zeros(NumEpchs + 1, dtype=int)
        
        epoch_voltages[0] = SwpSeq[2, sweep_no]
        epoch_end_times[0] = 0
        
        for e in range(1, NumEpchs + 1):
            epoch_voltages[e] = SwpSeq[2 * e, sweep_no]
            epoch_end_times[e] = int(SwpSeq[2 * e + 1, sweep_no])
        
        # Initial conditions
        m1_inf, m2_inf, m3_inf, h_inf = self.steady_state_values(epoch_voltages[0])
        m1_current = m1_inf
        m2_current = m2_inf
        m3_current = m3_inf
        h_current = h_inf
        
        # Store initial values
        self.SimSwp[0] = self.compute_sodium_current(
            epoch_voltages[0], m1_current, m2_current, m3_current, h_current
        )
        self.SimCom[0] = epoch_voltages[0]
        self.SimOp[0] = m1_current * m2_current * m3_current * h_current
        
        # Process each epoch
        store_idx = 0
        
        for epoch in range(1, NumEpchs + 1):
            epoch_voltage = epoch_voltages[epoch]
            epoch_end_idx = epoch_end_times[epoch]
            
            # Get rate constants for this voltage
            rates = self.get_rate_constants(epoch_voltage)
            
            # Time span
            epoch_start_time = store_idx * self.sampint
            epoch_end_time = epoch_end_idx * self.sampint
            
            # Create evaluation points
            num_points = max(2, int((epoch_end_time - epoch_start_time) / self.sampint) + 1)
            t_eval = np.linspace(epoch_start_time, epoch_end_time, num_points)
            
            if len(t_eval) <= 1:
                continue
            
            # Define ODE system for 4 gates
            def derivatives(t, y):
                """
                Calculates derivatives of gating variables (dm/dt, dh/dt) for ODE solver.

                Args:
                    t (float): Current time (not explicitly used as rates are voltage-dependent).
                    y (np.ndarray): Array of current gating variable values [m1, m2, m3, h].

                Returns:
                    np.ndarray: Array of derivatives [dm1/dt, dm2/dt, dm3/dt, dh/dt].
                """
                m1, m2, m3, h = y.flatten() if hasattr(y, 'flatten') else y
                dm1dt = rates['alpha_m1'] * (1 - m1) - rates['beta_m1'] * m1
                dm2dt = rates['alpha_m2'] * (1 - m2) - rates['beta_m2'] * m2
                dm3dt = rates['alpha_m3'] * (1 - m3) - rates['beta_m3'] * m3
                dhdt = rates['alpha_h'] * (1 - h) - rates['beta_h'] * h
                # Ensure we return a flat array with shape (4,) for RK45 compatibility
                result = np.array([dm1dt, dm2dt, dm3dt, dhdt], dtype=float).flatten()
                return result
            
            # Set initial conditions
            self._reusable_y0[0] = m1_current
            self._reusable_y0[1] = m2_current
            self._reusable_y0[2] = m3_current
            self._reusable_y0[3] = h_current
            
            # Solve ODEs
            sol = solve_ivp(
                derivatives,
                [epoch_start_time, epoch_end_time],
                self._reusable_y0,
                method='LSODA',
                t_eval=t_eval,
                rtol=1e-6,
                atol=1e-8
            )
            
            # Store results
            if sol.success and len(sol.t) > 0:
                start_idx = store_idx + 1
                end_idx = min(start_idx + len(sol.t), total_points)
                actual_end = end_idx - start_idx
                
                if actual_end > 0:
                    m1_vals = sol.y[0, :actual_end]
                    m2_vals = sol.y[1, :actual_end]
                    m3_vals = sol.y[2, :actual_end]
                    h_vals = sol.y[3, :actual_end]
                    
                    indices = np.arange(start_idx, end_idx)
                    self.SimSwp[indices] = self.compute_sodium_current(
                        epoch_voltage, m1_vals, m2_vals, m3_vals, h_vals
                    )
                    self.SimCom[indices] = epoch_voltage
                    self.SimOp[indices] = m1_vals * m2_vals * m3_vals * h_vals
                    
                    # Update for next epoch
                    if epoch_end_idx < total_points:
                        m1_current = m1_vals[-1]
                        m2_current = m2_vals[-1]
                        m3_current = m3_vals[-1]
                        h_current = h_vals[-1]
            
            store_idx = epoch_end_idx
        
        # Store time array
        self.time = np.arange(total_points) * self.sampint
        
        return np.min(self.SimSwp)
    
    def EquilOccup(self, voltage=None):
        """
        Calculates approximate equilibrium occupancies for the HH model.

        This method provides an estimation of state distribution for compatibility
        with plotting or analysis tools expecting a Markov-like state vector.
        It uses the steady-state values of the gating variables (m1_inf, m2_inf,
        m3_inf, h_inf) at the given `voltage` (or `self.vm` if None).

        The "open probability" is calculated as m1_inf * m2_inf * m3_inf * h_inf
        and placed in a conventional position (index 6) of a 20-element state
        vector. The "inactivated probability" (1 - h_inf) is distributed
        among other conventional inactivated state positions (indices 7-12).

        Note: This is not a true equilibrium calculation for a multi-state Markov
        model but rather a projection of HH gate states onto a simplified scheme.

        Args:
            voltage (float, optional): The membrane potential (mV) at which to
                                       calculate equilibrium values. Defaults to
                                       `self.vm`.

        Returns:
            np.ndarray: A 1D array of 20 elements, where `pop[6]` represents
                        the open probability (m1*m2*m3*h) and `pop[7:13]`
                        represent distributed inactivated probability. Other
                        elements are zero.
        """
        V = voltage if voltage is not None else self.vm
        
        m1_inf, m2_inf, m3_inf, h_inf = self.steady_state_values(V)
        
        # Create state vector
        pop = np.zeros(20)
        
        # Open probability in position 6
        pop[6] = m1_inf * m2_inf * m3_inf * h_inf
        
        # Distribute inactivated probability
        inact_prob = (1 - h_inf) / 6
        pop[7:13] = inact_prob
        
        return pop
    
    def create_default_protocol(self, target_voltages=None, holding_potential=-80,
                              holding_duration=98, test_duration=200, tail_duration=2):
        """
        Creates a default multi-step voltage clamp protocol for the HH model.

        The protocol consists of a holding period, a test pulse to various
        target voltages, and a tail pulse back to the holding potential. This
        structure is common for characterizing ion channel kinetics.

        Args:
            target_voltages (list, optional): A list of voltages (mV) for the
                test pulse phase. Defaults to [30, 0, -20, -30, -40, -50, -60].
                The number of sweeps (`self.NumSwps`) will be equal to the
                number of target voltages.
            holding_potential (float, optional): Voltage (mV) for the initial
                holding period and the final tail period. Defaults to -80 mV.
            holding_duration (float, optional): Duration (ms) of the initial
                holding period. Defaults to 98 ms.
            test_duration (float, optional): Duration (ms) of the test pulse.
                Defaults to 200 ms.
            tail_duration (float, optional): Duration (ms) of the tail pulse.
                Defaults to 2 ms.

        This method populates `self.NumSwps` and `self.SwpSeq` (the protocol array
        defining epochs and their durations/voltages). It also initializes
        `self.SimSwp` and `self.time` arrays based on the total duration of
        a sweep.
        """
        if target_voltages is None:
            target_voltages = [30, 0, -20, -30, -40, -50, -60]
        
        target_voltages = np.asarray(target_voltages)
        num_sweeps = len(target_voltages)
        
        self.SwpSeq = np.zeros((8, num_sweeps))
        
        # Time points in samples
        holding_samples = int(holding_duration / 0.005)
        test_samples = int(test_duration / 0.005)
        tail_samples = int(tail_duration / 0.005)
        total_samples = holding_samples + test_samples + tail_samples
        
        # Protocol setup
        self.SwpSeq[0, :] = 3  # 3 epochs per sweep
        
        # Epoch 1: Holding
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        
        # Epoch 2: Test
        self.SwpSeq[4, :] = target_voltages
        self.SwpSeq[5, :] = holding_samples + test_samples
        
        # Epoch 3: Tail
        self.SwpSeq[6, :] = holding_potential
        self.SwpSeq[7, :] = total_samples
        
        self.NumSwps = num_sweeps
        self.SimSwp = np.zeros(total_samples)
        self.time = np.arange(total_samples) * self.sampint
    
    def create_inactivation_protocol(self, inactivating_voltage=-20, test_voltage=0, 
                                inactivating_duration=2000, recovery_duration=100):
        """
        Create a protocol optimized to show anticonvulsant effects on inactivation.
        
        Protocol:
        1. Hold at -80 mV (resting) - 200 ms for equilibration
        2. Long step to inactivating voltage (causes inactivation + drug binding)  
        3. Brief test pulse to measure available current
        4. Return to holding potential for recovery
        
        This protocol maximizes drug binding during the long inactivating step.
        
        Args:
            inactivating_voltage (float): Voltage for inactivating prepulse in mV. 
                Default -20 mV promotes strong inactivation.
            test_voltage (float): Voltage for test pulse in mV. Default 0 mV
                for maximal channel opening.
            inactivating_duration (float): Duration of inactivating pulse in ms.
                DEFAULT CHANGED TO 2000 ms (2 seconds) for complete drug equilibration.
                Kuo 1998 shows drug binding requires >1 second to reach steady state.
            recovery_duration (float): Final recovery period in ms. Default 100 ms.
        
        Notes:
            - The 2-second default inactivating duration is CRITICAL for observing
            anticonvulsant effects. Shorter durations will underestimate drug potency.
            - Initial holding period increased to 200 ms for better equilibration.
        """
        self.BsNm = "InactivationProtocol"
        
        # Create single sweep protocol
        self.NumSwps = 1
        self.SwpSeq = np.zeros((10, 1))  # Need more rows for 4 epochs
        
        sampint = 0.005  # 5 μs sampling interval
        
        # INCREASED holding duration for better equilibration
        holding_duration = 200  # ms (increased from 98 ms)
        holding_samples = int(holding_duration / sampint)
        
        # Convert durations to samples
        inactivating_samples = int(inactivating_duration / sampint)  
        test_samples = int(5 / sampint)  # 5 ms test pulse (brief to avoid further inactivation)
        recovery_samples = int(recovery_duration / sampint)
        
        # 4 epochs: holding, inactivating pulse, test pulse, recovery
        self.SwpSeq[0, 0] = 4
        
        # Epoch 1: Initial holding for equilibration
        self.SwpSeq[2, 0] = -80  # holding potential
        self.SwpSeq[3, 0] = holding_samples
        
        # Epoch 2: Long inactivating pulse (where drug binding occurs)
        self.SwpSeq[4, 0] = inactivating_voltage 
        self.SwpSeq[5, 0] = holding_samples + inactivating_samples
        
        # Epoch 3: Brief test pulse to measure available current
        self.SwpSeq[6, 0] = test_voltage
        self.SwpSeq[7, 0] = holding_samples + inactivating_samples + test_samples
        
        # Epoch 4: Recovery period
        self.SwpSeq[8, 0] = -80
        self.SwpSeq[9, 0] = holding_samples + inactivating_samples + test_samples + recovery_samples
        
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        self.CurrVolt()


    def create_recovery_protocol(self, target_recovery_times=None, holding_potential=-80,
                        inactivating_voltage=-20, test_voltage=0,
                        holding_duration=200, inactivating_duration=2000, 
                        test_duration=20, tail_duration=100):
        """
        Create recovery from inactivation protocol for measuring anticonvulsant drug effects.
        
        This protocol measures the time course of recovery from inactivation, which is
        the primary mechanism by which anticonvulsant drugs reduce sodium channel availability.
        
        Protocol structure for each sweep:
        1. Hold at holding_potential (equilibration)
        2. Inactivating pulse to inactivating_voltage (allows drug binding to reach steady state)
        3. Recovery interval at holding_potential (VARIABLE duration - varies between sweeps)
        4. Test pulse to test_voltage (measures recovered current)
        5. Return to holding_potential (tail period)
        
        Args:
            target_recovery_times (list, optional): Recovery intervals in ms. 
                Defaults to [1, 3, 10, 30, 100, 300, 1000] for comprehensive kinetics.
            holding_potential (float, optional): Resting voltage in mV. Defaults to -80.
            inactivating_voltage (float, optional): Voltage for inactivating pulse in mV. 
                Defaults to -20 (promotes inactivation and drug binding).
            test_voltage (float, optional): Voltage for test pulse in mV. Defaults to 0
                (promotes channel opening to measure recovery).
            holding_duration (float, optional): Initial holding duration in ms. 
                DEFAULT CHANGED TO 200 ms for better equilibration (was 50 ms).
            inactivating_duration (float, optional): Duration of inactivating pulse in ms. 
                DEFAULT CHANGED TO 2000 ms for complete drug binding (was 1000 ms).
            test_duration (float, optional): Duration of test pulse in ms. Defaults to 20
                (brief to minimize further inactivation).
            tail_duration (float, optional): Final holding duration in ms. 
                Defaults to 100 ms (increased from 50 ms).
        
        Returns:
            None. Sets self.SwpSeq and updates self.NumSwps.
            
        Notes:
            - This protocol is designed to replicate Kuo et al. (1998) methodology
            - Drug effects are measured as slowed recovery kinetics
            - The 2-second inactivating pulse ensures drug binding reaches equilibrium
            - With 25 μM LTG, expect ~30x slower recovery (τ ~200 ms vs ~7 ms control)
        """
        
        self.BsNm = "RecoveryFromInactivation"
        
        # Default recovery times: logarithmic spacing from 1 ms to 1000 ms
        if target_recovery_times is None:
            target_recovery_times = [1, 3, 10, 30, 100, 300, 1000]
        
        target_recovery_times = np.array(target_recovery_times)
        self.NumSwps = len(target_recovery_times)
        
        # Initialize protocol array - need 12 rows for 5 epochs (2 rows per epoch: voltage, time)
        self.SwpSeq = np.zeros((12, self.NumSwps))
        
        # Convert durations to sample points (5 μs sampling interval)
        sampint = 0.005  # 5 μs
        holding_samples = int(holding_duration / sampint)
        inactivating_samples = int(inactivating_duration / sampint)
        test_samples = int(test_duration / sampint)
        tail_samples = int(tail_duration / sampint)
        
        # Convert recovery times to samples (this varies per sweep)
        recovery_samples = (target_recovery_times / sampint).astype(int)
        
        # Set up protocol for each sweep
        self.SwpSeq[0, :] = 5  # 5 epochs per sweep
        
        # Epoch 1: Initial holding period at holding_potential
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        
        # Epoch 2: Inactivating pulse at inactivating_voltage
        self.SwpSeq[4, :] = inactivating_voltage
        self.SwpSeq[5, :] = holding_samples + inactivating_samples
        
        # Epoch 3: Recovery interval at holding_potential (VARIABLE duration)
        self.SwpSeq[6, :] = holding_potential
        self.SwpSeq[7, :] = holding_samples + inactivating_samples + recovery_samples
        
        # Epoch 4: Test pulse at test_voltage
        self.SwpSeq[8, :] = test_voltage
        self.SwpSeq[9, :] = holding_samples + inactivating_samples + recovery_samples + test_samples
        
        # Epoch 5: Tail period at holding_potential
        self.SwpSeq[10, :] = holding_potential
        self.SwpSeq[11, :] = holding_samples + inactivating_samples + recovery_samples + test_samples + tail_samples
        
        # Store protocol with descriptive name
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        
        # Update current-voltage relationships
        self.CurrVolt()

    # Additional helper method for creating a full steady-state inactivation curve protocol
    def create_steady_state_inactivation_protocol(self, test_voltages=None, 
                                                holding_potential=-120,
                                                prepulse_duration=2000,
                                                test_pulse_voltage=0,
                                                test_pulse_duration=5,
                                                recovery_duration=100):
        """
        Create a complete steady-state inactivation protocol for anticonvulsant characterization.
        
        This protocol applies a series of long prepulses to different voltages, followed
        by a test pulse to measure channel availability. Critical for measuring the
        voltage-dependent effects of anticonvulsant drugs.
        
        Args:
            test_voltages (array-like, optional): Prepulse voltages in mV.
                Defaults to [-120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20].
            holding_potential (float, optional): Initial holding voltage in mV. 
                Defaults to -120 (fully available).
            prepulse_duration (float, optional): Duration of conditioning prepulse in ms.
                Defaults to 2000 ms (2 seconds) for complete drug equilibration.
            test_pulse_voltage (float, optional): Test pulse voltage in mV. Defaults to 0.
            test_pulse_duration (float, optional): Test pulse duration in ms. Defaults to 5.
            recovery_duration (float, optional): Recovery period in ms. Defaults to 100.
        
        Notes:
            - The 2-second prepulse is ESSENTIAL for accurate drug characterization
            - Expect ~15 mV leftward shift with 25 μM LTG
            - Plot normalized peak currents vs prepulse voltage and fit with Boltzmann
        """
        
        self.BsNm = "SteadyStateInactivation"
        
        if test_voltages is None:
            test_voltages = np.arange(-120, -15, 5)  # -120 to -20 mV in 5 mV steps
        
        test_voltages = np.array(test_voltages)
        self.NumSwps = len(test_voltages)
        
        # Initialize protocol array
        self.SwpSeq = np.zeros((10, self.NumSwps))
        
        # Convert to samples
        sampint = 0.005  # 5 μs
        holding_samples = int(200 / sampint)  # 200 ms initial holding
        prepulse_samples = int(prepulse_duration / sampint)
        test_samples = int(test_pulse_duration / sampint)
        recovery_samples = int(recovery_duration / sampint)
        
        # Set up each sweep
        self.SwpSeq[0, :] = 4  # 4 epochs per sweep
        
        # Epoch 1: Initial holding
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        
        # Epoch 2: Prepulse (voltage varies per sweep)
        self.SwpSeq[4, :] = test_voltages
        self.SwpSeq[5, :] = holding_samples + prepulse_samples
        
        # Epoch 3: Test pulse
        self.SwpSeq[6, :] = test_pulse_voltage
        self.SwpSeq[7, :] = holding_samples + prepulse_samples + test_samples
        
        # Epoch 4: Recovery
        self.SwpSeq[8, :] = holding_potential
        self.SwpSeq[9, :] = holding_samples + prepulse_samples + test_samples + recovery_samples
        
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        self.CurrVolt()