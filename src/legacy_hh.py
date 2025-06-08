import numpy as np
from scipy.integrate import solve_ivp

class HHModel:
    """Extended Hodgkin-Huxley model with 4 gates for sodium channels.
    
    Uses three separate activation gates (m1, m2, m3) and one inactivation gate (h)
    to better capture the sequential activation seen in Markov models.
    
    Current formula: I_Na = g_Na * m1 * m2 * m3 * h * (V - E_Na)
    """
    
    def __init__(self):
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
        """Initialize rate constants for all 4 gates"""
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
        """Compute GHK current for given voltages"""
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
        """Get all rate constants for a given voltage"""
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
        """Calculate steady-state values for all gates"""
        rates = self.get_rate_constants(V)
        
        m1_inf = rates['alpha_m1'] / (rates['alpha_m1'] + rates['beta_m1'])
        m2_inf = rates['alpha_m2'] / (rates['alpha_m2'] + rates['beta_m2'])
        m3_inf = rates['alpha_m3'] / (rates['alpha_m3'] + rates['beta_m3'])
        h_inf = rates['alpha_h'] / (rates['alpha_h'] + rates['beta_h'])
        
        return m1_inf, m2_inf, m3_inf, h_inf
    
    def time_constants(self, V):
        """Calculate time constants for all gates"""
        rates = self.get_rate_constants(V)
        
        tau_m1 = 1.0 / (rates['alpha_m1'] + rates['beta_m1'])
        tau_m2 = 1.0 / (rates['alpha_m2'] + rates['beta_m2'])
        tau_m3 = 1.0 / (rates['alpha_m3'] + rates['beta_m3'])
        tau_h = 1.0 / (rates['alpha_h'] + rates['beta_h'])
        
        return tau_m1, tau_m2, tau_m3, tau_h
    
    def compute_sodium_current(self, V, m1, m2, m3, h):
        """Compute sodium current using 4-gate model"""
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
        """Run simulation for a given sweep"""
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
        """Calculate equilibrium occupancy (compatible with MarkovModel)"""
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
        """Create voltage clamp protocol"""
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