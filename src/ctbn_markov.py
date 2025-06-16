"""
Module for simulating voltage-gated ion channels using Continuous-Time Markov Networks (CTBN).

This module provides classes for a basic Hodgkin-Huxley-like Markov model (`CTBNMarkovModel`)
and an extended version that incorporates anticonvulsant drug interactions 
(`AnticonvulsantCTBNMarkovModel`). It supports various voltage clamp protocols for 
studying channel kinetics, activation, inactivation, and drug effects.
"""
import numpy as np
from scipy.integrate import solve_ivp
class CTBNMarkovModel():
    """Simulates a voltage-gated ion channel using a Continuous-Time Markov Network.

    This model is based on a Hodgkin-Huxley-like structure with multiple closed,
    open, and inactivated states. It calculates state transitions and ionic currents
    in response to voltage clamp protocols.

    Attributes:
        NumSwps (int): Number of sweeps in the current protocol.
        vm (float): Current membrane potential (mV).
        state_probs_flat (np.ndarray): Flattened array of current state probabilities.
        # ... (other attributes related to parameters, rates, and protocol definitions)
    """
    def __init__(self):
        """Initializes the CTBNMarkovModel.

        Sets up default parameters, initializes state variables, pre-calculates
        voltage-dependent rates, and sets up a default voltage clamp protocol.
        """
        self.NumSwps = 0
        self.demonstrate_cooperative_transition = False
        self.k_coop = 100.0
        self.k_phantom = 1000000.0
        self.A = 0
        self.I = 0
        self.num_states = 12
        self.vm = (- 80)
        self.init_parameters()
        self.init_waves()
        self.update_rates()
        self.CurrVolt()
        self.state_probs_flat = self.EquilOccup(self.vm)
        self.create_default_protocol()
    def init_parameters(self):
        """Initializes the core biophysical parameters of the ion channel model.

        These parameters define the voltage dependence of transition rates
        between states (e.g., alpha and beta rate constants for activation/deactivation,
        rates for inactivation). They are typically based on experimental data
        or literature values for a specific ion channel type.
        """
        self.alcoeff = 20
        self.alslp = 40
        self.btcoeff = 0.3
        self.btslp = 18.5
        self.ConCoeff = 0.004
        self.CoffCoeff = 4.5
        self.ConSlp = 100000000.0
        self.CoffSlp = 100000000.0
        self.gmcoeff = 50
        self.gmslp = 100
        self.dlcoeff = 0.8
        self.dlslp = 6
        self.OpOnCoeff = 4
        self.OpOffCoeff = 0.008
        self.ConHiCoeff = 4
        self.CoffHiCoeff = 0.008
        self.OpOnSlp = 100000000.0
        self.OpOffSlp = 100000000.0
        self.konlo = self.kofflo = self.konhi = self.koffhi = 0
        self.konop = self.koffop = self.kdlo = self.kdhi = 0
        self.alfac = np.sqrt(np.sqrt((self.ConHiCoeff / self.ConCoeff)))
        self.btfac = np.sqrt(np.sqrt((self.CoffCoeff / self.CoffHiCoeff)))
        self.numchan = 1
        self.F = 96485
        self.Rgc = 8314
        self.Tkel = 295
        (self.Nao, self.Nai) = (150, 15)
        self.ClipRate = 6000
        self.current_scaling = 0.0125
        self.PNasc = 1e-05
        self.vm = (- 80)
    def init_waves(self):
        """Initializes data structures for storing pre-calculated rate constants
        and state probabilities across a defined range of voltages (`self.vt`).

        This pre-calculation speeds up simulations by avoiding repeated computation
        of voltage-dependent rates during ODE solving. Arrays for forward/backward
        activation rates and inactivation rates are initialized.
        """
        self.vt = np.arange((- 200), 201)
        self.iscft = np.zeros_like(self.vt)
        self.state_probs_flat = np.zeros(12)
        self.state_probs_flat[0] = 1.0
        num_v = len(self.vt)
        self.fwd_rates_I0 = np.zeros((num_v, 5))
        self.fwd_rates_I1 = np.zeros((num_v, 5))
        self.bwd_rates_I0 = np.zeros((num_v, 5))
        self.bwd_rates_I1 = np.zeros((num_v, 5))
        self.inact_on_rates = np.zeros((num_v, 6))
        self.inact_off_rates = np.zeros((num_v, 6))
        self.k613dis_vec = np.zeros(num_v)
        self.k136dis_vec = np.zeros(num_v)
        self.update_rates()
    def update_rates(self):
        """Updates all voltage-dependent rate constants.
        
        This method typically calls `stRatesVolt` to recalculate rates based on the
        current model parameters and voltage range.
        """
        self.stRatesVolt()
    def stRatesVolt(self):
        """Calculates and stores state transition rates as a function of voltage.

        This is a core method that defines the kinetics of the Markov model.
        It populates arrays like `fwd_rates_I0`, `bwd_rates_I0`, etc., which store
        the forward and backward transition rates between states for different
        voltage levels.
        """
        vt = self.vt
        activation_scale = 1.0
        deactivation_scale = 1.0
        amt = (self.alcoeff * np.exp((vt / self.alslp)))
        bmt = (self.btcoeff * np.exp(((- vt) / self.btslp)))
        gmt = (self.gmcoeff * np.exp((vt / self.gmslp)))
        dmt = (self.dlcoeff * np.exp(((- vt) / self.dlslp)))
        konlo = (self.ConCoeff * np.exp((vt / self.ConSlp)))
        kofflo = (self.CoffCoeff * np.exp(((- vt) / self.CoffSlp)))
        konop = (self.OpOnCoeff * np.exp((vt / self.OpOnSlp)))
        koffop = (self.OpOffCoeff * np.exp(((- vt) / self.OpOffSlp)))
        for a in range(5):
            if (a < 4):
                self.fwd_rates_I0[:, a] = np.minimum(((4 - a) * amt), self.ClipRate)
            else:
                self.fwd_rates_I0[:, a] = np.minimum(gmt, self.ClipRate)
        for a in range(5):
            if (a < 4):
                self.bwd_rates_I0[:, a] = np.minimum(((a + 1) * bmt), self.ClipRate)
            else:
                self.bwd_rates_I0[:, a] = np.minimum(dmt, self.ClipRate)
        for a in range(5):
            if (a < 4):
                self.fwd_rates_I1[:, a] = np.minimum((((4 - a) * amt) * self.alfac), self.ClipRate)
            else:
                self.fwd_rates_I1[:, a] = np.minimum(gmt, self.ClipRate)
        for a in range(5):
            if (a < 4):
                self.bwd_rates_I1[:, a] = np.minimum((((a + 1) * bmt) / self.btfac), self.ClipRate)
            else:
                self.bwd_rates_I1[:, a] = np.minimum(dmt, self.ClipRate)
        alfac_powers = np.array([(self.alfac ** a) for a in range(5)])
        btfac_powers = np.array([(self.btfac ** a) for a in range(5)])
        for a in range(5):
            self.inact_on_rates[:, a] = np.minimum((konlo * alfac_powers[a]), self.ClipRate)
            self.inact_off_rates[:, a] = np.minimum((kofflo / btfac_powers[a]), self.ClipRate)
        self.inact_on_rates[:, 5] = np.minimum(konop, self.ClipRate)
        self.inact_off_rates[:, 5] = np.minimum(koffop, self.ClipRate)
    def CurrVolt(self):
        """Calculates the ionic current at various membrane potentials.

        Uses the Goldman-Hodgkin-Katz (GHK) equation or a similar formulation
        to determine current based on channel conductance (derived from open state
        probabilities), permeability, and ion concentrations across the specified
        voltage range `self.vt`.
        Stores the calculated current in `self.iscft`.
        """
        scaled_PNasc = self.PNasc
        v_volts = (self.vt * 0.001)
        near_zero = (np.abs(v_volts) < 1e-06)
        not_zero = (~ near_zero)
        self.iscft = np.zeros_like(v_volts)
        if np.any(near_zero):
            du2_zero = ((self.F * self.F) / (self.Rgc * self.Tkel))
            self.iscft[near_zero] = ((scaled_PNasc * du2_zero) * (self.Nai - self.Nao))
        if np.any(not_zero):
            v_nz = v_volts[not_zero]
            du1 = ((v_nz * self.F) / (self.Rgc * self.Tkel))
            du3 = np.exp((- du1))
            du5_corrected = (((self.F * du1) * (self.Nai - (self.Nao * du3))) / (1 - du3))
            self.iscft[not_zero] = (scaled_PNasc * du5_corrected)
    def EquilOccup(self, vm):
        """Calculates the equilibrium (steady-state) occupancies of all states.

        Args:
            vm (float): The membrane potential (mV) at which to calculate equilibrium.

        Returns:
            np.ndarray: A flattened array of equilibrium state probabilities.
        """
        self.vm = vm
        self.update_rates()
        vidx = np.argmin(np.abs((self.vt - vm)))
        fwd_I0 = self.fwd_rates_I0[vidx]
        bwd_I0 = self.bwd_rates_I0[vidx]
        fwd_I1 = self.fwd_rates_I1[vidx]
        bwd_I1 = self.bwd_rates_I1[vidx]
        rel_prob_A_I0 = np.ones(6)
        rel_prob_A_I0[1:] = np.cumprod((fwd_I0 / bwd_I0))
        rel_prob_A_I1 = np.ones(6)
        rel_prob_A_I1[1:] = np.cumprod((fwd_I1 / bwd_I1))
        rel_prob_A_I0 /= rel_prob_A_I0.sum()
        rel_prob_A_I1 /= rel_prob_A_I1.sum()
        inact_on = self.inact_on_rates[vidx]
        inact_off = self.inact_off_rates[vidx]
        total_rate_I0_to_I1 = np.dot(rel_prob_A_I0, inact_on)
        total_rate_I1_to_I0 = np.dot(rel_prob_A_I1, inact_off)
        if (total_rate_I1_to_I0 > 0):
            rel_prob_I1 = (total_rate_I0_to_I1 / total_rate_I1_to_I0)
        else:
            rel_prob_I1 = 0
        total_prob = (1 + rel_prob_I1)
        prob_I0 = (1 / total_prob)
        prob_I1 = (rel_prob_I1 / total_prob)
        eq_probs_flat = np.zeros(12)
        eq_probs_flat[:6] = (rel_prob_A_I0 * prob_I0)
        eq_probs_flat[6:12] = (rel_prob_A_I1 * prob_I1)
        return eq_probs_flat
    def NowDerivs(self, t, y):
        """Calculates the time derivatives of state probabilities (dy/dt).

        This method is typically used by an ODE solver during simulations.
        It computes the rate of change of occupancy for each state in the model
        based on the current state probabilities `y` and the transition rates
        at the current time `t` (or current voltage, if voltage is time-dependent).

        Args:
            t (float): Current time point in the simulation.
            y (np.ndarray): Array of current state probabilities.

        Returns:
            np.ndarray: Array of time derivatives for each state probability.
        """
        dstdt = np.zeros_like(y)
        if ((not hasattr(self, '_voltage_lut_cache')) or (self._voltage_lut_cache[0] != self.vm)):
            vidx = np.searchsorted(self.vt, self.vm)
            vidx = min(max(vidx, 0), (len(self.vt) - 1))
            self._voltage_lut_cache = (self.vm, vidx)
        else:
            vidx = self._voltage_lut_cache[1]
        if ((not hasattr(self, '_rate_cache')) or (self._rate_cache[0] != vidx)):
            _fwd_I0_orig = self.fwd_rates_I0[vidx]
            _bwd_I0_orig = self.bwd_rates_I0[vidx]
            _fwd_I1_orig = self.fwd_rates_I1[vidx]
            _bwd_I1_orig = self.bwd_rates_I1[vidx]
            _inact_on_orig = self.inact_on_rates[vidx]
            _inact_off_orig = self.inact_off_rates[vidx]
            self._rate_cache = (vidx, _fwd_I0_orig, _bwd_I0_orig, _fwd_I1_orig, _bwd_I1_orig, _inact_on_orig, _inact_off_orig)
        else:
            (_, _fwd_I0_orig, _bwd_I0_orig, _fwd_I1_orig, _bwd_I1_orig, _inact_on_orig, _inact_off_orig) = self._rate_cache
        current_fwd_I0 = np.copy(_fwd_I0_orig)
        current_bwd_I0 = np.copy(_bwd_I0_orig)
        current_fwd_I1 = _fwd_I1_orig
        current_bwd_I1 = _bwd_I1_orig
        current_inact_on = _inact_on_orig
        current_inact_off = _inact_off_orig
        if self.demonstrate_cooperative_transition:
            current_fwd_I0[0] = self.k_coop
            current_fwd_I0[1] = self.k_phantom
            current_bwd_I0[0] = 0.0
            current_bwd_I0[1] = 0.0
        probs_I0 = y[:6]
        probs_I1 = y[6:12]
        deriv_I0 = dstdt[:6]
        deriv_I1 = dstdt[6:12]
        for i in range(5):
            flux = (current_fwd_I0[i] * probs_I0[i])
            deriv_I0[i] -= flux
            deriv_I0[(i + 1)] += flux
        for i in range(5):
            flux = (current_bwd_I0[i] * probs_I0[(i + 1)])
            deriv_I0[(i + 1)] -= flux
            deriv_I0[i] += flux
        for i in range(5):
            flux = (current_fwd_I1[i] * probs_I1[i])
            deriv_I1[i] -= flux
            deriv_I1[(i + 1)] += flux
        for i in range(5):
            flux = (current_bwd_I1[i] * probs_I1[(i + 1)])
            deriv_I1[(i + 1)] -= flux
            deriv_I1[i] += flux
        for i in range(6):
            flux = (current_inact_on[i] * probs_I0[i])
            deriv_I0[i] -= flux
            deriv_I1[i] += flux
        for i in range(6):
            flux = (current_inact_off[i] * probs_I1[i])
            deriv_I1[i] -= flux
            deriv_I0[i] += flux
        return dstdt
    def _get_rates_at_vm(self, vm):
        """Helper method to get all pre-calculated rates at a specific voltage.

        Args:
            vm (float): The membrane potential (mV) for which to retrieve rates.

        Returns:
            tuple: A tuple containing (vidx, fwd_rates_I0, bwd_rates_I0, 
                   fwd_rates_I1, bwd_rates_I1, inact_on_rates, inact_off_rates)
                   for the given vm.
        """
        vidx = np.argmin(np.abs((self.vt - vm)))
        return {'fwd_I0': self.fwd_rates_I0[vidx], 'bwd_I0': self.bwd_rates_I0[vidx], 'fwd_I1': self.fwd_rates_I1[vidx], 'bwd_I1': self.bwd_rates_I1[vidx], 'inact_on': self.inact_on_rates[vidx], 'inact_off': self.inact_off_rates[vidx], 'k613': self.k613dis_vec[vidx], 'k136': self.k136dis_vec[vidx]}
    def _update_scalar_rates(self):
        """Updates scalar rate constants based on the current `self.vm`.

        This method is typically called when `self.vm` changes, to ensure that
        scalar rate attributes (like `self.k1`, `self.k2`, etc., representing
        rates at the specific `self.vm`) are current. It uses `_get_rates_at_vm`
        to fetch the array rates for the current `self.vm` and then updates
        the corresponding scalar attributes.
        """
        if (not hasattr(self, 'vm')):
            print(((f'Warning: CTBNMarkovModel instance (id: {id(self)}) ' + "does not have 'vm' attribute when _update_scalar_rates is called. ") + 'Rates cannot be updated.'))
            return
        if ((not hasattr(self, 'vt')) or (not hasattr(self, 'fwd_rates_I0'))):
            print(((f'Warning: CTBNMarkovModel instance (id: {id(self)}) may not be fully initialized ' + '(missing self.vt or vectorized rate arrays like self.fwd_rates_I0) ') + 'when _update_scalar_rates is called. Proceeding, but _get_rates_at_vm might fail.'))
        try:
            rates_at_vm = self._get_rates_at_vm(self.vm)
            for (rate_name, rate_value) in rates_at_vm.items():
                setattr(self, rate_name, rate_value)
        except AttributeError as e:
            print((f'Error in CTBNMarkovModel._update_scalar_rates for vm={self.vm}: ' + f'Failed to get or set rates. Underlying error: {e}'))
            raise
    def Sweep(self, SwpNo):
        """Simulates a single sweep of a voltage clamp protocol.

        A sweep consists of multiple epochs, each with a defined voltage and duration.
        This method iterates through the epochs defined in `self.SwpSeq` for the
        given sweep number (`SwpNo`). It uses an ODE solver (`solve_ivp`) to
        calculate the evolution of state probabilities over time for each epoch.

        Key attributes updated/used:
        - `self.SwpSeq`: Array defining the voltage clamp protocol sequence.
        - `self.NumSwps`: Total number of sweeps in the protocol.
        - `self.SimSwp`: Array to store simulated current for the sweep.
        - `self.SimOp`, `self.SimIn`, `self.SimAv`, `self.SimCom`: Arrays for
          storing probabilities of open, inactivated, available, and combined states.
        - `self.state_probs_flat`: Current state probabilities, flattened.
        - `self.vm`: Current membrane potential, updated for each epoch.
        - `self.NowDerivs`: Method providing dy/dt for the ODE solver.
        - `self.CurrVolt`: Method to calculate current from state probabilities.

        Args:
            SwpNo (int): The sweep number to simulate (0-indexed).

        Raises:
            ValueError: If `SwpNo` is invalid or if protocol definition is incorrect.
        """
        if ((SwpNo >= self.NumSwps) or (SwpNo < 0)):
            raise ValueError(f'Invalid sweep number {SwpNo}')
        NumEpchs = int(self.SwpSeq[(0, SwpNo)])
        if (NumEpchs <= 0):
            raise ValueError('Invalid number of epochs in protocol')
        total_points = (int(self.SwpSeq[(((2 * NumEpchs) + 1), SwpNo)]) + 1)
        sampint = 0.005
        self.SimSwp = np.zeros(total_points)
        self.SimOp = np.zeros(total_points)
        self.SimIn = np.zeros(total_points)
        self.SimAv = np.zeros(total_points)
        self.SimCom = np.zeros(total_points)
        self.state_probs_flat = np.zeros(12)
        self.state_probs_flat[0] = 1.0
        epoch_voltages = np.zeros((NumEpchs + 1))
        epoch_end_times = np.zeros((NumEpchs + 1))
        for e in range((NumEpchs + 1)):
            if (e == 0):
                epoch_voltages[e] = self.SwpSeq[(2, SwpNo)]
                epoch_end_times[e] = 0.0
            else:
                epoch_voltages[e] = self.SwpSeq[((2 * e), SwpNo)]
                epoch_end_times[e] = (int(self.SwpSeq[(((2 * e) + 1), SwpNo)]) * sampint)
        self.vm = epoch_voltages[0]
        self.CurrVolt()
        eq_pop = self.EquilOccup(self.vm)
        self.state_probs_flat[:6] = eq_pop[:6]
        self.state_probs_flat[6:12] = eq_pop[6:12]
        self._store_ctbn_results(0, 0)
        if ((not hasattr(self, '_reusable_y0')) or (len(self._reusable_y0) < 12)):
            self._reusable_y0 = np.zeros(12)
        current_time = 0.0
        store_idx = 1
        for epoch in range(1, (NumEpchs + 1)):
            self.vm = epoch_voltages[epoch]
            epoch_end_time = epoch_end_times[epoch]
            self.update_rates()
            self.CurrVolt()
            num_points = max(2, (int(((epoch_end_time - current_time) / sampint)) + 1))
            t_eval = np.linspace(current_time, epoch_end_time, num_points)
            if (len(t_eval) <= 1):
                current_time = epoch_end_time
                continue
            self._reusable_y0 = self.state_probs_flat
            sol = solve_ivp(self.NowDerivs, [current_time, epoch_end_time], self._reusable_y0, method='LSODA', t_eval=t_eval, rtol=1e-06, atol=1e-08)
            batch_size = len(sol.t)
            if (batch_size > 0):
                end_idx = min((store_idx + batch_size), total_points)
                batch_indices = np.arange(store_idx, end_idx)
                actual_batch_size = len(batch_indices)
                if (actual_batch_size > 0):
                    states_subset = sol.y[:, :actual_batch_size]
                    batch_states = states_subset.T
                    batch_voltages = np.full(actual_batch_size, self.vm)
                    self._store_ctbn_results_vectorized(batch_indices, batch_states, batch_voltages)
                    self.state_probs_flat = sol.y[:, (- 1)]
                    store_idx = end_idx
            current_time = epoch_end_time
        self.time = np.arange(0, (total_points * sampint), sampint)[:total_points]
        return (sol.t, self.SimSwp)
    def _store_ctbn_results(self, idx, t):
        """Stores simulation results for a single time point.

        This is a convenience wrapper around `_store_ctbn_results_vectorized`
        for storing results from a single time point (e.g., initial conditions).

        Args:
            idx (int): The index in the simulation arrays (SimSwp, SimOp, etc.)
                       where the results should be stored.
            t (float): The time corresponding to this data point (currently unused
                       by the vectorized version but kept for potential future use
                       or consistency).
        """
        self._store_ctbn_results_vectorized([idx], np.array([self.state_probs_flat]), np.array([self.vm]))
    def _store_ctbn_results_vectorized(self, indices, state_probs_batch, voltages):
        """Stores simulation results for a batch of time points in a vectorized manner.

        This method calculates and stores the simulated current (`SimSwp`),
        open probability (`SimOp`), inactivated probability (`SimIn`),
        available probability (`SimAv`), and command voltage (`SimCom`)
        for a batch of simulation time points.

        Args:
            indices (np.ndarray): Array of indices in the simulation output arrays
                                  (e.g., `self.SimSwp`) where results should be stored.
            state_probs_batch (np.ndarray): A 2D array where each row contains the
                                            state probabilities for a time point.
                                            Shape: (batch_size, num_states).
            voltages (np.ndarray): Array of membrane potentials corresponding to
                                   each time point in `state_probs_batch`.
                                   Shape: (batch_size,).
        """
        if (len(indices) == 0):
            return
        voltage_indices = np.searchsorted(self.vt, voltages)
        voltage_indices = np.clip(voltage_indices, 0, (len(self.vt) - 1))
        current_factors = self.iscft[voltage_indices]
        open_probs = state_probs_batch[:, 5]
        scale_factor = (self.numchan * self.current_scaling)
        currents = ((open_probs * current_factors) * scale_factor)
        inactivation = np.sum(state_probs_batch[:, 6:12], axis=1)
        available = np.sum(state_probs_batch[:, :6], axis=1)
        self.SimSwp[indices] = currents
        self.SimOp[indices] = open_probs
        self.SimIn[indices] = inactivation
        self.SimAv[indices] = available
        self.SimCom[indices] = voltages
    def create_default_protocol(self, target_voltages=None, holding_potential=(- 80), holding_duration=98, test_duration=200, tail_duration=2):
        """Creates a default multi-step voltage clamp protocol.

        This protocol typically consists of:
        1. A holding period at `holding_potential`.
        2. A series of test pulses to different `target_voltages`.
        3. A brief tail pulse back to `holding_potential` after each test pulse.

        The durations of holding, test, and tail periods are configurable.
        The generated protocol sequence is stored in `self.SwpSeq` and also
        saved as an attribute `self.SwpSeq{self.BsNm}`.

        Args:
            target_voltages (list or np.ndarray, optional): A list of voltages (mV)
                for the test pulses. Defaults to `[30, 0, -20, -30, -40, -50, -60]`.
            holding_potential (float, optional): Voltage (mV) for holding and tail
                periods. Defaults to -80 mV.
            holding_duration (float, optional): Duration (ms) of the initial holding
                period. Defaults to 98 ms.
            test_duration (float, optional): Duration (ms) of each test pulse.
                Defaults to 200 ms.
            tail_duration (float, optional): Duration (ms) of the tail pulse.
                Defaults to 2 ms.
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
        self.CurrVolt()
    def create_inactivation_protocol(self, inactivating_voltage=(- 20), test_voltage=0, inactivating_duration=2000, recovery_duration=100):
        """Creates a voltage clamp protocol to assess recovery from inactivation or use-dependent block.

        This protocol typically involves:
        1. An initial holding period (fixed at -80mV for 200ms).
        2. An inactivating/conditioning pre-pulse to `inactivating_voltage` for `inactivating_duration`.
        3. A test pulse to `test_voltage` (fixed duration of 10ms).
        4. A recovery period at `inactivating_voltage` (acting as holding) for `recovery_duration`.

        This protocol structure is suitable for measuring the time course of recovery
        from inactivation or for assessing use-dependent block by a drug, rather than
        generating a full steady-state inactivation (SSI) curve which would typically
        involve a series of varied pre-pulse potentials.

        The generated protocol sequence is stored in `self.SwpSeq` and also
        saved as an attribute `self.SwpSeq{self.BsNm}`.

        Args:
            inactivating_voltage (float, optional): Voltage (mV) of the inactivating
                pre-pulse and the subsequent recovery period. Defaults to -20 mV.
            test_voltage (float, optional): Voltage (mV) of the test pulse.
                Defaults to 0 mV.
            inactivating_duration (float, optional): Duration (ms) of the inactivating
                pre-pulse. Defaults to 2000 ms.
            recovery_duration (float, optional): Duration (ms) of the recovery period
                (effectively a tail pulse or inter-pulse interval at `inactivating_voltage`).
                Defaults to 100 ms.
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
        self.CurrVolt()
    def create_recovery_protocol(self, target_recovery_times=None, holding_potential=(- 80), inactivating_voltage=(- 20), test_voltage=0, holding_duration=200, inactivating_duration=2000, test_duration=20, tail_duration=100):
        """Creates a voltage clamp protocol to measure recovery from inactivation.

        This protocol is designed to assess the time course of recovery from
        steady-state inactivation. It typically consists of:
        1. An initial holding period at `holding_potential`.
        2. A long conditioning pre-pulse to `inactivating_voltage` to induce inactivation.
        3. A variable recovery interval at `holding_potential` (defined by `target_recovery_times`).
        4. A test pulse to `test_voltage` to measure available current after recovery.
        5. A final tail pulse back to `holding_potential`.

        The generated protocol sequence is stored in `self.SwpSeq` and also
        saved as an attribute `self.SwpSeq{self.BsNm}`.

        Args:
            target_recovery_times (list or np.ndarray, optional): A list of durations (ms)
                for the recovery intervals. Defaults to `[1, 3, 10, 30, 100, 300, 1000]`.
            holding_potential (float, optional): Voltage (mV) for holding, recovery, and
                tail periods. Defaults to -80 mV.
            inactivating_voltage (float, optional): Voltage (mV) of the conditioning
                pre-pulse. Defaults to -20 mV.
            test_voltage (float, optional): Voltage (mV) of the test pulse.
                Defaults to 0 mV.
            holding_duration (float, optional): Duration (ms) of the initial holding
                period. Defaults to 200 ms.
            inactivating_duration (float, optional): Duration (ms) of the conditioning
                pre-pulse. Defaults to 2000 ms.
            test_duration (float, optional): Duration (ms) of the test pulse.
                Defaults to 20 ms.
            tail_duration (float, optional): Duration (ms) of the final tail pulse.
                Defaults to 100 ms.
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
        self.CurrVolt()
    def create_steady_state_inactivation_protocol(self, test_voltages=None, holding_potential=(- 120), prepulse_duration=2000, test_pulse_voltage=0, test_pulse_duration=5, recovery_duration=100):
        """Creates a steady-state inactivation (SSI) voltage clamp protocol.

        This protocol is designed to determine the voltage-dependence of
        steady-state inactivation. It typically consists of:
        1. An initial holding period (e.g., at -80mV for 200ms, fixed in this implementation).
        2. A series of long conditioning pre-pulses to different `test_voltages`
           (which act as conditioning potentials) to allow channels to reach
           steady-state inactivation.
        3. A brief test pulse to a fixed `test_pulse_voltage` after each
           conditioning pre-pulse to measure the fraction of available channels.
        4. A final recovery/tail pulse.

        The generated protocol sequence is stored in `self.SwpSeq` and also
        saved as an attribute `self.SwpSeq{self.BsNm}`.

        Args:
            test_voltages (list or np.ndarray, optional): Voltages (mV) for the
                conditioning pre-pulses. Defaults to a range from -120 mV to
                -15 mV in 5 mV steps. Note: these are referred to as `test_voltages`
                in the parameters but function as conditioning potentials.
            holding_potential (float, optional): This parameter's usage in the code
                is nuanced. The initial segment is fixed at -80mV. If `test_voltages`
                is `None`, this `holding_potential` is used as the *single*
                conditioning voltage. Otherwise, `test_voltages` array is used for
                conditioning pulses. The final recovery pulse is also to -80mV.
                Defaults to -120 mV.
            prepulse_duration (float, optional): Duration (ms) of each conditioning
                pre-pulse. Defaults to 2000 ms.
            test_pulse_voltage (float, optional): Voltage (mV) for the test pulse.
                Defaults to 0 mV.
            test_pulse_duration (float, optional): Duration (ms) of the test pulse.
                Defaults to 5 ms.
            recovery_duration (float, optional): Duration (ms) of the final
                recovery/tail pulse. Defaults to 100 ms.
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
        self.CurrVolt()
class AnticonvulsantCTBNMarkovModel(CTBNMarkovModel):
    """Extends CTBNMarkovModel to simulate anticonvulsant drug effects.

    This model incorporates drug binding to different channel states (resting, open,
    inactivated) based on specified drug parameters (affinity, on/off rates).
    It allows simulation of how anticonvulsants modulate channel gating and current.

    Attributes:
        drug_concentration (float): Concentration of the drug (e.g., in µM).
        drug_type (str): Type of drug being simulated (e.g., 'DPH', 'CBZ', 'LTG').
        KI_inactivated (float): Dissociation constant for drug binding to inactivated state (µM).
        KR_resting (float): Dissociation constant for drug binding to resting state (µM).
        k_on_inactivated_base (float): Base on-rate for drug binding to inactivated state.
        k_off (float): Off-rate for drug unbinding.
        # ... (other attributes inherited or specific to drug interactions)
    """
    def __init__(self, drug_concentration=0.0, drug_type='DPH'):
        """Initializes the AnticonvulsantCTBNMarkovModel.

        Sets the drug concentration and type, then calls the superclass initializer
        and initializes drug-specific parameters.

        Args:
            drug_concentration (float, optional): Initial drug concentration (e.g., µM).
                Defaults to 0.0 (no drug).
            drug_type (str, optional): Type of drug. Defaults to 'DPH' (Phenytoin).
        """
        self.NumSwps = 0
        self.num_states = 25
        self.drug_concentration = drug_concentration
        self.drug_type = drug_type.upper()
        self.A = 0
        self.I = 0
        self.D = 0
        self.vm = (- 80)
        self.init_parameters()
        self.init_waves()
        self.update_rates()
        self.CurrVolt()
        self.state_probs_flat = self.EquilOccup(self.vm)
        self.create_default_protocol()
    def set_drug_type(self, drug_type):
        """Sets the type of anticonvulsant drug and re-initializes parameters.

        Args:
            drug_type (str): The new drug type (e.g., 'CBZ', 'LTG').
        """
        self.drug_type = drug_type.upper()
        self.init_parameters()
        self.update_rates()
    def set_drug_concentration(self, drug_concentration):
        """Sets the drug concentration and updates drug-related rates.

        Args:
            drug_concentration (float): The new drug concentration (e.g., µM).
        """
        self.drug_concentration = drug_concentration
        self.update_rates()
    def init_parameters(self):
        """Initializes biophysical and drug-specific parameters.

        This method first calls the superclass's `init_parameters` to set up
        the core ion channel model parameters. It then initializes drug-specific
        parameters based on the `self.drug_type`.

        Drug parameters include:
        - `KI_inactivated` (float): Dissociation constant for drug binding to the
          inactivated state (µM), based on literature values (e.g., Kuo 1998 for
          DPH, CBZ, LTG). This value is stored as `self.KI_inactivated`.
        - `k_off_base` (float): Base off-rate (s^-1) for drug unbinding from any state.
        - `k_off_scaling` (float): A drug-specific scaling factor applied to
          `k_off_base` to derive the effective `self.k_off`. This scaling
          is used to calibrate model predictions for individual drugs.
        - `k_on_inactivated_base` (float): Base on-rate (µM^-1 s^-1) for drug
          binding to the inactivated state. It is calculated internally as
          `self.k_off / self.KI_inactivated`.
        - `KR_resting` (float): Dissociation constant for drug binding to resting
          states (µM), typically set to a much higher value than `KI_inactivated`
          (e.g., `self.KI_inactivated * 1000.0`), reflecting lower affinity for
          resting states.
        - `k_on_resting_base` (float): Base on-rate (µM^-1 s^-1) for drug binding
          to resting states, calculated as `self.k_off / self.KR_resting`.

        The method sets these parameters according to the selected `self.drug_type`
        (e.g., 'DPH', 'CBZ', 'LTG'), using predefined values for each drug.
        After setting these base parameters, it calls `_update_drug_rates` to
        calculate the concentration-dependent effective on-rates.
        """
        self.alcoeff = 20
        self.alslp = 40
        self.btcoeff = 0.3
        self.btslp = 18.5
        self.ConCoeff = 0.004
        self.CoffCoeff = 4.5
        self.ConSlp = 100000000.0
        self.CoffSlp = 100000000.0
        self.gmcoeff = 50
        self.gmslp = 100
        self.dlcoeff = 0.8
        self.dlslp = 6
        self.OpOnCoeff = 4
        self.OpOffCoeff = 0.008
        self.ConHiCoeff = 4
        self.CoffHiCoeff = 0.008
        self.OpOnSlp = 100000000.0
        self.OpOffSlp = 100000000.0
        self.alfac = np.sqrt(np.sqrt((self.ConHiCoeff / self.ConCoeff)))
        self.btfac = np.sqrt(np.sqrt((self.CoffCoeff / self.CoffHiCoeff)))
        self.drug_params = {'CBZ': {'KI_inactivated': 25.0, 'recovery_tau': 189.0, 'k_off_base': (1.0 / 189.0), 'k_off_scaling': 0.55}, 'LTG': {'KI_inactivated': 9.0, 'recovery_tau': 321.0, 'k_off_base': (1.0 / 321.0), 'k_off_scaling': 0.42}, 'DPH': {'KI_inactivated': 9.0, 'recovery_tau': 600.0, 'k_off_base': (1.0 / 600.0), 'k_off_scaling': 0.5}}
        if (self.drug_type in self.drug_params):
            params = self.drug_params[self.drug_type]
        else:
            print(f"Warning: Unknown drug type '{self.drug_type}' (or default 'DPH'), using DPH parameters as fallback.")
            params = self.drug_params['DPH']
        self.KI_inactivated = params['KI_inactivated']
        self.recovery_tau = params['recovery_tau']
        self.k_off_base = params['k_off_base']
        self.k_off = (params['k_off_base'] * params.get('k_off_scaling', 1.0))
        self.KR_resting = (self.KI_inactivated * 1000.0)
        self.k_on_inactivated_base = (self.k_off / self.KI_inactivated)
        self.k_on_resting_base = (self.k_off / self.KR_resting)
        self.numchan = 1
        self.F = 96485
        self.Rgc = 8314
        self.Tkel = 298
        (self.Nao, self.Nai) = (150, 15)
        self.ClipRate = 6000
        self.current_scaling = 0.0125
        self.PNasc = 1e-05
        self._update_drug_rates()
    def _update_drug_rates(self):
        """Updates drug binding and unbinding rates based on current parameters.

        Calculates effective on-rates based on `k_on_inactivated_base`,
        `k_on_resting_base`, and `drug_concentration`. The `k_off` rate is also set.
        The effective dissociation constant for inactivated states (k_off / k_on_inactivated)
        is equivalent to `self.KI_inactivated` (the literature value).
        """
        self.k_on_resting = 0
        self.k_on_inactivated = (self.k_on_inactivated_base * self.drug_concentration)
        self.k_off_resting = 0
        self.k_off_inactivated = self.k_off
    def init_waves(self):
        """Initializes data structures for pre-calculated rates and state probabilities.

        This method first calls the superclass's `init_waves` to initialize
        arrays for voltage-dependent transition rates (e.g., activation,
        deactivation, inactivation) for the drug-free channel states, as well
        as the voltage vector `self.vt` and current vector `self.iscft`.

        It then expands the state space to accommodate drug-bound states by:
        - Setting `self.num_states` to 36 (representing resting, open, and
          inactivated states, each potentially unbound or drug-bound).
        - Re-initializing `self.state_probs_flat` as a zero array of size
          `self.num_states` to store the probabilities of these expanded states.

        Note: This method primarily relies on the superclass to initialize the
        voltage-dependent rate arrays. The effects of the drug on transition
        rates are incorporated within other methods (e.g., `stRatesVolt`,
        `_update_drug_rates`) by modifying how these base rates are used or by
        adding drug binding/unbinding rates, rather than by creating entirely
        separate sets of voltage-dependent rate arrays for drug-bound transitions.
        """
        self.vt = np.arange((- 200), 201)
        self.iscft = np.zeros_like(self.vt)
        self.state_probs_flat = np.zeros(25)
        self.state_probs_flat[0] = 1.0
        num_v = len(self.vt)
        self.fwd_rates_flat = np.zeros((num_v, 20))
        self.bwd_rates_flat = np.zeros((num_v, 20))
        self.inact_on_rates_flat = np.zeros((num_v, 12))
        self.inact_off_rates_flat = np.zeros((num_v, 12))
        self.drug_on_rates_I0 = np.zeros(6)
        self.drug_off_rates_I0 = np.zeros(6)
        self.drug_on_rates_I1 = np.zeros(6)
        self.drug_off_rates_I1 = np.zeros(6)
        self._rate_cache_buffer = {'fwd': np.zeros(20), 'bwd': np.zeros(20), 'inact_on': np.zeros(12), 'inact_off': np.zeros(12)}
        self._last_vidx = (- 1)
        self._state_work_array = np.zeros((4, 6))
        self._deriv_work_array = np.zeros((4, 6))
        self.update_rates()
    def update_rates(self):
        """Updates all voltage-dependent and drug-related rate constants.

        This method first calls the superclass's `update_rates` method, which
        typically recalculates the intrinsic voltage-dependent transition rates
        of the ion channel (e.g., by calling `stRatesVolt` in the superclass).

        After the base channel rates are updated, this method then calls
        `self._update_drug_rates()` to ensure that drug binding and unbinding rates
        (e.g., `self.k_on_inactivated`, `self.k_off_inactivated`) are recalculated
        based on the current drug concentration and other drug-specific parameters.
        This ensures that all rates influencing channel and drug-channel complex
        dynamics are consistent with the current model state.
        """
        self.stRatesVolt()
    def stRatesVolt(self):
        """Calculates and stores state transition rates as a function of voltage, including drug effects.

        This method overrides the superclass's `stRatesVolt` to incorporate
        drug binding and unbinding kinetics into the overall state transition
        rate calculations. It defines the transition rates for an expanded state model
        (typically 36 states if considering resting, open, and inactivated drug binding)
        that includes unbound (drug-free) states and drug-bound states.

        The method calculates:
        1. Intrinsic voltage-dependent transition rates (activation, deactivation,
           inactivation, recovery from inactivation) for the unbound channel,
           by calling `super().stRatesVolt()`.
        2. It then populates or modifies rate arrays (e.g., `self.fwd_rates_I0`,
           `self.bwd_rates_I0`, etc.) to include transitions representing:
           - Drug binding (on-rates) to different channel conformations. These rates
             depend on `self.drug_concentration` and base on-rates like
             `self.k_on_inactivated` (derived from `self.k_on_inactivated_base`).
           - Drug unbinding (off-rates) from different channel conformations,
             using rates like `self.k_off_inactivated` (derived from `self.k_off`).

        The specific implementation details involve mapping these drug-related
        transitions onto the expanded state space. Helper functions like `act_idx`
        and `inact_idx` (defined as inner functions within this method) are often
        used to correctly index into the rate arrays, considering both the
        channel's conformational state (e.g., C1-C5, O, I) and its drug-bound status.

        The rates are calculated across the pre-defined voltage range `self.vt`.
        The method ensures that the `fwd_rates_I0`, `bwd_rates_I0`, `fwd_rates_I1`,
        `bwd_rates_I1`, `inact_on_rates`, and `inact_off_rates` arrays reflect
        all transitions in the drug-affected model.
        """
        vt = self.vt
        amt = (self.alcoeff * np.exp((vt / self.alslp)))
        bmt = (self.btcoeff * np.exp(((- vt) / self.btslp)))
        gmt = (self.gmcoeff * np.exp((vt / self.gmslp)))
        dmt = (self.dlcoeff * np.exp(((- vt) / self.dlslp)))
        konlo = (self.ConCoeff * np.exp((vt / self.ConSlp)))
        kofflo = (self.CoffCoeff * np.exp(((- vt) / self.CoffSlp)))
        konop = (self.OpOnCoeff * np.exp((vt / self.OpOnSlp)))
        koffop = (self.OpOffCoeff * np.exp(((- vt) / self.OpOffSlp)))
        def act_idx(i, d, a):
            return ((((i * 2) + d) * 5) + a)
        for a in range(4):
            self.fwd_rates_flat[:, act_idx(0, 0, a)] = np.minimum(((4 - a) * amt), self.ClipRate)
            self.fwd_rates_flat[:, act_idx(0, 1, a)] = np.minimum(((4 - a) * amt), self.ClipRate)
            self.fwd_rates_flat[:, act_idx(1, 0, a)] = np.minimum((((4 - a) * amt) * self.alfac), self.ClipRate)
            self.fwd_rates_flat[:, act_idx(1, 1, a)] = np.minimum((((4 - a) * amt) * self.alfac), self.ClipRate)
        for i in range(2):
            for d in range(2):
                self.fwd_rates_flat[:, act_idx(i, d, 4)] = np.minimum(gmt, self.ClipRate)
        for a in range(4):
            rate_I0 = np.minimum(((a + 1) * bmt), self.ClipRate)
            rate_I1 = np.minimum((((a + 1) * bmt) / self.btfac), self.ClipRate)
            self.bwd_rates_flat[:, act_idx(0, 0, a)] = rate_I0
            self.bwd_rates_flat[:, act_idx(0, 1, a)] = rate_I0
            self.bwd_rates_flat[:, act_idx(1, 0, a)] = rate_I1
            self.bwd_rates_flat[:, act_idx(1, 1, a)] = rate_I1
        for i in range(2):
            for d in range(2):
                self.bwd_rates_flat[:, act_idx(i, d, 4)] = np.minimum(dmt, self.ClipRate)
        def inact_idx(d, a):
            return ((d * 6) + a)
        alfac_powers = (self.alfac ** np.arange(5))
        btfac_powers = (self.btfac ** np.arange(5))
        for d in range(2):
            for a in range(5):
                self.inact_on_rates_flat[:, inact_idx(d, a)] = np.minimum((konlo * alfac_powers[a]), self.ClipRate)
                self.inact_off_rates_flat[:, inact_idx(d, a)] = np.minimum((kofflo / btfac_powers[a]), self.ClipRate)
            self.inact_on_rates_flat[:, inact_idx(d, 5)] = np.minimum(konop, self.ClipRate)
            self.inact_off_rates_flat[:, inact_idx(d, 5)] = np.minimum(koffop, self.ClipRate)
        self._update_drug_rates()
        self.drug_on_rates_I0[:] = self.k_on_resting
        self.drug_off_rates_I0[:] = self.k_off_resting
        self.drug_on_rates_I1[:] = self.k_on_inactivated
        self.drug_off_rates_I1[:] = self.k_off_inactivated
    def CurrVolt(self):
        """Calculates the ionic current, considering drug effects on channel availability.

        This method overrides the superclass's `CurrVolt` to adjust the
        calculation of total ionic current based on the presence of a drug.
        It typically calls the superclass's `CurrVolt` method to get the
        current-voltage relationship for a drug-free channel population or
        calculates it based on the GHK equation.

        The key modification in this overridden version is to account for the
        fact that drug-bound channels may not conduct current, or may conduct
        differently. The calculation of `self.iscft` (current as a function of
        test voltage) is adjusted based on the proportion of channels that are
        in a conducting, drug-free state.

        Specifically, it considers the open probability of only the unbound channels
        (e.g., state O0, not OD_O or similar drug-bound open states if they are
        assumed non-conducting or have altered conductance). The GHK equation or
        a similar formulation is then applied using this adjusted open probability
        and the channel's permeability (`self.PNasc`).

        The resulting `self.iscft` array stores the net ionic current across
        the membrane for the range of test voltages `self.vt`, reflecting the
        impact of the drug on the total available conducting channels.
        """
        scaled_PNasc = self.PNasc
        v_volts = (self.vt * 0.001)
        near_zero = (np.abs(v_volts) < 1e-06)
        not_zero = (~ near_zero)
        self.iscft = np.zeros_like(v_volts)
        if np.any(near_zero):
            du2_zero = ((self.F * self.F) / (self.Rgc * self.Tkel))
            self.iscft[near_zero] = ((scaled_PNasc * du2_zero) * (self.Nai - self.Nao))
        if np.any(not_zero):
            v_nz = v_volts[not_zero]
            du1 = ((v_nz * self.F) / (self.Rgc * self.Tkel))
            du3 = np.exp((- du1))
            du5_corrected = (((self.F * du1) * (self.Nai - (self.Nao * du3))) / (1 - du3))
            self.iscft[not_zero] = (scaled_PNasc * du5_corrected)
    def EquilOccup(self, vm):
        """Calculates equilibrium state occupancies, including drug-bound states.

        This method overrides the superclass's `EquilOccup` to determine the
        steady-state probabilities of all states in the expanded model that
        includes drug binding. It considers the intrinsic channel transition rates
        (activation, inactivation) and the drug binding/unbinding rates at a
        given membrane potential `vm`.

        The calculation involves:
        1. Retrieving the voltage-dependent intrinsic channel rates and drug-related
           rates (e.g., `k_on_inactivated`, `k_off_inactivated`) at the specified `vm`.
           This often involves calling `_get_rates_at_vm` (from superclass, for intrinsic rates)
           and ensuring drug rates are current (e.g., via `_update_drug_rates` if `vm` changed).
        2. Constructing the full transition rate matrix (Q matrix) for the expanded
           state model (e.g., 36 states for this model). This matrix includes all transitions:
           - Intrinsic channel gating (e.g., C1 <-> C2, C5 <-> O, O <-> I).
           - Drug binding to various channel conformations (e.g., I + D <-> ID_I).
           - Drug unbinding from various channel conformations (e.g., ID_I <-> I + D).
        3. Solving the system of linear equations Q * p = 0, subject to the
           constraint that the sum of probabilities sum(p) = 1, to find the
           steady-state probability vector `p`. This is often done by finding the
           null space of Q or by solving a related linear system.

        Helper functions like `act_idx` and `inact_idx` (often defined as inner
        functions) are typically used to map states and transitions correctly
        onto the indices of the Q matrix and the resulting probability vector.
        The method returns a flattened array of these equilibrium state probabilities.

        Args:
            vm (float): The membrane potential (mV) at which to calculate
                        equilibrium occupancies.

        Returns:
            np.ndarray: A flattened array of equilibrium state probabilities for
                        all (e.g., 36) states in the drug-affected model.
        """
        self.vm = vm
        self.update_rates()
        vidx = np.argmin(np.abs((self.vt - vm)))
        def safe_div(a, b, default=0.0):
            if np.isscalar(b):
                return ((a / b) if (abs(b) > 1e-10) else default)
            else:
                result = np.full_like(a, default, dtype=float)
                mask = (np.abs(b) > 1e-10)
                if np.any(mask):
                    result[mask] = (a[mask] / b[mask])
                return result
        def act_idx(i, d, a):
            return ((((i * 2) + d) * 5) + a)
        def inact_idx(d, a):
            return ((d * 6) + a)
        fwd_I0D0 = np.array([self.fwd_rates_flat[(vidx, act_idx(0, 0, a))] for a in range(5)])
        bwd_I0D0 = np.array([self.bwd_rates_flat[(vidx, act_idx(0, 0, a))] for a in range(5)])
        fwd_I1D0 = np.array([self.fwd_rates_flat[(vidx, act_idx(1, 0, a))] for a in range(5)])
        bwd_I1D0 = np.array([self.bwd_rates_flat[(vidx, act_idx(1, 0, a))] for a in range(5)])
        fwd_I0D1 = np.array([self.fwd_rates_flat[(vidx, act_idx(0, 1, a))] for a in range(5)])
        bwd_I0D1 = np.array([self.bwd_rates_flat[(vidx, act_idx(0, 1, a))] for a in range(5)])
        fwd_I1D1 = np.array([self.fwd_rates_flat[(vidx, act_idx(1, 1, a))] for a in range(5)])
        bwd_I1D1 = np.array([self.bwd_rates_flat[(vidx, act_idx(1, 1, a))] for a in range(5)])
        inact_on_D0 = np.array([self.inact_on_rates_flat[(vidx, inact_idx(0, a))] for a in range(6)])
        inact_off_D0 = np.array([self.inact_off_rates_flat[(vidx, inact_idx(0, a))] for a in range(6)])
        inact_on_D1 = np.array([self.inact_on_rates_flat[(vidx, inact_idx(1, a))] for a in range(6)])
        inact_off_D1 = np.array([self.inact_off_rates_flat[(vidx, inact_idx(1, a))] for a in range(6)])
        rel_prob_A_I0D0 = np.ones(6)
        rel_prob_A_I0D0[1:] = np.cumprod(safe_div(fwd_I0D0, bwd_I0D0, 1.0))
        rel_prob_A_I0D0 /= rel_prob_A_I0D0.sum()
        rel_prob_A_I1D0 = np.ones(6)
        rel_prob_A_I1D0[1:] = np.cumprod(safe_div(fwd_I1D0, bwd_I1D0, 1.0))
        rel_prob_A_I1D0 /= rel_prob_A_I1D0.sum()
        rel_prob_A_I0D1 = np.ones(6)
        rel_prob_A_I0D1[1:] = np.cumprod(safe_div(fwd_I0D1, bwd_I0D1, 1.0))
        rel_prob_A_I0D1 /= rel_prob_A_I0D1.sum()
        rel_prob_A_I1D1 = np.ones(6)
        rel_prob_A_I1D1[1:] = np.cumprod(safe_div(fwd_I1D1, bwd_I1D1, 1.0))
        rel_prob_A_I1D1 /= rel_prob_A_I1D1.sum()
        total_rate_I0_to_I1_D0 = np.dot(rel_prob_A_I0D0, inact_on_D0)
        total_rate_I1_to_I0_D0 = np.dot(rel_prob_A_I1D0, inact_off_D0)
        rel_prob_I1_D0 = safe_div(total_rate_I0_to_I1_D0, total_rate_I1_to_I0_D0, 0.0)
        total_rate_I0_to_I1_D1 = np.dot(rel_prob_A_I0D1, inact_on_D1)
        total_rate_I1_to_I0_D1 = np.dot(rel_prob_A_I1D1, inact_off_D1)
        rel_prob_I1_D1 = safe_div(total_rate_I0_to_I1_D1, total_rate_I1_to_I0_D1, 0.0)
        drug_factor_I0 = safe_div(self.k_on_resting, self.k_off_resting, 0.0)
        drug_factor_I1 = safe_div(self.k_on_inactivated, self.k_off_inactivated, 0.0)
        prob_I0D0_unnorm = 1.0
        prob_I1D0_unnorm = rel_prob_I1_D0
        prob_I0D1_unnorm = drug_factor_I0
        prob_I1D1_unnorm = (drug_factor_I1 * rel_prob_I1_D1)
        total_unnorm = (((prob_I0D0_unnorm + prob_I1D0_unnorm) + prob_I0D1_unnorm) + prob_I1D1_unnorm)
        if (total_unnorm > 1e-10):
            prob_I0D0 = (prob_I0D0_unnorm / total_unnorm)
            prob_I1D0 = (prob_I1D0_unnorm / total_unnorm)
            prob_I0D1 = (prob_I0D1_unnorm / total_unnorm)
            prob_I1D1 = (prob_I1D1_unnorm / total_unnorm)
        else:
            prob_I0D0 = 1.0
            prob_I1D0 = prob_I0D1 = prob_I1D1 = 0.0
        eq_probs = np.zeros(25)
        eq_probs[0:6] = (rel_prob_A_I0D0 * prob_I0D0)
        eq_probs[6:12] = (rel_prob_A_I1D0 * prob_I1D0)
        eq_probs[12:18] = (rel_prob_A_I0D1 * prob_I0D1)
        eq_probs[18:24] = (rel_prob_A_I1D1 * prob_I1D1)
        total_prob = np.sum(eq_probs[:24])
        if (total_prob > 1e-10):
            eq_probs[:24] /= total_prob
        else:
            eq_probs[:] = 0.0
            eq_probs[0] = 1.0
        eq_probs = np.nan_to_num(eq_probs, nan=0.0)
        self.state_probs_flat[:] = eq_probs[:]
        self.pop = np.zeros(25)
        self.pop[:] = eq_probs[:]
        return self.pop
    def NowDerivs(self, t, y):
        """Calculates the time derivatives of state probabilities (dp/dt).

        This method overrides the superclass's `NowDerivs` to compute the
        derivatives of the state probability vector `y` (representing state probabilities `p`)
        with respect to time `t`. The membrane potential `vm` is typically accessed
        via `self.vm`, which should be set prior to calling the ODE solver.
        This is the core function used by ODE solvers for simulating the model's
        dynamics.

        The calculation involves:
        1. Retrieving the voltage-dependent intrinsic channel rates and drug-related
           rates at `self.vm`. This may involve calling `_get_rates_at_vm` (from
           superclass, for intrinsic rates) and ensuring drug rates are current.
           A rate cache (`self._rate_cache_buffer`) is often used to avoid redundant
           recalculations if `self.vm` hasn't changed since the last call.
        2. Constructing the full transition rate matrix (Q matrix) for the expanded
           state model (e.g., 36 states for this model). This Q matrix is identical
           to the one used in `EquilOccup` for the given `self.vm`.
        3. Calculating the derivatives `dy/dt = Q * y`. The input `y` is a
           flattened array of current state probabilities, and the method returns
           a flattened array of their time derivatives.

        Helper functions like `act_idx` and `inact_idx` (often defined as inner
        functions) are typically used to map states and transitions correctly
        when constructing the Q matrix and performing the matrix-vector multiplication.

        The method is designed to be compatible with ODE solver interfaces (e.g.,
        those in `scipy.integrate`), which expect a function that takes `y` (state vector)
        and `t` (time) and returns `dy/dt`.

        Args:
            t (float): The current time point (may be used by the ODE solver,
                       but often not directly in rate calculations if `self.vm` is fixed).
            y (np.ndarray): A flattened array of current state probabilities for
                            all (e.g., 36) states in the drug-affected model.

        Returns:
            np.ndarray: A flattened array of the time derivatives (dy/dt) for
                        each state.
        """
        if (np.any(np.isnan(y)) or np.any(np.isinf(y))):
            return np.zeros_like(y)
        if ((not hasattr(self, '_voltage_lut_cache')) or (self._voltage_lut_cache[0] != self.vm)):
            vidx = np.searchsorted(self.vt, self.vm)
            vidx = min(max(vidx, 0), (len(self.vt) - 1))
            self._voltage_lut_cache = (self.vm, vidx)
        else:
            vidx = self._voltage_lut_cache[1]
        if (vidx != self._last_vidx):
            self._rate_cache_buffer['fwd'][:] = self.fwd_rates_flat[vidx, :]
            self._rate_cache_buffer['bwd'][:] = self.bwd_rates_flat[vidx, :]
            self._rate_cache_buffer['inact_on'][:] = self.inact_on_rates_flat[vidx, :]
            self._rate_cache_buffer['inact_off'][:] = self.inact_off_rates_flat[vidx, :]
            self._last_vidx = vidx
        fwd_rates = self._rate_cache_buffer['fwd']
        bwd_rates = self._rate_cache_buffer['bwd']
        inact_on = self._rate_cache_buffer['inact_on']
        inact_off = self._rate_cache_buffer['inact_off']
        self._state_work_array[0, :] = y[0:6]
        self._state_work_array[1, :] = y[12:18]
        self._state_work_array[2, :] = y[6:12]
        self._state_work_array[3, :] = y[18:24]
        self._deriv_work_array[:] = 0.0
        for combo_idx in range(4):
            rate_start = (combo_idx * 5)
            probs = self._state_work_array[combo_idx, :]
            deriv = self._deriv_work_array[combo_idx, :]
            fwd_flux = (fwd_rates[rate_start:(rate_start + 5)] * probs[:5])
            deriv[:5] -= fwd_flux
            deriv[1:6] += fwd_flux
            bwd_flux = (bwd_rates[rate_start:(rate_start + 5)] * probs[1:6])
            deriv[1:6] -= bwd_flux
            deriv[:5] += bwd_flux
        for d in range(2):
            i0_idx = (0 if (d == 0) else 1)
            i1_idx = (2 if (d == 0) else 3)
            rate_idx = slice((d * 6), ((d + 1) * 6))
            inact_flux = (inact_on[rate_idx] * self._state_work_array[i0_idx, :])
            self._deriv_work_array[i0_idx, :] -= inact_flux
            self._deriv_work_array[i1_idx, :] += inact_flux
            recov_flux = (inact_off[rate_idx] * self._state_work_array[i1_idx, :])
            self._deriv_work_array[i1_idx, :] -= recov_flux
            self._deriv_work_array[i0_idx, :] += recov_flux
        drug_flux_I0 = ((self.drug_on_rates_I0 * self._state_work_array[0, :]) - (self.drug_off_rates_I0 * self._state_work_array[1, :]))
        self._deriv_work_array[0, :] -= drug_flux_I0
        self._deriv_work_array[1, :] += drug_flux_I0
        drug_flux_I1 = ((self.drug_on_rates_I1 * self._state_work_array[2, :]) - (self.drug_off_rates_I1 * self._state_work_array[3, :]))
        self._deriv_work_array[2, :] -= drug_flux_I1
        self._deriv_work_array[3, :] += drug_flux_I1
        dstdt = np.zeros_like(y)
        dstdt[0:6] = self._deriv_work_array[0, :]
        dstdt[6:12] = self._deriv_work_array[2, :]
        dstdt[12:18] = self._deriv_work_array[1, :]
        dstdt[18:24] = self._deriv_work_array[3, :]
        return dstdt
    def _get_rates_at_vm(self, vm):
        """Retrieves pre-calculated transition rates for a given membrane potential.

        This method fetches all relevant transition rates (both intrinsic
        voltage-dependent channel rates and drug interaction rates) for a
        specified membrane potential `vm`. It uses `np.searchsorted` to find
        the closest pre-calculated voltage index in `self.vt` and returns
        the corresponding rates from the cached arrays.

        The returned rates include:
        - `fwd_flat`: Forward rates for intrinsic channel transitions (e.g., C1->C2).
        - `bwd_flat`: Backward rates for intrinsic channel transitions (e.g., C2->C1).
        - `inact_on_flat`: Rates for transitions into the inactivated state (e.g., O->I).
        - `inact_off_flat`: Rates for transitions out of the inactivated state (e.g., I->O).
        - `drug_on_I0`: Drug on-rates for channels in resting/closed states.
        - `drug_off_I0`: Drug off-rates for channels in resting/closed states.
        - `drug_on_I1`: Drug on-rates for channels in inactivated states.
        - `drug_off_I1`: Drug off-rates for channels in inactivated states.

        These rates are typically pre-calculated and stored by the `stRatesVolt`
        method. This method provides a convenient way to access these rates
        for a specific `vm` during simulations (e.g., within `NowDerivs` or
        `EquilOccup`).

        Args:
            vm (float): The membrane potential (mV) for which to retrieve rates.

        Returns:
            dict: A dictionary containing flat arrays of forward, backward,
                  inactivation on/off, and drug on/off rates corresponding
                  to the specified `vm`.
        """
        vidx = np.searchsorted(self.vt, vm)
        vidx = np.clip(vidx, 0, (len(self.vt) - 1))
        return {'fwd_flat': self.fwd_rates_flat[vidx, :], 'bwd_flat': self.bwd_rates_flat[vidx, :], 'inact_on_flat': self.inact_on_rates_flat[vidx, :], 'inact_off_flat': self.inact_off_rates_flat[vidx, :], 'drug_on_I0': self.drug_on_rates_I0, 'drug_off_I0': self.drug_off_rates_I0, 'drug_on_I1': self.drug_on_rates_I1, 'drug_off_I1': self.drug_off_rates_I1}
    def _update_scalar_rates(self):
        """Updates scalar rate attributes based on the current `self.vm`.

        This method is responsible for populating scalar rate attributes
        (e.g., `self.alpha_m`, `self.beta_m`, `self.k_on_inactivated_vm`, etc.)
        with values corresponding to the current membrane potential `self.vm`.
        It calls `self._get_rates_at_vm(self.vm)` to fetch a dictionary of
        rate arrays and then extracts the specific scalar rates from these arrays
        or directly from attributes like `self.k_on_inactivated`.

        This is often used to make individual rates at the current `vm` easily
        accessible as direct attributes of the model instance, for example,
        when constructing the Q-matrix in `NowDerivs` or `EquilOccup` without
        needing to pass around the full dictionary of rates.

        It includes checks to ensure `self.vm` is set and that essential
        rate arrays (like `self.fwd_rates_flat`) are initialized, printing
        warnings if not.

        The scalar rates updated typically include:
        - Intrinsic forward rates (e.g., `self.am1` to `self.am5`).
        - Intrinsic backward rates (e.g., `self.bm1` to `self.bm5`).
        - Inactivation on/off rates (e.g., `self.ai0`, `self.bi0`).
        - Drug on/off rates for resting states (e.g., `self.k_on_resting_vm`, `self.k_off_resting_vm`).
        - Drug on/off rates for inactivated states (e.g., `self.k_on_inactivated_vm`, `self.k_off_inactivated_vm`).
        """
        if (not hasattr(self, 'vm')):
            print(((f'Warning: AnticonvulsantCTBNMarkovModel instance (id: {id(self)}) ' + "does not have 'vm' attribute when _update_scalar_rates is called. ") + 'Rates cannot be updated.'))
            return
        if ((not hasattr(self, 'vt')) or (not hasattr(self, 'fwd_rates_flat'))):
            print(((f'Warning: AnticonvulsantCTBNMarkovModel instance (id: {id(self)}) may not be fully initialized ' + '(missing self.vt or vectorized rate arrays like self.fwd_rates_flat) ') + 'when _update_scalar_rates is called. Proceeding, but _get_rates_at_vm might fail.'))
        try:
            rates_at_vm_dict = self._get_rates_at_vm(self.vm)
            for (rate_name, rate_value_array) in rates_at_vm_dict.items():
                setattr(self, rate_name, rate_value_array)
        except AttributeError as e:
            print((f'Error in AnticonvulsantCTBNMarkovModel._update_scalar_rates for vm={self.vm}: ' + f'Failed to get or set rates. Underlying error: {e}'))
    def Sweep(self, SwpNo):
        """Runs a single voltage clamp sweep and stores results, considering drug effects.

        This method overrides the superclass's `Sweep` to simulate the model's
        response to a predefined voltage protocol (identified by `SwpNo`),
        while accounting for the presence and kinetics of an anticonvulsant drug.

        Key operations typically include:
        1. Setting the initial state probabilities: This might involve using
           equilibrium occupancies at a holding potential, considering drug binding.
        2. Iterating through the voltage steps of the specified protocol:
           - For each voltage step, `self.vm` is updated.
           - `self.update_rates()` is called to ensure all intrinsic and drug-related
             rates are current for the new `vm`. This is crucial as it calls
             `self.stRatesVolt()` (which updates voltage-dependent rates) and
             `self._update_drug_rates()` (which updates drug on/off rates based
             on concentration and potentially other factors like `k_on_base` values).
           - The system of ODEs (`self.NowDerivs`) is solved for the duration
             of the voltage step to get the time evolution of state probabilities.
        3. Storing results: Time-dependent state probabilities, currents, and other
           relevant data are stored, often using helper methods like
           `_store_ctbn_results_vectorized` or `_store_ctbn_results`.

        This overridden version ensures that the `update_rates` call correctly
        propagates drug effects into the rate constants used by `NowDerivs`
        throughout the simulation of the sweep.

        Args:
            SwpNo (int): The index of the sweep protocol to run, corresponding
                         to an entry in `self.SwpSeq` or `self.SwpSeq{self.BsNm}`.

        Returns:
            tuple: Typically returns (current_trace, time_vector, state_probabilities_trace)
                   for the simulated sweep, though the exact return can vary.
                   Consult the superclass or specific implementation for details.
                   (Note: The base class `Sweep` returns (I, t, StSwp), so this
                   is likely similar).
        """
        if ((SwpNo >= self.NumSwps) or (SwpNo < 0)):
            raise ValueError(f'Invalid sweep number {SwpNo}')
        NumEpchs = int(self.SwpSeq[(0, SwpNo)])
        if (NumEpchs <= 0):
            raise ValueError('Invalid number of epochs in protocol')
        total_points = (int(self.SwpSeq[(((2 * NumEpchs) + 1), SwpNo)]) + 1)
        sampint = 0.005
        self.SimSwp = np.zeros(total_points)
        self.SimOp = np.zeros(total_points)
        self.SimIn = np.zeros(total_points)
        self.SimAv = np.zeros(total_points)
        self.SimCom = np.zeros(total_points)
        self.SimDrugBound = np.zeros(total_points)
        self.state_probs_flat = np.zeros(25)
        self.state_probs_flat[0] = 1.0
        epoch_voltages = np.zeros((NumEpchs + 1))
        epoch_end_times = np.zeros((NumEpchs + 1))
        for e in range((NumEpchs + 1)):
            if (e == 0):
                epoch_voltages[e] = self.SwpSeq[(2, SwpNo)]
                epoch_end_times[e] = 0.0
            else:
                epoch_voltages[e] = self.SwpSeq[((2 * e), SwpNo)]
                epoch_end_times[e] = (int(self.SwpSeq[(((2 * e) + 1), SwpNo)]) * sampint)
        self.vm = epoch_voltages[0]
        self.CurrVolt()
        eq_pop = self.EquilOccup(self.vm)
        self.state_probs_flat[:] = eq_pop[:]
        self.pop = np.zeros(25)
        self.pop[:] = eq_pop[:]
        self._store_ctbn_results(0, 0)
        if ((not hasattr(self, '_reusable_y0')) or (len(self._reusable_y0) < 24)):
            self._reusable_y0 = np.zeros(24)
        current_time = 0.0
        store_idx = 1
        for epoch in range(1, (NumEpchs + 1)):
            self.vm = epoch_voltages[epoch]
            epoch_end_time = epoch_end_times[epoch]
            self.update_rates()
            self.CurrVolt()
            num_points = max(2, (int(((epoch_end_time - current_time) / sampint)) + 1))
            t_eval = np.linspace(current_time, epoch_end_time, num_points)
            if (len(t_eval) <= 1):
                current_time = epoch_end_time
                continue
            self._reusable_y0[:] = self.state_probs_flat[:24]
            sol = solve_ivp(self.NowDerivs, [current_time, epoch_end_time], self._reusable_y0, method='LSODA', t_eval=t_eval, rtol=1e-06, atol=1e-08)
            if hasattr(self, 'full_sol_t'):
                self.full_sol_t = sol.t
                self.full_sol_y = sol.y
            batch_size = len(sol.t)
            if (batch_size > 0):
                end_idx = min((store_idx + batch_size), total_points)
                batch_indices = np.arange(store_idx, end_idx)
                actual_batch_size = len(batch_indices)
                if (actual_batch_size > 0):
                    states_subset = sol.y[:, :actual_batch_size]
                    batch_states = states_subset.T
                    batch_voltages = np.full(actual_batch_size, self.vm)
                    self._store_ctbn_results_vectorized(batch_indices, batch_states, batch_voltages)
                    self.state_probs_flat[:24] = sol.y[:, (- 1)]
                    self.state_probs_flat[24] = 0.0
                    self.pop[:24] = sol.y[:, (- 1)]
                    self.pop[24] = 0.0
                    store_idx = end_idx
            current_time = epoch_end_time
        self.time = np.arange(0, (total_points * sampint), sampint)[:total_points]
        return (sol.t, self.SimSwp)
    def _store_ctbn_results(self, idx, t):
        """Stores simulation results for a single time point by wrapping the vectorized version.

        This method is an override of the superclass's `_store_ctbn_results`.
        It adapts the single time point data (`idx`, `t`, and implicitly `self.state_probs_flat`, `self.vm`)
        to the batch-oriented interface of `_store_ctbn_results_vectorized`.

        It packages the current flat state probabilities (`self.state_probs_flat[:24]`,
        representing the 24 states of the unbound and drug-bound channel without
        considering open-drug-bound states if they are modeled separately or not at all)
        and the current voltage (`self.vm`) into numpy arrays suitable for the
        vectorized storage method.

        Args:
            idx (int): The index in the simulation output arrays where the current
                       results should be stored.
            t (float): The current simulation time. (Note: `t` is not explicitly
                       used in the call to the vectorized version in this implementation,
                       as time is typically handled by the `Sweep` method's main loop or
                       the vectorized method itself if it stores time).
        """
        self._store_ctbn_results_vectorized([idx], np.array([self.state_probs_flat[:24]]), np.array([self.vm]))
    def _store_ctbn_results_vectorized(self, indices, state_probs_batch, voltages):
        """Stores simulation results for a batch of time points, accounting for drug states.

        This method overrides the superclass's `_store_ctbn_results_vectorized`
        to handle the storage of simulation results (state probabilities, currents)
        from the expanded state model that includes drug-bound states.

        Key operations include:
        1. Processing a batch of simulation data points, provided as `indices`,
           `state_probs_batch` (containing probabilities for all, e.g., 36 states),
           and corresponding `voltages`.
        2. Calculating and storing the total ionic current for each data point.
           This involves:
           - Summing the probabilities of the unbound open state (O0) from `state_probs_batch`.
             (Assuming drug-bound open states, if they exist, are non-conducting or
             their conductance is handled differently).
           - Using the GHK equation or a similar formulation with the channel's
             permeability (`self.PNasc`) and the open probability of unbound channels
             to calculate current at the given `voltages`.
        3. Storing the calculated currents in `self.SimSwp['I']` at the specified `indices`.
        4. Storing the full state probability vectors (e.g., all 36 state probabilities)
           from `state_probs_batch` into `self.SimSwp['St']` at the specified `indices`.
           This ensures that the occupancy of drug-bound states is also recorded.
        5. Storing other relevant data like open probabilities for unbound channels
           (`self.SimSwp['Po']`) and potentially summed probabilities for different
           categories of states (e.g., all resting, all inactivated, for both unbound
           and drug-bound).

        This overridden version ensures that calculations (like current) and storage
        correctly reflect the expanded state space and the assumptions about which
        states (particularly drug-bound states) contribute to the macroscopic current.

        Args:
            indices (list or np.ndarray): A list or array of integer indices
                indicating where in the output arrays the batch data should be stored.
            state_probs_batch (np.ndarray): A 2D array where each row corresponds
                to a time point (matching `indices`) and columns are the probabilities
                of each state in the expanded (e.g., 36-state) model.
            voltages (np.ndarray): A 1D array of membrane potentials corresponding
                to each time point in the batch.
        """
        if (len(indices) == 0):
            return
        if np.isscalar(voltages):
            voltage_indices = np.searchsorted(self.vt, voltages)
            voltage_indices = np.clip(voltage_indices, 0, (len(self.vt) - 1))
            current_factors = self.iscft[voltage_indices]
        else:
            voltage_indices = np.searchsorted(self.vt, voltages)
            voltage_indices = np.clip(voltage_indices, 0, (len(self.vt) - 1))
            current_factors = self.iscft[voltage_indices]
        conducting_open_probs = state_probs_batch[:, 5]
        total_open_probs = (state_probs_batch[:, 5] + state_probs_batch[:, 17])
        scale_factor = (self.numchan * self.current_scaling)
        currents = ((conducting_open_probs * current_factors) * scale_factor)
        inactivated = (np.sum(state_probs_batch[:, 6:12], axis=1) + np.sum(state_probs_batch[:, 18:24], axis=1))
        available = np.sum(state_probs_batch[:, 0:6], axis=1)
        drug_bound = (np.sum(state_probs_batch[:, 12:18], axis=1) + np.sum(state_probs_batch[:, 18:24], axis=1))
        self.SimSwp[indices] = currents
        self.SimOp[indices] = total_open_probs
        self.SimIn[indices] = inactivated
        self.SimAv[indices] = available
        self.SimCom[indices] = voltages
        self.SimDrugBound[indices] = drug_bound
    def create_default_protocol(self, target_voltages=None, holding_potential=(- 80), holding_duration=98, test_duration=200, tail_duration=2):
        """Creates a default multi-step voltage clamp protocol.

        This method sets up a standard voltage clamp protocol consisting of a
        holding period, a series of test pulses to different target voltages,
        and a final tail pulse. The protocol parameters (voltages, durations)
        can be customized.

        The generated protocol sequence is stored in `self.SwpSeq` and also
        cached under an attribute named `SwpSeq{self.BsNm}`, where `self.BsNm`
        is set to 'MultiStepKeyVoltages'.

        This method may be an override of a superclass method. In the context
        of `AnticonvulsantCTBNMarkovModel`, it's important to note that this
        method concludes by calling `self.CurrVolt()`. Since `CurrVolt` is
        overridden in this subclass to account for drug effects on channel
        availability and current, calling it here ensures that the initial
        current calculation based on the protocol's starting conditions
        reflects the drug's presence.

        Args:
            target_voltages (list or np.ndarray, optional): A list of voltages
                for the test pulses. Defaults to `[30, 0, -20, -30, -40, -50, -60]` mV.
            holding_potential (float, optional): The voltage during the initial
                holding period and the final tail pulse. Defaults to -80 mV.
            holding_duration (float, optional): Duration of the initial holding
                period in ms. Defaults to 98 ms.
            test_duration (float, optional): Duration of each test pulse in ms.
                Defaults to 200 ms.
            tail_duration (float, optional): Duration of the final tail pulse
                in ms. Defaults to 2 ms.
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
        self.CurrVolt()
    def create_inactivation_protocol(self, inactivating_voltage=(- 20), test_voltage=0, inactivating_duration=2000, recovery_duration=100):
        """Creates a voltage clamp protocol to assess inactivation characteristics.

        This method sets up a protocol typically used to study voltage-dependent
        inactivation or use-dependent block. It involves a holding potential,
        a conditioning pulse to an inactivating voltage, a brief test pulse,
        and a final recovery period.

        The generated protocol sequence is stored in `self.SwpSeq` and also
        cached under an attribute named `SwpSeq{self.BsNm}`, where `self.BsNm`
        is set to 'InactivationProtocol'. This protocol consists of a single sweep.

        This method may be an override of a superclass method. In the context
        of `AnticonvulsantCTBNMarkovModel`, the concluding call to `self.CurrVolt()`
        is significant. Since `CurrVolt` is overridden in this subclass to
        incorporate drug effects on channel availability and current, this call
        ensures that any initial current calculation based on the protocol's
        starting conditions correctly reflects the drug's presence.

        Args:
            inactivating_voltage (float, optional): The voltage of the
                conditioning pulse used to induce inactivation. Defaults to -20 mV.
            test_voltage (float, optional): The voltage of the brief test pulse
                following the inactivating pulse. Defaults to 0 mV.
            inactivating_duration (float, optional): Duration of the conditioning
                inactivating pulse in ms. Defaults to 2000 ms.
            recovery_duration (float, optional): Duration of the final recovery
                period at the holding potential in ms. Defaults to 100 ms.
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
        self.CurrVolt()
    def create_recovery_protocol(self, target_recovery_times=None, holding_potential=(- 80), inactivating_voltage=(- 20), test_voltage=0, holding_duration=200, inactivating_duration=2000, test_duration=20, tail_duration=100):
        """Creates a protocol to measure recovery from inactivation.

        This method generates a series of voltage clamp sweeps designed to
        assess the time course of recovery from inactivation. Each sweep typically
        involves a holding potential, a conditioning pulse to induce inactivation,
        a variable recovery interval at the holding potential, a test pulse
        to measure available current, and a final tail pulse.

        The generated protocol sequence is stored in `self.SwpSeq` and also
        cached under an attribute named `SwpSeq{self.BsNm}`, where `self.BsNm`
        is set to 'RecoveryFromInactivation'. The number of sweeps corresponds
        to the number of `target_recovery_times`.

        This method may be an override of a superclass method. The concluding
        call to `self.CurrVolt()` is important in the context of
        `AnticonvulsantCTBNMarkovModel`. Since `CurrVolt` is overridden to
        account for drug effects, this ensures that initial current calculations
        reflect the drug's presence.

        Args:
            target_recovery_times (list or np.ndarray, optional): A list of
                durations (in ms) for the recovery interval. Defaults to
                `[1, 3, 10, 30, 100, 300, 1000]` ms.
            holding_potential (float, optional): Voltage for holding periods,
                including recovery intervals and tail pulse. Defaults to -80 mV.
            inactivating_voltage (float, optional): Voltage of the conditioning
                pulse to induce inactivation. Defaults to -20 mV.
            test_voltage (float, optional): Voltage of the test pulse to measure
                current after the recovery interval. Defaults to 0 mV.
            holding_duration (float, optional): Duration of the initial holding
                period in ms. Defaults to 200 ms.
            inactivating_duration (float, optional): Duration of the
                inactivating pulse in ms. Defaults to 2000 ms.
            test_duration (float, optional): Duration of the test pulse in ms.
                Defaults to 20 ms.
            tail_duration (float, optional): Duration of the final tail pulse
                in ms. Defaults to 100 ms.
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
        self.SwpSeq[0, :] = 5
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        self.SwpSeq[4, :] = inactivating_voltage
        self.SwpSeq[5, :] = (holding_samples + inactivating_samples)
        self.SwpSeq[6, :] = holding_potential
        self.SwpSeq[7, :] = ((holding_samples + inactivating_samples) + recovery_samples)
        self.SwpSeq[8, :] = test_voltage
        self.SwpSeq[9, :] = (((holding_samples + inactivating_samples) + recovery_samples) + test_samples)
        self.SwpSeq[10, :] = holding_potential
        self.SwpSeq[11, :] = ((((holding_samples + inactivating_samples) + recovery_samples) + test_samples) + tail_samples)
        setattr(self, f'SwpSeq{self.BsNm}', self.SwpSeq.copy())
        self.CurrVolt()
    def create_steady_state_inactivation_protocol(self, test_voltages=None, holding_potential=(- 120), prepulse_duration=2000, test_pulse_voltage=0, test_pulse_duration=5, recovery_duration=100):
        """Creates a protocol to determine steady-state inactivation (availability).

        This method generates a series of voltage clamp sweeps used to assess
        the voltage dependence of steady-state inactivation. Each sweep involves
        a holding potential, a long conditioning prepulse to various test voltages
        to allow channels to reach steady-state, a brief test pulse (usually to a
        fixed voltage) to measure the fraction of available channels, and a
        final recovery period.

        The generated protocol sequence is stored in `self.SwpSeq` and also
        cached under an attribute named `SwpSeq{self.BsNm}`, where `self.BsNm`
        is set to 'SteadyStateInactivation'. The number of sweeps corresponds
        to the number of `test_voltages` (conditioning prepulse voltages).

        This method may be an override of a superclass method. The concluding
        call to `self.CurrVolt()` is significant in the context of
        `AnticonvulsantCTBNMarkovModel`. Since `CurrVolt` is overridden to
        account for drug effects, this ensures that initial current calculations
        reflect the drug's presence.

        Args:
            test_voltages (list or np.ndarray, optional): Voltages for the
                conditioning prepulses. Defaults to `np.arange(-120, -15, 5)` mV.
            holding_potential (float, optional): Voltage for the initial holding
                period and the final recovery period. Defaults to -120 mV.
            prepulse_duration (float, optional): Duration of the conditioning
                prepulses in ms. Defaults to 2000 ms.
            test_pulse_voltage (float, optional): Voltage of the brief test pulse
                used to measure channel availability. Defaults to 0 mV.
            test_pulse_duration (float, optional): Duration of the test pulse
                in ms. Defaults to 5 ms.
            recovery_duration (float, optional): Duration of the final recovery
                period in ms. Defaults to 100 ms.
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
        self.SwpSeq[0, :] = 4
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        self.SwpSeq[4, :] = test_voltages
        self.SwpSeq[5, :] = (holding_samples + prepulse_samples)
        self.SwpSeq[6, :] = test_pulse_voltage
        self.SwpSeq[7, :] = ((holding_samples + prepulse_samples) + test_samples)
        self.SwpSeq[8, :] = holding_potential
        self.SwpSeq[9, :] = (((holding_samples + prepulse_samples) + test_samples) + recovery_samples)
        setattr(self, f'SwpSeq{self.BsNm}', self.SwpSeq.copy())
        self.CurrVolt()