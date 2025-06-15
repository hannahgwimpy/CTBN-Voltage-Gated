import numpy as np
from scipy.integrate import solve_ivp
class CTBNMarkovModel():
    def __init__(self):
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
        self.stRatesVolt()
    def stRatesVolt(self):
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
        vidx = np.argmin(np.abs((self.vt - vm)))
        return {'fwd_I0': self.fwd_rates_I0[vidx], 'bwd_I0': self.bwd_rates_I0[vidx], 'fwd_I1': self.fwd_rates_I1[vidx], 'bwd_I1': self.bwd_rates_I1[vidx], 'inact_on': self.inact_on_rates[vidx], 'inact_off': self.inact_off_rates[vidx], 'k613': self.k613dis_vec[vidx], 'k136': self.k136dis_vec[vidx]}
    def _update_scalar_rates(self):
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
        self._store_ctbn_results_vectorized([idx], np.array([self.state_probs_flat]), np.array([self.vm]))
    def _store_ctbn_results_vectorized(self, indices, state_probs_batch, voltages):
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
    def __init__(self, drug_concentration=0.0, drug_type='DPH'):
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
        self.drug_type = drug_type.upper()
        self.init_parameters()
        self.update_rates()
    def set_drug_concentration(self, drug_concentration):
        self.drug_concentration = drug_concentration
        self.update_rates()
    def init_parameters(self):
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
        self.k_on_resting = 0
        self.k_on_inactivated = (self.k_on_inactivated_base * self.drug_concentration)
        self.k_off_resting = 0
        self.k_off_inactivated = self.k_off
    def init_waves(self):
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
        self.stRatesVolt()
    def stRatesVolt(self):
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
        vidx = np.searchsorted(self.vt, vm)
        vidx = np.clip(vidx, 0, (len(self.vt) - 1))
        return {'fwd_flat': self.fwd_rates_flat[vidx, :], 'bwd_flat': self.bwd_rates_flat[vidx, :], 'inact_on_flat': self.inact_on_rates_flat[vidx, :], 'inact_off_flat': self.inact_off_rates_flat[vidx, :], 'drug_on_I0': self.drug_on_rates_I0, 'drug_off_I0': self.drug_off_rates_I0, 'drug_on_I1': self.drug_on_rates_I1, 'drug_off_I1': self.drug_off_rates_I1}
    def _update_scalar_rates(self):
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
        self._store_ctbn_results_vectorized([idx], np.array([self.state_probs_flat[:24]]), np.array([self.vm]))
    def _store_ctbn_results_vectorized(self, indices, state_probs_batch, voltages):
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