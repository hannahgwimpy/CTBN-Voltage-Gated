import numpy as np
from scipy.integrate import solve_ivp
class HHModel():
    def __init__(self):
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
        V = np.atleast_1d(V)
        v_idx = np.searchsorted(self.v_range, V)
        v_idx = np.clip(v_idx, 0, (len(self.v_range) - 1))
        return {'alpha_m1': self.alpha_m1_vec[v_idx], 'beta_m1': self.beta_m1_vec[v_idx], 'alpha_m2': self.alpha_m2_vec[v_idx], 'beta_m2': self.beta_m2_vec[v_idx], 'alpha_m3': self.alpha_m3_vec[v_idx], 'beta_m3': self.beta_m3_vec[v_idx], 'alpha_h': self.alpha_h_vec[v_idx], 'beta_h': self.beta_h_vec[v_idx]}
    def steady_state_values(self, V):
        rates = self.get_rate_constants(V)
        m1_inf = (rates['alpha_m1'] / (rates['alpha_m1'] + rates['beta_m1']))
        m2_inf = (rates['alpha_m2'] / (rates['alpha_m2'] + rates['beta_m2']))
        m3_inf = (rates['alpha_m3'] / (rates['alpha_m3'] + rates['beta_m3']))
        h_inf = (rates['alpha_h'] / (rates['alpha_h'] + rates['beta_h']))
        return (m1_inf, m2_inf, m3_inf, h_inf)
    def time_constants(self, V):
        rates = self.get_rate_constants(V)
        tau_m1 = (1.0 / (rates['alpha_m1'] + rates['beta_m1']))
        tau_m2 = (1.0 / (rates['alpha_m2'] + rates['beta_m2']))
        tau_m3 = (1.0 / (rates['alpha_m3'] + rates['beta_m3']))
        tau_h = (1.0 / (rates['alpha_h'] + rates['beta_h']))
        return (tau_m1, tau_m2, tau_m3, tau_h)
    def compute_sodium_current(self, V, m1, m2, m3, h):
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
        V = (voltage if (voltage is not None) else self.vm)
        (m1_inf, m2_inf, m3_inf, h_inf) = self.steady_state_values(V)
        pop = np.zeros(20)
        pop[6] = (((m1_inf * m2_inf) * m3_inf) * h_inf)
        inact_prob = ((1 - h_inf) / 6)
        pop[7:13] = inact_prob
        return pop
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