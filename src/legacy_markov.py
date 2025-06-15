import numpy as np
from scipy.integrate import solve_ivp
class MarkovModel():
    def __init__(self):
        self.NumSwps = 0
        self.num_states = 13
        self.vm = (- 80)
        self.init_parameters()
        self.init_waves()
        self.stRatesVolt()
        self.CurrVolt()
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
        self.vm = (- 80)
        self.PNasc = 1e-05
        self._reusable_y0 = np.zeros(12)
    def init_waves(self):
        self.vt = np.arange((- 200), 201)
        self.pop = np.zeros(13)
        self.dstdt = np.zeros(12)
        self._reusable_y0 = np.zeros(12)
        self.iscft = np.zeros_like(self.vt)
        self.create_rate_waves()
        self.stRatesVolt()
    def create_rate_waves(self):
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis', 'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis', 'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis', 'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis', 'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis', 'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis', 'k98dis', 'k87dis']
        for name in rate_names:
            setattr(self, (name + '_vec'), np.zeros_like(self.vt, dtype=float))
    def stRatesVolt(self):
        if ((not hasattr(self, 'ClipRate')) or (self.ClipRate is None)):
            self.ClipRate = 1000
        if (not hasattr(self, 'k12dis_vec')):
            self.create_rate_waves()
        vt = self.vt
        amt = (self.alcoeff * np.exp((vt / self.alslp)))
        bmt = (self.btcoeff * np.exp(((- vt) / self.btslp)))
        gmt = (self.gmcoeff * np.exp((vt / self.gmslp)))
        dmt = (self.dlcoeff * np.exp(((- vt) / self.dlslp)))
        konlo = (self.ConCoeff * np.exp((vt / self.ConSlp)))
        kofflo = (self.CoffCoeff * np.exp(((- vt) / self.CoffSlp)))
        konop = (self.OpOnCoeff * np.exp((vt / self.OpOnSlp)))
        koffop = (self.OpOffCoeff * np.exp(((- vt) / self.OpOffSlp)))
        self.k12dis_vec = np.minimum((4 * amt), self.ClipRate)
        self.k23dis_vec = np.minimum((3 * amt), self.ClipRate)
        self.k34dis_vec = np.minimum((2 * amt), self.ClipRate)
        self.k45dis_vec = np.minimum(amt, self.ClipRate)
        self.k56dis_vec = np.minimum(gmt, self.ClipRate)
        self.k65dis_vec = np.minimum(dmt, self.ClipRate)
        self.k54dis_vec = np.minimum((4 * bmt), self.ClipRate)
        self.k43dis_vec = np.minimum((3 * bmt), self.ClipRate)
        self.k32dis_vec = np.minimum((2 * bmt), self.ClipRate)
        self.k21dis_vec = np.minimum(bmt, self.ClipRate)
        dph = 1
        self.k17dis_vec = np.minimum((konlo * dph), self.ClipRate)
        self.k71dis_vec = np.minimum(kofflo, self.ClipRate)
        self.k28dis_vec = np.minimum((self.k17dis_vec * self.alfac), self.ClipRate)
        self.k82dis_vec = np.minimum((self.k71dis_vec / self.btfac), self.ClipRate)
        self.k39dis_vec = np.minimum((self.k17dis_vec * (self.alfac ** 2)), self.ClipRate)
        self.k93dis_vec = np.minimum((self.k71dis_vec / (self.btfac ** 2)), self.ClipRate)
        self.k410dis_vec = np.minimum((self.k17dis_vec * (self.alfac ** 3)), self.ClipRate)
        self.k104dis_vec = np.minimum((self.k71dis_vec / (self.btfac ** 3)), self.ClipRate)
        self.k511dis_vec = np.minimum((self.k17dis_vec * (self.alfac ** 4)), self.ClipRate)
        self.k115dis_vec = np.minimum((self.k71dis_vec / (self.btfac ** 4)), self.ClipRate)
        self.k612dis_vec = np.minimum(konop, self.ClipRate)
        self.k126dis_vec = np.minimum(koffop, self.ClipRate)
        self.k78dis_vec = np.minimum(((4 * amt) * self.alfac), self.ClipRate)
        self.k89dis_vec = np.minimum(((3 * amt) * self.alfac), self.ClipRate)
        self.k910dis_vec = np.minimum(((2 * amt) * self.alfac), self.ClipRate)
        self.k1011dis_vec = np.minimum((amt * self.alfac), self.ClipRate)
        self.k1112dis_vec = np.minimum(gmt, self.ClipRate)
        self.k1110dis_vec = np.minimum(((4 * bmt) * (1 / self.btfac)), self.ClipRate)
        self.k109dis_vec = np.minimum(((3 * bmt) * (1 / self.btfac)), self.ClipRate)
        self.k98dis_vec = np.minimum(((2 * bmt) * (1 / self.btfac)), self.ClipRate)
        self.k87dis_vec = np.minimum((bmt * (1 / self.btfac)), self.ClipRate)
        k115_safe = np.where((self.k115dis_vec > 0), self.k115dis_vec, 1.0)
        self.k1211dis_vec = np.minimum((((self.k65dis_vec * self.k511dis_vec) * self.k126dis_vec) / (self.k612dis_vec * k115_safe)), self.ClipRate)
        self._update_scalar_rates()
    def _update_scalar_rates(self):
        vidx = np.argmin(np.abs((self.vt - self.vm)))
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis', 'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis', 'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis', 'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis', 'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis', 'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis', 'k98dis', 'k87dis']
        for name in rate_names:
            vec_name = (name + '_vec')
            if hasattr(self, vec_name):
                vec_array = getattr(self, vec_name)
                if (isinstance(vec_array, np.ndarray) and (len(vec_array) > vidx)):
                    setattr(self, name, vec_array[vidx])
                else:
                    setattr(self, name, 0.0)
            else:
                setattr(self, name, 0.0)
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
        if (not hasattr(self, 'k12dis_vec')):
            self.create_rate_waves()
        self.stRatesVolt()
        self._update_scalar_rates()
        def safe_div(a, b, default=0.0):
            if np.isscalar(b):
                if (abs(b) > 1e-10):
                    return (a / b)
                else:
                    return default
            else:
                result = np.full_like(a, default, dtype=float)
                mask = (np.abs(b) > 1e-10)
                if np.any(mask):
                    result[mask] = (a[mask] / b[mask])
                return result
        du1 = safe_div(self.k12dis, self.k21dis)
        du2 = safe_div(self.k23dis, self.k32dis)
        du3 = safe_div(self.k34dis, self.k43dis)
        du4 = safe_div(self.k45dis, self.k54dis)
        du5 = safe_div(self.k56dis, self.k65dis)
        du7 = safe_div(self.k17dis, self.k71dis)
        du8 = safe_div(self.k78dis, self.k87dis)
        du9 = safe_div(self.k89dis, self.k98dis)
        du10 = safe_div(self.k910dis, self.k109dis)
        du11 = safe_div(self.k1011dis, self.k1110dis)
        du12 = safe_div(self.k1112dis, self.k1211dis)
        dusuma = (((((1 + du1) + (du1 * du2)) + ((du1 * du2) * du3)) + (((du1 * du2) * du3) * du4)) + ((((du1 * du2) * du3) * du4) * du5))
        dusumb = (((((du7 + (du7 * du8)) + ((du7 * du8) * du9)) + (((du7 * du8) * du9) * du10)) + ((((du7 * du8) * du9) * du10) * du11)) + (((((du7 * du8) * du9) * du10) * du11) * du12))
        dusum = (dusuma + dusumb)
        pop = np.zeros(12)
        if (dusum > 1e-10):
            du_products = np.array([1, du1, (du1 * du2), ((du1 * du2) * du3), (((du1 * du2) * du3) * du4), ((((du1 * du2) * du3) * du4) * du5)])
            du7_products = np.array([du7, (du7 * du8), ((du7 * du8) * du9), (((du7 * du8) * du9) * du10), ((((du7 * du8) * du9) * du10) * du11), (((((du7 * du8) * du9) * du10) * du11) * du12)])
            pop[:6] = (du_products / dusum)
            pop[6:12] = (du7_products / dusum)
        else:
            pop[0] = 0.98
            pop[1] = 0.02
        pop = np.nan_to_num(pop, nan=0.0)
        return pop
    def NowDerivs(self, t, y):
        vidx = np.searchsorted(self.vt, self.vm)
        vidx = np.clip(vidx, 0, (len(self.vt) - 1))
        if (np.any(np.isnan(y)) or np.any(np.isinf(y))):
            return np.zeros_like(y)
        k12dis = self.k12dis_vec[vidx]
        k23dis = self.k23dis_vec[vidx]
        k34dis = self.k34dis_vec[vidx]
        k45dis = self.k45dis_vec[vidx]
        k56dis = self.k56dis_vec[vidx]
        k65dis = self.k65dis_vec[vidx]
        k54dis = self.k54dis_vec[vidx]
        k43dis = self.k43dis_vec[vidx]
        k32dis = self.k32dis_vec[vidx]
        k21dis = self.k21dis_vec[vidx]
        k17dis = self.k17dis_vec[vidx]
        k71dis = self.k71dis_vec[vidx]
        k28dis = self.k28dis_vec[vidx]
        k82dis = self.k82dis_vec[vidx]
        k39dis = self.k39dis_vec[vidx]
        k93dis = self.k93dis_vec[vidx]
        k410dis = self.k410dis_vec[vidx]
        k104dis = self.k104dis_vec[vidx]
        k511dis = self.k511dis_vec[vidx]
        k115dis = self.k115dis_vec[vidx]
        k612dis = self.k612dis_vec[vidx]
        k126dis = self.k126dis_vec[vidx]
        k78dis = self.k78dis_vec[vidx]
        k89dis = self.k89dis_vec[vidx]
        k910dis = self.k910dis_vec[vidx]
        k1011dis = self.k1011dis_vec[vidx]
        k1112dis = self.k1112dis_vec[vidx]
        k1211dis = self.k1211dis_vec[vidx]
        k1110dis = self.k1110dis_vec[vidx]
        k109dis = self.k109dis_vec[vidx]
        k98dis = self.k98dis_vec[vidx]
        k87dis = self.k87dis_vec[vidx]
        st = y.copy()
        Q = np.zeros((12, 12))
        Q[(0, 1)] = k21dis
        Q[(0, 6)] = k71dis
        Q[(1, 0)] = k12dis
        Q[(1, 2)] = k32dis
        Q[(1, 7)] = k82dis
        Q[(2, 1)] = k23dis
        Q[(2, 3)] = k43dis
        Q[(2, 8)] = k93dis
        Q[(3, 2)] = k34dis
        Q[(3, 4)] = k54dis
        Q[(3, 9)] = k104dis
        Q[(4, 3)] = k45dis
        Q[(4, 5)] = k65dis
        Q[(4, 10)] = k115dis
        Q[(5, 4)] = k56dis
        Q[(5, 11)] = k126dis
        Q[(6, 0)] = k17dis
        Q[(6, 7)] = k87dis
        Q[(7, 6)] = k78dis
        Q[(7, 8)] = k98dis
        Q[(7, 1)] = k28dis
        Q[(8, 7)] = k89dis
        Q[(8, 9)] = k109dis
        Q[(8, 2)] = k39dis
        Q[(9, 8)] = k910dis
        Q[(9, 10)] = k1110dis
        Q[(9, 3)] = k410dis
        Q[(10, 9)] = k1011dis
        Q[(10, 11)] = k1211dis
        Q[(10, 4)] = k511dis
        Q[(11, 10)] = k1112dis
        Q[(11, 5)] = k612dis
        Q[(0, 0)] = (- (k12dis + k17dis))
        Q[(1, 1)] = (- ((k21dis + k23dis) + k28dis))
        Q[(2, 2)] = (- ((k32dis + k34dis) + k39dis))
        Q[(3, 3)] = (- ((k43dis + k45dis) + k410dis))
        Q[(4, 4)] = (- ((k54dis + k56dis) + k511dis))
        Q[(5, 5)] = (- (k65dis + k612dis))
        Q[(6, 6)] = (- (k71dis + k78dis))
        Q[(7, 7)] = (- ((k82dis + k87dis) + k89dis))
        Q[(8, 8)] = (- ((k93dis + k98dis) + k910dis))
        Q[(9, 9)] = (- ((k104dis + k109dis) + k1011dis))
        Q[(10, 10)] = (- ((k115dis + k1110dis) + k1112dis))
        Q[(11, 11)] = (- (k126dis + k1211dis))
        dstdt = np.zeros_like(y)
        for i in range(12):
            for j in range(12):
                dstdt[i] += (Q[(i, j)] * st[j])
        if (np.any(np.isnan(dstdt)) or np.any(np.isinf(dstdt))):
            return np.zeros_like(st)
        return dstdt
    def Sweep(self, SwpNo):
        if ((SwpNo >= self.SwpSeq.shape[1]) or (SwpNo < 0)):
            raise ValueError(f'Invalid sweep number {SwpNo}')
        SwpSeq = self.SwpSeq
        NumEpchs = int(SwpSeq[(0, SwpNo)])
        if ((NumEpchs <= 0) or (((2 * NumEpchs) + 1) >= SwpSeq.shape[0])):
            raise ValueError('Invalid number of epochs in protocol')
        total_points = (int(SwpSeq[(((2 * NumEpchs) + 1), SwpNo)]) + 1)
        sampint = 0.005
        self.SimSwp = np.zeros(total_points)
        self.SimOp = np.zeros(total_points)
        self.SimIn = np.zeros(total_points)
        self.SimAv = np.zeros(total_points)
        self.SimCom = np.zeros(total_points)
        self.pop = np.zeros(13)
        self.pop[0] = 1.0
        epoch_voltages = np.zeros((NumEpchs + 1))
        epoch_end_times = np.zeros((NumEpchs + 1))
        epoch_voltages[0] = SwpSeq[(2, SwpNo)]
        epoch_end_times[0] = 0.0
        for e in range(1, (NumEpchs + 1)):
            epoch_voltages[e] = SwpSeq[((2 * e), SwpNo)]
            epoch_end_times[e] = (int(SwpSeq[(((2 * e) + 1), SwpNo)]) * sampint)
        self.vm = epoch_voltages[0]
        self.CurrVolt()
        self.pop = self.EquilOccup(self.vm)
        self._store_results(0, 0)
        current_time = 0.0
        store_idx = 1
        for epoch in range(1, (NumEpchs + 1)):
            self.vm = epoch_voltages[epoch]
            epoch_end_time = epoch_end_times[epoch]
            self._update_scalar_rates()
            self.CurrVolt()
            num_points = max(2, (int(((epoch_end_time - current_time) / sampint)) + 1))
            t_eval = np.linspace(current_time, epoch_end_time, num_points)
            if (len(t_eval) <= 1):
                current_time = epoch_end_time
                continue
            self._reusable_y0[:] = self.pop[:12]
            sol = solve_ivp(self.NowDerivs, [current_time, epoch_end_time], self._reusable_y0, method='LSODA', t_eval=t_eval, rtol=1e-06, atol=1e-08)
            if (sol.success and (len(sol.t) > 0)):
                batch_size = len(sol.t)
                end_idx = min((store_idx + batch_size), total_points)
                batch_indices = np.arange(store_idx, end_idx)
                actual_batch = len(batch_indices)
                if (actual_batch > 0):
                    batch_states = sol.y[:, :actual_batch].T
                    self._store_results_vectorized(batch_indices, batch_states, self.vm)
                    self.pop[:12] = sol.y[:, - 1]
                    store_idx = end_idx
            current_time = epoch_end_time
        self.time = np.arange(0, (total_points * sampint), sampint)[:total_points]
        return (sol.t, self.SimSwp)
    def _store_results(self, idx, t):
        vidx = np.searchsorted(self.vt, self.vm)
        vidx = np.clip(vidx, 0, (len(self.vt) - 1))
        open_prob = self.pop[5]
        current = (((open_prob * self.iscft[vidx]) * self.numchan) * self.current_scaling)
        self.SimSwp[idx] = current
        self.SimOp[idx] = self.pop[5]
        self.SimIn[idx] = np.sum(self.pop[6:])
        self.SimAv[idx] = np.sum(self.pop[:6])
        self.SimCom[idx] = self.vm
    def _store_results_vectorized(self, indices, batch_states, voltage):
        if (len(indices) == 0):
            return
        vidx = np.searchsorted(self.vt, voltage)
        vidx = np.clip(vidx, 0, (len(self.vt) - 1))
        current_factor = self.iscft[vidx]
        open_probs = batch_states[:, 5]
        currents = (((open_probs * current_factor) * self.numchan) * self.current_scaling)
        inactivated = np.sum(batch_states[:, 6:], axis=1)
        available = np.sum(batch_states[:, :6], axis=1)
        self.SimSwp[indices] = currents
        self.SimOp[indices] = open_probs
        self.SimIn[indices] = inactivated
        self.SimAv[indices] = available
        self.SimCom[indices] = voltage
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
class AnticonvulsantMarkovModel(MarkovModel):
    def __init__(self, drug_concentration=0.0, drug_type='DPH'):
        self.NumSwps = 0
        self.num_states = 25
        self.drug_concentration = drug_concentration
        self.drug_type = drug_type.upper()
        self.vm = (- 80)
        self.init_parameters()
        self.init_waves()
        self._update_drug_rates()
        self.CurrVolt()
        self.create_default_protocol()
        self.pop = self.EquilOccup(self.vm)
    def set_drug_type(self, drug_type):
        self.drug_type = drug_type.upper()
        self.init_parameters()
        self._update_drug_rates()
        self.pop = self.EquilOccup(self.vm)
    def set_drug_concentration(self, drug_concentration):
        self.drug_concentration = drug_concentration
        self._update_drug_rates()
        self.pop = self.EquilOccup(self.vm)
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
        self.k_off = (params['k_off_base'] * params.get('k_off_scaling', 1.0))
        self.KR_resting = (self.KI_inactivated * 100.0)
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
        self._reusable_y0 = np.zeros(24)
        self._update_drug_rates()
    def _update_drug_rates(self):
        self.k_on_resting = 0
        self.k_on_inactivated = (self.k_on_inactivated_base * self.drug_concentration)
        self.k_off_resting = 0
        self.k_off_inactivated = self.k_off
    def init_waves(self):
        self.vt = np.arange((- 200), 201)
        self.pop = np.zeros(24)
        self.dstdt = np.zeros(24)
        self._reusable_y0 = np.zeros(24)
        self.iscft = np.zeros_like(self.vt)
        self.create_rate_waves()
        self.stRatesVolt()
    def create_rate_waves(self):
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis', 'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis', 'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis', 'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis', 'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis', 'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis', 'k98dis', 'k87dis']
        for name in rate_names:
            setattr(self, (name + '_vec'), np.zeros_like(self.vt, dtype=float))
    def stRatesVolt(self):
        if ((not hasattr(self, 'ClipRate')) or (self.ClipRate is None)):
            self.ClipRate = 1000
        if (not hasattr(self, 'k12dis_vec')):
            self.create_rate_waves()
        vt = self.vt
        amt = (self.alcoeff * np.exp((vt / self.alslp)))
        bmt = (self.btcoeff * np.exp(((- vt) / self.btslp)))
        gmt = (self.gmcoeff * np.exp((vt / self.gmslp)))
        dmt = (self.dlcoeff * np.exp(((- vt) / self.dlslp)))
        konlo = (self.ConCoeff * np.exp((vt / self.ConSlp)))
        kofflo = (self.CoffCoeff * np.exp(((- vt) / self.CoffSlp)))
        konop = (self.OpOnCoeff * np.exp((vt / self.OpOnSlp)))
        koffop = (self.OpOffCoeff * np.exp(((- vt) / self.OpOffSlp)))
        self.k12dis_vec = np.minimum((4 * amt), self.ClipRate)
        self.k23dis_vec = np.minimum((3 * amt), self.ClipRate)
        self.k34dis_vec = np.minimum((2 * amt), self.ClipRate)
        self.k45dis_vec = np.minimum(amt, self.ClipRate)
        self.k56dis_vec = np.minimum(gmt, self.ClipRate)
        self.k65dis_vec = np.minimum(dmt, self.ClipRate)
        self.k54dis_vec = np.minimum((4 * bmt), self.ClipRate)
        self.k43dis_vec = np.minimum((3 * bmt), self.ClipRate)
        self.k32dis_vec = np.minimum((2 * bmt), self.ClipRate)
        self.k21dis_vec = np.minimum(bmt, self.ClipRate)
        dph = 1
        self.k17dis_vec = np.minimum((konlo * dph), self.ClipRate)
        self.k71dis_vec = np.minimum(kofflo, self.ClipRate)
        self.k28dis_vec = np.minimum((self.k17dis_vec * self.alfac), self.ClipRate)
        self.k82dis_vec = np.minimum((self.k71dis_vec / self.btfac), self.ClipRate)
        self.k39dis_vec = np.minimum((self.k17dis_vec * (self.alfac ** 2)), self.ClipRate)
        self.k93dis_vec = np.minimum((self.k71dis_vec / (self.btfac ** 2)), self.ClipRate)
        self.k410dis_vec = np.minimum((self.k17dis_vec * (self.alfac ** 3)), self.ClipRate)
        self.k104dis_vec = np.minimum((self.k71dis_vec / (self.btfac ** 3)), self.ClipRate)
        self.k511dis_vec = np.minimum((self.k17dis_vec * (self.alfac ** 4)), self.ClipRate)
        self.k115dis_vec = np.minimum((self.k71dis_vec / (self.btfac ** 4)), self.ClipRate)
        self.k612dis_vec = np.minimum(konop, self.ClipRate)
        self.k126dis_vec = np.minimum(koffop, self.ClipRate)
        self.k78dis_vec = np.minimum(((4 * amt) * self.alfac), self.ClipRate)
        self.k89dis_vec = np.minimum(((3 * amt) * self.alfac), self.ClipRate)
        self.k910dis_vec = np.minimum(((2 * amt) * self.alfac), self.ClipRate)
        self.k1011dis_vec = np.minimum((amt * self.alfac), self.ClipRate)
        self.k1112dis_vec = np.minimum(gmt, self.ClipRate)
        self.k1110dis_vec = np.minimum(((4 * bmt) * (1 / self.btfac)), self.ClipRate)
        self.k109dis_vec = np.minimum(((3 * bmt) * (1 / self.btfac)), self.ClipRate)
        self.k98dis_vec = np.minimum(((2 * bmt) * (1 / self.btfac)), self.ClipRate)
        self.k87dis_vec = np.minimum((bmt * (1 / self.btfac)), self.ClipRate)
        k115_safe = np.where((self.k115dis_vec > 0), self.k115dis_vec, 1.0)
        self.k1211dis_vec = np.minimum((((self.k65dis_vec * self.k511dis_vec) * self.k126dis_vec) / (self.k612dis_vec * k115_safe)), self.ClipRate)
        self._update_scalar_rates()
    def _update_scalar_rates(self):
        vidx = np.argmin(np.abs((self.vt - self.vm)))
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis', 'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis', 'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis', 'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis', 'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis', 'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis', 'k98dis', 'k87dis']
        for name in rate_names:
            vec_name = (name + '_vec')
            if hasattr(self, vec_name):
                vec_array = getattr(self, vec_name)
                if (isinstance(vec_array, np.ndarray) and (len(vec_array) > vidx)):
                    setattr(self, name, vec_array[vidx])
                else:
                    setattr(self, name, 0.0)
            else:
                setattr(self, name, 0.0)
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
        if (not hasattr(self, 'k12dis_vec')):
            self.create_rate_waves()
        self.stRatesVolt()
        self._update_scalar_rates()
        def safe_div(a, b, default=0.0):
            if np.isscalar(b):
                if (abs(b) > 1e-10):
                    return (a / b)
                else:
                    return default
            else:
                result = np.full_like(a, default, dtype=float)
                mask = (np.abs(b) > 1e-10)
                if np.any(mask):
                    result[mask] = (a[mask] / b[mask])
                return result
        du1 = safe_div(self.k12dis, self.k21dis)
        du2 = safe_div(self.k23dis, self.k32dis)
        du3 = safe_div(self.k34dis, self.k43dis)
        du4 = safe_div(self.k45dis, self.k54dis)
        du5 = safe_div(self.k56dis, self.k65dis)
        du7 = safe_div(self.k17dis, self.k71dis)
        du8 = safe_div(self.k78dis, self.k87dis)
        du9 = safe_div(self.k89dis, self.k98dis)
        du10 = safe_div(self.k910dis, self.k109dis)
        du11 = safe_div(self.k1011dis, self.k1110dis)
        du12 = safe_div(self.k1112dis, self.k1211dis)
        drug_factor_closed = 0
        drug_factor_inactivated = (self.k_on_inactivated / self.k_off_inactivated)
        dusuma_free = (((((1 + du1) + (du1 * du2)) + ((du1 * du2) * du3)) + (((du1 * du2) * du3) * du4)) + ((((du1 * du2) * du3) * du4) * du5))
        dusumb_free = (((((du7 + (du7 * du8)) + ((du7 * du8) * du9)) + (((du7 * du8) * du9) * du10)) + ((((du7 * du8) * du9) * du10) * du11)) + (((((du7 * du8) * du9) * du10) * du11) * du12))
        dusuma_drug = (drug_factor_closed * dusuma_free)
        dusumb_drug = (drug_factor_inactivated * dusumb_free)
        dusum_total = (((dusuma_free + dusumb_free) + dusuma_drug) + dusumb_drug)
        pop = np.zeros(24)
        if (dusum_total > 1e-10):
            closed_products = np.array([1, du1, (du1 * du2), ((du1 * du2) * du3), (((du1 * du2) * du3) * du4), ((((du1 * du2) * du3) * du4) * du5)])
            pop[0:6] = (closed_products / dusum_total)
            inact_products = np.array([du7, (du7 * du8), ((du7 * du8) * du9), (((du7 * du8) * du9) * du10), ((((du7 * du8) * du9) * du10) * du11), (((((du7 * du8) * du9) * du10) * du11) * du12)])
            pop[6:12] = (inact_products / dusum_total)
            pop[12:18] = ((drug_factor_closed * closed_products) / dusum_total)
            pop[18:24] = ((drug_factor_inactivated * inact_products) / dusum_total)
        else:
            pop[0] = 0.98
            pop[1] = 0.02
        pop = np.nan_to_num(pop, nan=0.0)
        return pop
    def NowDerivs(self, t, y):
        vidx = np.searchsorted(self.vt, self.vm)
        vidx = np.clip(vidx, 0, (len(self.vt) - 1))
        if (np.any(np.isnan(y)) or np.any(np.isinf(y))):
            return np.zeros_like(y)
        k12dis = self.k12dis_vec[vidx]
        k21dis = self.k21dis_vec[vidx]
        k23dis = self.k23dis_vec[vidx]
        k32dis = self.k32dis_vec[vidx]
        k34dis = self.k34dis_vec[vidx]
        k43dis = self.k43dis_vec[vidx]
        k45dis = self.k45dis_vec[vidx]
        k54dis = self.k54dis_vec[vidx]
        k56dis = self.k56dis_vec[vidx]
        k65dis = self.k65dis_vec[vidx]
        k17dis = self.k17dis_vec[vidx]
        k71dis = self.k71dis_vec[vidx]
        k28dis = self.k28dis_vec[vidx]
        k82dis = self.k82dis_vec[vidx]
        k39dis = self.k39dis_vec[vidx]
        k93dis = self.k93dis_vec[vidx]
        k410dis = self.k410dis_vec[vidx]
        k104dis = self.k104dis_vec[vidx]
        k511dis = self.k511dis_vec[vidx]
        k115dis = self.k115dis_vec[vidx]
        k612dis = self.k612dis_vec[vidx]
        k126dis = self.k126dis_vec[vidx]
        k78dis = self.k78dis_vec[vidx]
        k87dis = self.k87dis_vec[vidx]
        k89dis = self.k89dis_vec[vidx]
        k98dis = self.k98dis_vec[vidx]
        k910dis = self.k910dis_vec[vidx]
        k109dis = self.k109dis_vec[vidx]
        k1011dis = self.k1011dis_vec[vidx]
        k1110dis = self.k1110dis_vec[vidx]
        k1112dis = self.k1112dis_vec[vidx]
        k1211dis = self.k1211dis_vec[vidx]
        k_on_closed = self.k_on_resting
        k_off_closed = self.k_off_resting
        k_on_inact = self.k_on_inactivated
        k_off_inact = self.k_off_inactivated
        st = y.copy()
        Q = np.zeros((24, 24))
        Q[(0, 1)] = k21dis
        Q[(0, 6)] = k71dis
        Q[(1, 0)] = k12dis
        Q[(1, 2)] = k32dis
        Q[(1, 7)] = k82dis
        Q[(2, 1)] = k23dis
        Q[(2, 3)] = k43dis
        Q[(2, 8)] = k93dis
        Q[(3, 2)] = k34dis
        Q[(3, 4)] = k54dis
        Q[(3, 9)] = k104dis
        Q[(4, 3)] = k45dis
        Q[(4, 5)] = k65dis
        Q[(4, 10)] = k115dis
        Q[(5, 4)] = k56dis
        Q[(5, 11)] = k126dis
        Q[(6, 0)] = k17dis
        Q[(6, 7)] = k87dis
        Q[(7, 1)] = k28dis
        Q[(7, 6)] = k78dis
        Q[(7, 8)] = k98dis
        Q[(8, 2)] = k39dis
        Q[(8, 7)] = k89dis
        Q[(8, 9)] = k109dis
        Q[(9, 3)] = k410dis
        Q[(9, 8)] = k910dis
        Q[(9, 10)] = k1110dis
        Q[(10, 4)] = k511dis
        Q[(10, 9)] = k1011dis
        Q[(10, 11)] = k1211dis
        Q[(11, 5)] = k612dis
        Q[(11, 10)] = k1112dis
        Q[(12, 13)] = k21dis
        Q[(12, 18)] = k71dis
        Q[(13, 12)] = k12dis
        Q[(13, 14)] = k32dis
        Q[(13, 19)] = k82dis
        Q[(14, 13)] = k23dis
        Q[(14, 15)] = k43dis
        Q[(14, 20)] = k93dis
        Q[(15, 14)] = k34dis
        Q[(15, 16)] = k54dis
        Q[(15, 21)] = k104dis
        Q[(16, 15)] = k45dis
        Q[(16, 17)] = k65dis
        Q[(16, 22)] = k115dis
        Q[(17, 16)] = k56dis
        Q[(17, 23)] = k126dis
        Q[(18, 12)] = k17dis
        Q[(18, 19)] = k87dis
        Q[(19, 13)] = k28dis
        Q[(19, 18)] = k78dis
        Q[(19, 20)] = k98dis
        Q[(20, 14)] = k39dis
        Q[(20, 19)] = k89dis
        Q[(20, 21)] = k109dis
        Q[(21, 15)] = k410dis
        Q[(21, 20)] = k910dis
        Q[(21, 22)] = k1110dis
        Q[(22, 16)] = k511dis
        Q[(22, 21)] = k1011dis
        Q[(22, 23)] = k1211dis
        Q[(23, 17)] = k612dis
        Q[(23, 22)] = k1112dis
        Q[(12, 0)] = k_on_closed
        Q[(0, 12)] = k_off_closed
        Q[(13, 1)] = k_on_closed
        Q[(1, 13)] = k_off_closed
        Q[(14, 2)] = k_on_closed
        Q[(2, 14)] = k_off_closed
        Q[(15, 3)] = k_on_closed
        Q[(3, 15)] = k_off_closed
        Q[(16, 4)] = k_on_closed
        Q[(4, 16)] = k_off_closed
        Q[(17, 5)] = k_on_closed
        Q[(5, 17)] = k_off_closed
        Q[(18, 6)] = k_on_inact
        Q[(6, 18)] = k_off_inact
        Q[(19, 7)] = k_on_inact
        Q[(7, 19)] = k_off_inact
        Q[(20, 8)] = k_on_inact
        Q[(8, 20)] = k_off_inact
        Q[(21, 9)] = k_on_inact
        Q[(9, 21)] = k_off_inact
        Q[(22, 10)] = k_on_inact
        Q[(10, 22)] = k_off_inact
        Q[(23, 11)] = k_on_inact
        Q[(11, 23)] = k_off_inact
        Q[(0, 0)] = (- ((k12dis + k17dis) + k_on_closed))
        Q[(1, 1)] = (- (((k21dis + k23dis) + k28dis) + k_on_closed))
        Q[(2, 2)] = (- (((k32dis + k34dis) + k39dis) + k_on_closed))
        Q[(3, 3)] = (- (((k43dis + k45dis) + k410dis) + k_on_closed))
        Q[(4, 4)] = (- (((k54dis + k56dis) + k511dis) + k_on_closed))
        Q[(5, 5)] = (- ((k65dis + k612dis) + k_on_closed))
        Q[(6, 6)] = (- ((k71dis + k78dis) + k_on_inact))
        Q[(7, 7)] = (- (((k82dis + k87dis) + k89dis) + k_on_inact))
        Q[(8, 8)] = (- (((k93dis + k98dis) + k910dis) + k_on_inact))
        Q[(9, 9)] = (- (((k104dis + k109dis) + k1011dis) + k_on_inact))
        Q[(10, 10)] = (- (((k115dis + k1110dis) + k1112dis) + k_on_inact))
        Q[(11, 11)] = (- ((k126dis + k1211dis) + k_on_inact))
        Q[(12, 12)] = (- ((k12dis + k17dis) + k_off_closed))
        Q[(13, 13)] = (- (((k21dis + k23dis) + k28dis) + k_off_closed))
        Q[(14, 14)] = (- (((k32dis + k34dis) + k39dis) + k_off_closed))
        Q[(15, 15)] = (- (((k43dis + k45dis) + k410dis) + k_off_closed))
        Q[(16, 16)] = (- (((k54dis + k56dis) + k511dis) + k_off_closed))
        Q[(17, 17)] = (- ((k65dis + k612dis) + k_off_closed))
        Q[(18, 18)] = (- ((k71dis + k78dis) + k_off_inact))
        Q[(19, 19)] = (- (((k82dis + k87dis) + k89dis) + k_off_inact))
        Q[(20, 20)] = (- (((k93dis + k98dis) + k910dis) + k_off_inact))
        Q[(21, 21)] = (- (((k104dis + k109dis) + k1011dis) + k_off_inact))
        Q[(22, 22)] = (- (((k115dis + k1110dis) + k1112dis) + k_off_inact))
        Q[(23, 23)] = (- ((k126dis + k1211dis) + k_off_inact))
        dstdt = np.zeros_like(y)
        for i in range(24):
            for j in range(24):
                dstdt[i] += (Q[(i, j)] * st[j])
        if (np.any(np.isnan(dstdt)) or np.any(np.isinf(dstdt))):
            return np.zeros_like(st)
        return dstdt
    def Sweep(self, SwpNo):
        if ((SwpNo >= self.SwpSeq.shape[1]) or (SwpNo < 0)):
            raise ValueError(f'Invalid sweep number {SwpNo}')
        SwpSeq = self.SwpSeq
        NumEpchs = int(SwpSeq[(0, SwpNo)])
        if ((NumEpchs <= 0) or (((2 * NumEpchs) + 1) >= SwpSeq.shape[0])):
            raise ValueError('Invalid number of epochs in protocol')
        total_points = (int(SwpSeq[(((2 * NumEpchs) + 1), SwpNo)]) + 1)
        sampint = 0.005
        self.SimSwp = np.zeros(total_points)
        self.SimOp = np.zeros(total_points)
        self.SimIn = np.zeros(total_points)
        self.SimAv = np.zeros(total_points)
        self.SimCom = np.zeros(total_points)
        self.SimDrugBound = np.zeros(total_points)
        epoch_voltages = np.zeros((NumEpchs + 1))
        epoch_end_times = np.zeros((NumEpchs + 1))
        epoch_voltages[0] = SwpSeq[(2, SwpNo)]
        epoch_end_times[0] = 0.0
        for e in range(1, (NumEpchs + 1)):
            epoch_voltages[e] = SwpSeq[((2 * e), SwpNo)]
            epoch_end_times[e] = (int(SwpSeq[(((2 * e) + 1), SwpNo)]) * sampint)
        self.vm = epoch_voltages[0]
        self.CurrVolt()
        self.pop = self.EquilOccup(self.vm)
        self._store_results_24(0, 0)
        current_time = 0.0
        store_idx = 1
        for epoch in range(1, (NumEpchs + 1)):
            self.vm = epoch_voltages[epoch]
            epoch_end_time = epoch_end_times[epoch]
            self._update_scalar_rates()
            self.CurrVolt()
            num_points = max(2, (int(((epoch_end_time - current_time) / sampint)) + 1))
            t_eval = np.linspace(current_time, epoch_end_time, num_points)
            if (len(t_eval) <= 1):
                current_time = epoch_end_time
                continue
            self._reusable_y0[:] = self.pop[:24]
            sol = solve_ivp(self.NowDerivs, [current_time, epoch_end_time], self._reusable_y0, method='LSODA', t_eval=t_eval, rtol=1e-06, atol=1e-08)
            self.full_sol_t = sol.t
            self.full_sol_y = sol.y
            if (sol.success and (len(sol.t) > 0)):
                batch_size = len(sol.t)
                end_idx = min((store_idx + batch_size), total_points)
                batch_indices = np.arange(store_idx, end_idx)
                actual_batch = len(batch_indices)
                if (actual_batch > 0):
                    batch_states = sol.y[:, :actual_batch].T
                    self._store_results_vectorized_24(batch_indices, batch_states, self.vm)
                    self.pop[:24] = sol.y[:, (- 1)]
                    store_idx = end_idx
            current_time = epoch_end_time
        self.time = np.arange(0, (total_points * sampint), sampint)[:total_points]
        return (sol.t, self.SimSwp)
    def _store_results_24(self, idx, t):
        vidx = np.searchsorted(self.vt, self.vm)
        vidx = np.clip(vidx, 0, (len(self.vt) - 1))
        open_prob_free = self.pop[5]
        open_prob_drug = self.pop[17]
        conducting_open_prob = open_prob_free
        current = (((conducting_open_prob * self.iscft[vidx]) * self.numchan) * self.current_scaling)
        self.SimSwp[idx] = current
        self.SimOp[idx] = (open_prob_free + open_prob_drug)
        self.SimIn[idx] = (np.sum(self.pop[6:12]) + np.sum(self.pop[18:24]))
        self.SimAv[idx] = np.sum(self.pop[:6])
        self.SimCom[idx] = self.vm
        self.SimDrugBound[idx] = np.sum(self.pop[12:24])
    def _store_results_vectorized_24(self, indices, batch_states, voltage):
        if (len(indices) == 0):
            return
        vidx = np.searchsorted(self.vt, voltage)
        vidx = np.clip(vidx, 0, (len(self.vt) - 1))
        current_factor = self.iscft[vidx]
        conducting_open_probs = batch_states[:, 5]
        total_open_probs = (batch_states[:, 5] + batch_states[:, 17])
        currents = (((conducting_open_probs * current_factor) * self.numchan) * self.current_scaling)
        inactivated = (np.sum(batch_states[:, 6:12], axis=1) + np.sum(batch_states[:, 18:24], axis=1))
        available = np.sum(batch_states[:, :6], axis=1)
        drug_bound = np.sum(batch_states[:, 12:24], axis=1)
        self.SimSwp[indices] = currents
        self.SimOp[indices] = total_open_probs
        self.SimIn[indices] = inactivated
        self.SimAv[indices] = available
        self.SimCom[indices] = voltage
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