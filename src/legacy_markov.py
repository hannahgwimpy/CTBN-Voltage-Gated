"""
Defines legacy Markov models for simulating ion channel kinetics.

This module provides classes for simulating voltage-gated ion channels using
Markov chain models. These models represent the channel as transitioning
between a discrete set of conformational states, with transition rates
that can be voltage-dependent and, in the 24 state case, drug-dependent.
"""
import numpy as np
from scipy.integrate import solve_ivp
class MarkovModel():
    def __init__(self):
        """Initializes the 13-state Markov model for a sodium channel.

        This constructor sets up the initial state of the model by:
        1. Defining the number of states (13) and the initial membrane potential.
        2. Calling `init_parameters` to set biophysical and simulation constants.
        3. Calling `init_waves` to prepare voltage-dependent rate arrays.
        4. Calling `stRatesVolt` to pre-compute the rate constants across a
           voltage range.
        5. Calling `CurrVolt` to pre-compute the GHK current factor.
        6. Calling `create_default_protocol` to set up a default voltage-clamp
           simulation protocol.
        """
        self.NumSwps = 0
        self.num_states = 13
        self.vm = (- 80)
        self.init_parameters()
        self.init_waves()
        self.stRatesVolt()
        self.CurrVolt()
        self.create_default_protocol()

    def init_parameters(self):
        """Initializes the model's biophysical and simulation parameters.

        This method sets up the fundamental constants and coefficients that define
        the behavior of the 13-state sodium channel Markov model. These parameters
        are used to calculate the voltage-dependent transition rates between states.

        The parameters are grouped as follows:

        Activation/Deactivation Rates (alpha/beta type):
        - alcoeff, alslp: Coefficients for forward activation rates (amt).
        - btcoeff, btslp: Coefficients for backward activation rates (bmt).

        Inactivation/Recovery Rates (gamma/delta type):
        - gmcoeff, gmslp: Coefficients for inactivation rates (gmt).
        - dlcoeff, dlslp: Coefficients for recovery from inactivation rates (dmt).

        Drug Binding/Unbinding Rates (placeholders for this model):
        - ConCoeff, CoffCoeff, ConSlp, CoffSlp: Base drug binding/unbinding rates.
        - OpOnCoeff, OpOffCoeff, OpOnSlp, OpOffSlp: Open-state drug binding rates.
        - ConHiCoeff, CoffHiCoeff: High-affinity drug binding rates.
        - alfac, btfac: Factors modifying drug rates based on channel state.

        Physical and Simulation Constants:
        - F, Rgc, Tkel: Faraday's constant, gas constant, and temperature in Kelvin.
        - Nao, Nai: Extracellular and intracellular sodium concentrations (mM).
        - PNasc: Sodium permeability scaling factor.
        - ClipRate: A ceiling value to prevent rates from becoming excessively large.
        - current_scaling: A final scaling factor for the computed current.
        - vm: Default holding potential (-80 mV).
        - _reusable_y0: A pre-allocated numpy array for ODE solver initial conditions.
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
        self.vm = (- 80)
        self.PNasc = 1e-05
        self._reusable_y0 = np.zeros(12)

    def init_waves(self):
        """Initializes numpy arrays for simulation and pre-computation.

        This method sets up several essential numpy arrays (referred to as "waves")
        that are used to store pre-computed values and simulation results.

        Specifically, it:
        1. Creates `self.vt`, a voltage vector from -200 to 200 mV, which serves
           as the domain for pre-calculating voltage-dependent rates.
        2. Initializes `self.pop`, `self.dstdt`, and `self._reusable_y0` as zero
           arrays to hold state populations and their derivatives.
        3. Initializes `self.iscft` to store the pre-computed GHK current factor.
        4. Calls `create_rate_waves` to initialize placeholder arrays for each
           transition rate.
        5. Calls `stRatesVolt` to populate these rate arrays with their
           voltage-dependent values.
        """
        self.vt = np.arange((- 200), 201)
        self.pop = np.zeros(13)
        self.dstdt = np.zeros(12)
        self._reusable_y0 = np.zeros(12)
        self.iscft = np.zeros_like(self.vt)
        self.create_rate_waves()
        self.stRatesVolt()

    def create_rate_waves(self):
        """Dynamically creates placeholder arrays for all transition rates.

        This method iterates through a predefined list of rate names, each
        representing a transition between two states in the Markov model (e.g.,
        'k12dis' for the transition from state 1 to 2).

        For each name, it dynamically creates a class attribute (e.g.,
        `self.k12dis_vec`) and initializes it as a numpy array of zeros with the
        same shape as the voltage vector `self.vt`. These arrays serve as
        placeholders that are later populated with voltage-dependent rate values
        by the `stRatesVolt` method.
        """
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis', 'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis', 'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis', 'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis', 'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis', 'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis', 'k98dis', 'k87dis']
        for name in rate_names:
            setattr(self, (name + '_vec'), np.zeros_like(self.vt, dtype=float))

    def stRatesVolt(self):
        """Calculates and stores all voltage-dependent transition rates.

        This method computes the values for all transition rate constants (e.g.,
        k12dis, k21dis) across the pre-defined voltage vector `self.vt`.
        The calculations are based on exponential functions of voltage, using
        coefficients and slope factors defined in `init_parameters` (e.g.,
        `alcoeff`, `alslp` for activation-like rates; `btcoeff`, `btslp` for
        deactivation-like rates; `gmcoeff`, `gmslp` for inactivation-like rates;
        `dlcoeff`, `dlslp` for recovery-like rates).

        The method performs the following steps:
        1. Ensures `ClipRate` and rate vectors (e.g., `k12dis_vec`) are initialized.
           If rate vectors are not present, `create_rate_waves` is called.
        2. Calculates primary voltage-dependent terms: `amt`, `bmt`, `gmt`, `dmt`,
           and drug-related terms `konlo`, `kofflo`, `konop`, `koffop` (though
           drug effects are minimal in this base model as `dph` is fixed to 1).
        3. Populates each rate vector (e.g., `self.k12dis_vec`) by applying these
           terms, often with multipliers (e.g., `4 * amt`).
        4. Applies `np.minimum` with `self.ClipRate` to each calculated rate to
           prevent them from exceeding a maximum value, ensuring numerical stability.
        5. Handles specific rate calculations that depend on other rates, such as
           those involving `alfac` and `btfac` (related to drug binding affinity
           changes with channel state) and the microscopically reversible rate
           `k1211dis_vec`.
        6. Calls `self._update_scalar_rates()` at the end to set the scalar rate
           constants to their values at the current `self.vm`.

        The resulting rate vectors (e.g., `self.k12dis_vec`) store the rate
        constants for each transition at each voltage point in `self.vt`.
        """
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
        """Updates scalar rate constants to values at the current membrane potential.

        This helper method is typically called after `stRatesVolt` (which populates
        the voltage-dependent rate vectors like `self.k12dis_vec`) or when
        `self.vm` (membrane potential) is changed.

        It performs the following for each rate constant defined in `rate_names`:
        1. Finds the index (`vidx`) in the `self.vt` voltage vector that is
           closest to the current `self.vm`.
        2. Retrieves the corresponding pre-computed rate array (e.g., `self.k12dis_vec`).
        3. Sets the scalar rate attribute (e.g., `self.k12dis`) to the value from
           the rate array at `vidx`.
        4. Includes safety checks: if the vector attribute doesn't exist or is not a
           properly sized numpy array, the scalar rate is set to 0.0.

        This ensures that the scalar rate constants used in calculations (e.g., by
        `NowDerivs`) reflect the rates at the specific current membrane potential.
        """
        vidx = np.argmin(np.abs((self.vt - self.vm)))
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis', 'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis', 'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis', 'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis', 'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis', 'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis', 'k98dis', 'k87dis']
        for name in rate_names:
            vec_name = (name + '_vec')
            if hasattr(self, vec_name):
                vec_array = getattr(self, vec_name)
                if (isinstance(vec_array, np.ndarray) and (len(vec_array) > vidx)):
                    setattr(self, name, vec_array[vidx])
                else:
                    setattr(self, name, 0.0) # If vec_array is not valid or too short
            else:
                setattr(self, name, 0.0) # If vec_name attribute doesn't exist

    def CurrVolt(self):
        """Pre-computes the Goldman-Hodgkin-Katz (GHK) current scaling factor.

        This method calculates `self.iscft` (ion-specific current factor times)
        across the voltage vector `self.vt`. This factor is used in conjunction
        with the open probability of the channel to determine the sodium current.

        The GHK flux equation is used:
        I_Na = P_Na * F^2 * V / (R*T) * ([Na_i] - [Na_o]*exp(-FV/RT)) / (1 - exp(-FV/RT))

        `self.iscft` stores: P_Na * F^2 * V / (R*T) * ([Na_i] - [Na_o]*exp(-FV/RT)) / (1 - exp(-FV/RT))
        divided by V (effectively P_Na * GHK_permeability_term).
        When multiplied by the sum of open state populations and V, it yields current.
        However, the current is typically calculated as `self.iscft * sum(open_states)`,
        implying `iscft` stores P_Na * GHK_permeability_term * V.

        The calculation handles voltages near zero separately to avoid division by
        zero, using L'Hôpital's rule for the GHK equation limit at V=0:
        Limit (V -> 0) of GHK = P_Na * F^2/(R*T) * ([Na_i] - [Na_o])

        Args:
            None

        Updates:
            self.iscft (numpy.ndarray): Array of GHK current scaling factors for each
                                    voltage in `self.vt`.
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
        """Calculates equilibrium (steady-state) occupancies of the 12 model states.

        This method determines the probability distribution of the channel across its
        12 independent states (C1-C6, O, I1-I6) when it has reached equilibrium
        at a given membrane potential `vm`. The occupancy of the 13th state (O_I, drug-bound open)
        is implicitly zero in this drug-free model version, or can be calculated as
        1 - sum(returned 12 states), though the model structure here implies 12 primary states
        derived from a 13-state scheme where one is dependent.

        The calculation relies on the principle of detailed balance, where the ratio
        of occupancies of two connected states at equilibrium is equal to the ratio
        of the forward to backward rate constants between them.

        Args:
            vm (float): The membrane potential (in mV) at which to calculate
                        equilibrium occupancies.

        Returns:
            numpy.ndarray: A 1D array of 12 floats representing the equilibrium
                           occupancies of states C1-C6, O (indices 0-5) and
                           I1-I6 (indices 6-11) respectively.

        Internal Steps:
        1. Updates `self.vm` and ensures all voltage-dependent rate constants
           (`self.k*dis`) are current for this `vm` by calling `stRatesVolt`
           and `_update_scalar_rates`.
        2. Defines a `safe_div` helper to avoid division by zero when calculating
           ratios of rate constants (e.g., k_forward/k_backward).
        3. Calculates elementary ratios (`du1` to `du12`) representing
           P(state_j)/P(state_i) for adjacent states i, j along the main
           activation (C1..O) and inactivation (I1..I6) pathways.
        4. Computes cumulative product terms based on these ratios to express each
           state's occupancy relative to the first state in its pathway (C1 or I1).
        5. Sums all relative occupancies to get a normalization factor (`dusum`).
        6. Divides individual relative occupancies by `dusum` to get absolute
           probabilities for each of the 12 states.
        7. Handles a near-zero `dusum` (e.g., at very negative potentials) by assigning
           a default distribution (e.g., 0.98 to C1, 0.02 to C2) to prevent errors.
        8. Ensures no NaNs are returned.
        """
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
        """Calculates the time derivatives of the 12 state occupancies (d(state)/dt).

        This method defines the system of ordinary differential equations (ODEs) for
        the 12 independent states of the Markov model (C1-C6, O, I1-I6). It is intended
        to be used by an ODE solver (e.g., `scipy.integrate.solve_ivp`) to simulate
        the time evolution of state probabilities at a constant membrane potential `self.vm`.

        The state vector `y` represents the current occupancies of the 12 states.
        The time `t` is usually provided by the ODE solver but is not explicitly used
        in this method as rate constants are assumed constant for a given `self.vm`.

        Args:
            t (float): Current time point (typically from ODE solver, not used directly).
            y (numpy.ndarray): A 1D array of 12 floats representing the current
                               occupancies of states C1-C6, O (indices 0-5) and
                               I1-I6 (indices 6-11).

        Returns:
            numpy.ndarray: A 1D array of 12 floats representing the time derivatives
                           (d(state)/dt) for each of the 12 states.

        Internal Steps:
        1. Retrieves pre-calculated voltage-dependent rate constants (`k*dis_vec`)
           for the current `self.vm`.
        2. Performs a sanity check on the input state vector `y` for NaNs or Infs,
           returning zeros if found to prevent solver errors.
        3. Constructs the 12x12 transition rate matrix `Q`:
           - Off-diagonal elements `Q[i, j]` (i != j): Rate constant from state `j` to state `i`.
             (Note: Standard Q-matrix definition often has `Q[i,j]` as rate from `i` to `j`.
              The implementation here `dstdt[i] += (Q[(i, j)] * st[j])` implies
              `Q[i,j]` is rate from `j` to `i` if `dstdt[i]` is `dy_i/dt`.)
           - Diagonal elements `Q[i, i]`: Negative sum of all rate constants *leaving* state `i`.
             (i.e., `Q[i,i] = - sum(rates from i to j for all j != i)`).
        4. Calculates the derivatives `dstdt` using the matrix equation: `d(st)/dt = Q * st`,
           where `st` is the current state vector `y`.
        5. Performs a final sanity check on the calculated `dstdt` for NaNs or Infs.
        """
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
        """Simulates the channel's response to a defined voltage-clamp sweep protocol.

        This method executes a single sweep, as defined by `SwpNo` in the
        `self.SwpSeq` array, which specifies the sequence of voltage epochs
        (voltages and durations). It simulates the model's behavior over time,
        calculating state occupancies and resulting ionic current.

        Args:
            SwpNo (int): The index of the sweep protocol to execute, as defined
                         in `self.SwpSeq`.

        Returns:
            tuple: A tuple containing:
                - sol.t (numpy.ndarray): Time points from the ODE solver for the
                                         last epoch.
                - self.SimSwp (numpy.ndarray): Array of simulated total current
                                               values at each time point of the sweep.

        Raises:
            ValueError: If `SwpNo` is invalid or if the protocol definition
                        in `self.SwpSeq` is malformed.

        Key Operations:
        1. Parses the voltage protocol (epochs, voltages, durations) from
           `self.SwpSeq` for the given `SwpNo`.
        2. Initializes simulation arrays (e.g., `self.SimSwp`, `self.SimOp`).
        3. Sets initial state occupancies (`self.pop`) to equilibrium at the
           first epoch's voltage (holding potential) using `self.EquilOccup()`.
        4. Stores the initial state. Note: Uses `self._store_results(0, 0)`, which
           might be a legacy call; `_store_results_vectorized` is used later.
           This behavior should be reviewed.
        5. Iterates through each subsequent epoch:
           a. Updates `self.vm`, rate constants (`_update_scalar_rates`), and
              GHK factors (`CurrVolt`).
           b. Solves the ODE system `self.NowDerivs` using `scipy.integrate.solve_ivp`
              (LSODA method, rtol=1e-6, atol=1e-8) to get state occupancies over time.
           c. Stores results (current, open probability, etc.) using
              `self._store_results_vectorized`.
           d. Updates `self.pop` to the state occupancies at the end of the epoch.
        6. Populates `self.time` array with the time points of the simulation.
        7. Updates instance attributes like `self.SimSwp`, `self.SimOp`, `self.SimIn`,
           `self.SimAv`, `self.SimCom` with simulation results.
        8. Uses a hardcoded sampling interval (`sampint`) of 0.005 ms.
        """
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
                    self.pop[:12] = sol.y[:, -1]
                    store_idx = end_idx
            current_time = epoch_end_time
        self.time = np.arange(0, (total_points * sampint), sampint)[:total_points]
        return (sol.t, self.SimSwp)

    def _store_results(self, idx, t):
        """Stores simulation results for a single time point/state.

        This helper method is typically used to store the initial state of the
        simulation (e.g., at t=0 after equilibrium) before starting the main
        epoch-by-epoch simulation loop in `Sweep`.

        It calculates the macroscopic current based on the open probability (`self.pop[5]`),
        the GHK current factor (`self.iscft`), number of channels (`self.numchan`),
        and a scaling factor (`self.current_scaling`). It also stores the open
        probability, sum of inactivated states, sum of available (non-inactivated)
        states, and the command voltage at the given index `idx`.

        Args:
            idx (int): The index in the simulation arrays (e.g., `self.SimSwp`)
                       where the results should be stored.
            t (float): The time point corresponding to this state. Note: this argument
                       is present in the method signature but not explicitly used in
                       the current implementation.

        Updates:
            self.SimSwp (numpy.ndarray): Updated with the calculated current at `idx`.
            self.SimOp (numpy.ndarray): Updated with the open probability at `idx`.
            self.SimIn (numpy.ndarray): Updated with the sum of inactivated state
                                        probabilities at `idx`.
            self.SimAv (numpy.ndarray): Updated with the sum of available (non-inactivated)
                                        state probabilities at `idx`.
            self.SimCom (numpy.ndarray): Updated with the command voltage `self.vm` at `idx`.
        """
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
        """Stores simulation results for a batch of states at specified time indices.

        This helper method is used within simulation loops (e.g., `Sweep`) to efficiently
        store calculated values from a batch of state vectors corresponding to multiple
        time points or parallel simulations, all at a single command voltage.

        Args:
            indices (numpy.ndarray): Array of integer indices indicating where to store
                                     the results in the `Sim*` arrays.
            batch_states (numpy.ndarray): A 2D array where each row is a state vector
                                        (populations of the 13 states) for a given index.
            voltage (float): The command voltage (in mV) at which these states were observed.

        Updates:
            self.SimSwp (numpy.ndarray): Stores the calculated total current.
            self.SimOp (numpy.ndarray): Stores the probability of the channel being in the open state (state O, index 5).
            self.SimIn (numpy.ndarray): Stores the sum of probabilities of inactivated states (states I1-I6, indices 6-11).
            self.SimAv (numpy.ndarray): Stores the sum of probabilities of available (non-inactivated) states (states C1-C6, O, indices 0-5).
            self.SimCom (numpy.ndarray): Stores the command voltage.

        The method performs the following:
        1. Returns early if `indices` is empty.
        2. Finds the index (`vidx`) in `self.vt` corresponding to the `voltage` to get the correct `self.iscft` value.
        3. Extracts open state probabilities (state O, index 5) from `batch_states`.
        4. Calculates currents using `open_probs`, `current_factor` from `self.iscft`, `self.numchan`, and `self.current_scaling`.
        5. Calculates total inactivated probability (sum of populations in states I1-I6, indices 6 to 11, which is `batch_states[:, 6:]`).
        6. Calculates total available probability (sum of populations in states C1-C6 and O, indices 0 to 5, which is `batch_states[:, :6]`)
        7. Stores these calculated values into the `Sim*` arrays at the specified `indices`.
        """
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
        """Creates a default multi-step voltage clamp protocol.

        This protocol is designed for characterizing channel activation and basic
        voltage dependence. It consists of a series of sweeps, each with:
        1. A holding period at `holding_potential`.
        2. A test pulse to one of the `target_voltages`.
        3. A brief tail pulse back to `holding_potential`.

        The durations are specified in milliseconds and converted to sample points
        assuming a sampling interval of 0.005 ms (200 kHz sampling rate).

        Args:
            target_voltages (list or numpy.ndarray, optional):
                A list of test potentials (in mV) to apply. Defaults to
                `[30, 0, -20, -30, -40, -50, -60]`.
            holding_potential (float, optional): The potential (in mV) for the holding
                and tail periods. Defaults to -80 mV.
            holding_duration (float, optional): Duration (in ms) of the initial holding
                period. Defaults to 98 ms.
            test_duration (float, optional): Duration (in ms) of the test pulse period.
                Defaults to 200 ms.
            tail_duration (float, optional): Duration (in ms) of the final tail pulse
                period. Defaults to 2 ms.

        Updates:
            self.BsNm (str): Base name for the sweep sequence, set to 'MultiStepKeyVoltages'.
            self.NumSwps (int): Number of sweeps, determined by the length of `target_voltages`.
            self.SwpSeq (numpy.ndarray): An 8xNumSwps array defining the protocol segments.
                Row 0: Number of segments in the sweep (always 3: hold, test, tail).
                Row 1: Not explicitly set, likely 0 (voltage for segment 1, unused if segment is duration-based).
                Row 2: Voltage for segment 1 (holding_potential).
                Row 3: Duration (samples) of segment 1 (holding_samples).
                Row 4: Voltage for segment 2 (target_voltages[sweep_index]).
                Row 5: End time (samples) of segment 2 (holding_samples + test_samples).
                Row 6: Voltage for segment 3 (holding_potential).
                Row 7: End time (samples) of segment 3 (total_samples).
            self.SwpSeqMultiStepKeyVoltages (numpy.ndarray): A copy of `self.SwpSeq` stored
                dynamically using `self.BsNm`.

        Calls:
            self.CurrVolt(): Recalculates GHK current factors, possibly redundant if called
                            during initialization but ensures factors are up-to-date if
                            parameters like temperature or concentrations were changed.
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
        """Creates a voltage-clamp protocol to measure voltage-dependent inactivation.

        This protocol consists of a single sweep with four segments:
        1. Initial Holding: At -80 mV for a fixed 200 ms to stabilize channels.
        2. Inactivating Pre-pulse: To `inactivating_voltage` for `inactivating_duration`
           to allow channels to inactivate to varying degrees.
        3. Test Pulse: To `test_voltage` for a fixed 5 ms to measure available current
           (channels not inactivated by the pre-pulse).
        4. Recovery Period: At -80 mV for `recovery_duration` to allow channels to
           recover from inactivation.

        Args:
            inactivating_voltage (float, optional): The voltage (in mV) of the
                inactivating pre-pulse. Defaults to -20 mV.
            test_voltage (float, optional): The voltage (in mV) of the test pulse.
                Defaults to 0 mV.
            inactivating_duration (float, optional): Duration (in ms) of the
                inactivating pre-pulse. Defaults to 2000 ms.
            recovery_duration (float, optional): Duration (in ms) of the final
                recovery period. Defaults to 100 ms.

        Updates:
            self.BsNm (str): Base name for the sweep sequence, set to 'InactivationProtocol'.
            self.NumSwps (int): Number of sweeps, set to 1 for this protocol.
            self.SwpSeq (numpy.ndarray): A 10x1 array defining the 4-segment protocol.
                Row 0: Number of segments (4).
                Row 2: Voltage for segment 1 (-80 mV).
                Row 3: Duration (samples) of segment 1.
                Row 4: Voltage for segment 2 (`inactivating_voltage`).
                Row 5: End time (samples) of segment 2.
                Row 6: Voltage for segment 3 (`test_voltage`).
                Row 7: End time (samples) of segment 3.
                Row 8: Voltage for segment 4 (-80 mV).
                Row 9: End time (samples) of segment 4.
            self.SwpSeqInactivationProtocol (numpy.ndarray): A copy of `self.SwpSeq`.

        Calls:
            self.CurrVolt(): To update GHK current factors.
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
        """Creates a protocol to measure the time course of recovery from inactivation.

        This protocol generates a family of sweeps. Each sweep has five segments:
        1. Initial Holding: At `holding_potential` for `holding_duration`.
        2. Inactivating Pulse: To `inactivating_voltage` for `inactivating_duration`
           to inactivate the channels.
        3. Recovery Interval: At `holding_potential` for a variable duration specified
           by `target_recovery_times`. This is the key varying parameter across sweeps.
        4. Test Pulse: To `test_voltage` for `test_duration` to measure current from
           channels that have recovered during the interval.
        5. Tail Pulse: To `holding_potential` for `tail_duration`.

        Args:
            target_recovery_times (list or numpy.ndarray, optional):
                A list of recovery interval durations (in ms). Defaults to
                `[1, 3, 10, 30, 100, 300, 1000]`.
            holding_potential (float, optional): Voltage (mV) for holding, recovery,
                and tail segments. Defaults to -80 mV.
            inactivating_voltage (float, optional): Voltage (mV) for the inactivating
                pulse. Defaults to -20 mV.
            test_voltage (float, optional): Voltage (mV) for the test pulse.
                Defaults to 0 mV.
            holding_duration (float, optional): Duration (ms) of the initial holding
                period. Defaults to 200 ms.
            inactivating_duration (float, optional): Duration (ms) of the inactivating
                pulse. Defaults to 2000 ms.
            test_duration (float, optional): Duration (ms) of the test pulse.
                Defaults to 20 ms.
            tail_duration (float, optional): Duration (ms) of the final tail pulse.
                Defaults to 100 ms.

        Updates:
            self.BsNm (str): Base name, 'RecoveryFromInactivation'.
            self.NumSwps (int): Number of sweeps, from `len(target_recovery_times)`.
            self.SwpSeq (numpy.ndarray): A 12xNumSwps array defining the 5-segment protocol.
                Row 0: Number of segments (5).
                Row 2: Voltage for segment 1 (`holding_potential`).
                Row 3: Duration (samples) of segment 1.
                Row 4: Voltage for segment 2 (`inactivating_voltage`).
                Row 5: End time (samples) of segment 2.
                Row 6: Voltage for segment 3 (`holding_potential` - recovery interval).
                Row 7: End time (samples) of segment 3 (varies with `recovery_samples`).
                Row 8: Voltage for segment 4 (`test_voltage`).
                Row 9: End time (samples) of segment 4.
                Row 10: Voltage for segment 5 (`holding_potential` - tail).
                Row 11: End time (samples) of segment 5.
            self.SwpSeqRecoveryFromInactivation (numpy.ndarray): Copy of `self.SwpSeq`.

        Calls:
            self.CurrVolt(): To update GHK current factors.
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
        """Creates a protocol to determine steady-state inactivation (availability).

        This protocol generates a family of sweeps. Each sweep has four segments:
        1. Initial Holding: At `holding_potential` (default -120 mV) for a fixed
           200 ms to ensure channels start from a fully recovered state.
        2. Conditioning Pre-pulse: To one of the `test_voltages` (which are actually
           conditioning potentials here) for `prepulse_duration` (default 2000 ms).
           This long pulse allows channels to reach steady-state inactivation at that
           potential.
        3. Test Pulse: To `test_pulse_voltage` (default 0 mV) for `test_pulse_duration`
           (default 5 ms) to measure the fraction of channels still available (not inactivated).
        4. Recovery Period: At `holding_potential` for `recovery_duration`.

        Args:
            test_voltages (list or numpy.ndarray, optional):
                A list of conditioning pre-pulse potentials (in mV). Defaults to
                `np.arange(-120, -15, 5)` (i.e., -120, -115, ..., -20 mV).
            holding_potential (float, optional): Voltage (mV) for the initial holding
                and final recovery segments. Defaults to -120 mV.
            prepulse_duration (float, optional): Duration (ms) of the conditioning
                pre-pulse. Defaults to 2000 ms.
            test_pulse_voltage (float, optional): Voltage (mV) of the test pulse used
                to assess availability. Defaults to 0 mV.
            test_pulse_duration (float, optional): Duration (ms) of the test pulse.
                Defaults to 5 ms.
            recovery_duration (float, optional): Duration (ms) of the final recovery
                period. Defaults to 100 ms.

        Updates:
            self.BsNm (str): Base name, 'SteadyStateInactivation'.
            self.NumSwps (int): Number of sweeps, from `len(test_voltages)`.
            self.SwpSeq (numpy.ndarray): A 10xNumSwps array defining the 4-segment protocol.
                Row 0: Number of segments (4).
                Row 2: Voltage for segment 1 (`holding_potential`).
                Row 3: Duration (samples) of segment 1.
                Row 4: Voltage for segment 2 (varies with `test_voltages`).
                Row 5: End time (samples) of segment 2.
                Row 6: Voltage for segment 3 (`test_pulse_voltage`).
                Row 7: End time (samples) of segment 3.
                Row 8: Voltage for segment 4 (`holding_potential`).
                Row 9: End time (samples) of segment 4.
            self.SwpSeqSteadyStateInactivation (numpy.ndarray): Copy of `self.SwpSeq`.

        Calls:
            self.CurrVolt(): To update GHK current factors.
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
class AnticonvulsantMarkovModel(MarkovModel):
    def __init__(self, drug_concentration=0.0, drug_type='DPH'):
        """Initializes the 25-state Markov model for a sodium channel with drug interactions.

        This constructor extends the base `MarkovModel` to incorporate the effects
        of a drug, such as an anticonvulsant. It expands the state space to 25 states
        to account for drug-bound channel states (e.g., drug bound to open, closed,
        or inactivated states).

        Args:
            drug_concentration (float, optional): The concentration of the drug
                in micromolar (uM). Defaults to 0.0 (no drug).
            drug_type (str, optional): A string identifier for the drug type, used
                to select appropriate drug binding/unbinding parameters. Examples
                might include 'DPH' (phenytoin), 'CBZ' (carbamazepine), etc.
                Defaults to 'DPH'. The value is converted to uppercase.

        The initialization process involves:
        1. Setting the number of states to 25.
        2. Storing `drug_concentration` and `drug_type`.
        3. Setting the initial membrane potential (`vm`) to -80 mV.
        4. Calling `self.init_parameters()`: This method is overridden from
           `MarkovModel` to include or modify parameters related to drug binding
           (e.g., on/off rates for different channel states, affinity constants).
        5. Calling `self.init_waves()`: Also potentially overridden to handle the
           expanded state space and additional rate constants for drug interactions.
        6. Calling `self._update_drug_rates()`: A new method specific to this class
           that calculates and updates drug-related transition rates based on the
           `drug_concentration` and `drug_type`.
        7. Calling `self.CurrVolt()`: To pre-compute GHK current factors, likely
           inherited or similar to `MarkovModel` but operating on the expanded model.
        8. Calling `self.create_default_protocol()`: To set up a default voltage-clamp
           simulation protocol. This might be overridden to use different parameters
           or to better highlight drug effects.
        9. Calculating initial equilibrium state occupancies (`self.pop`) using
           `self.EquilOccup(self.vm)`, which is overridden to handle the 25-state system.
        """
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
        """Sets or changes the type of drug being modeled and updates parameters.

        This method allows for changing the simulated drug type (e.g., from
        'DPH' to 'CBZ') after the model has been initialized. It triggers a
        re-initialization of drug-specific parameters and recalculates equilibrium
        conditions.

        Args:
            drug_type (str): A string identifier for the new drug type. This
                identifier is used to fetch the appropriate set of drug interaction
                parameters (e.g., binding affinities, on/off rates for different
                channel states). The value is converted to uppercase.

        Updates:
            self.drug_type (str): Updated to the new `drug_type`.
            Calls `self.init_parameters()`: To re-load parameters specific to the
                new `drug_type`.
            Calls `self._update_drug_rates()`: To recalculate drug binding/unbinding
                rates based on the new parameters and existing `drug_concentration`.
            self.pop (numpy.ndarray): Recalculated equilibrium state occupancies for
                the current `self.vm` using the new drug parameters.
        """
        self.drug_type = drug_type.upper()
        self.init_parameters()
        self._update_drug_rates()
        self.pop = self.EquilOccup(self.vm)

    def set_drug_concentration(self, drug_concentration):
        """Sets or changes the drug concentration and updates dependent rates and equilibrium.

        This method allows for adjusting the concentration of the currently selected
        drug after the model has been initialized. It triggers an update of
        concentration-dependent drug binding/unbinding rates and recalculates
        equilibrium conditions.

        Args:
            drug_concentration (float): The new concentration of the drug in
                micromolar (uM).

        Updates:
            self.drug_concentration (float): Updated to the new `drug_concentration`.
            Calls `self._update_drug_rates()`: To recalculate drug binding/unbinding
                rates based on the new `drug_concentration` and existing drug parameters.
            self.pop (numpy.ndarray): Recalculated equilibrium state occupancies for
                the current `self.vm` using the new drug concentration.
        """
        self.drug_concentration = drug_concentration
        self._update_drug_rates()
        self.pop = self.EquilOccup(self.vm)

    def init_parameters(self):
        """Initializes biophysical and drug-specific parameters for the 25-state model.

        This method overrides `MarkovModel.init_parameters` to set up parameters
        for a sodium channel model that incorporates drug interactions. It initializes
        the intrinsic channel gating parameters (activation, deactivation, inactivation,
        recovery coefficients and slopes) and then loads drug-specific parameters
        based on `self.drug_type`.

        Key drug-specific parameters defined here include:
        - `KI_inactivated`: Drug affinity (dissociation constant) for the inactivated state.
        - `recovery_tau`: Time constant for recovery from drug block (used to derive k_off).
        - `k_off_base`, `k_off_scaling`: Base off-rate for drug dissociation and its scaling factor.
        - `k_off`: Calculated effective drug dissociation rate.
        - `KR_resting`: Drug affinity for the resting state (often much lower than for inactivated).
        - `k_on_inactivated_base`: Base on-rate for drug binding to the inactivated state.
        - `k_on_resting_base`: Base on-rate for drug binding to the resting state.

        The method performs the following steps:
        1. Sets hardcoded coefficients and slopes for intrinsic channel gating (e.g.,
           `alcoeff`, `alslp`, `btcoeff`, `btslp`, `gmcoeff`, `gmslp`, `dlcoeff`, `dlslp`).
           These are typically inherited or similar to the base `MarkovModel`.
        2. Defines a dictionary `self.drug_params` containing parameters for known
           drug types (e.g., 'CBZ', 'LTG', 'DPH').
        3. Selects the appropriate parameter set from `self.drug_params` based on
           `self.drug_type`. If the drug type is unknown, it defaults to 'DPH'
           parameters and issues a warning.
        4. Calculates and sets instance attributes for the selected drug's kinetic
           properties (e.g., `self.KI_inactivated`, `self.k_off`).
        5. Sets physical constants (Faraday's constant, gas constant, temperature in
           Kelvin - note: `self.Tkel` is set to 298K here), ion concentrations,
           `ClipRate`, `current_scaling`, and `PNasc`.
        6. Initializes `self._reusable_y0` as a NumPy array for the 24 independent
           states of the ODE system.
        7. Calls `self._update_drug_rates()` to calculate the actual concentration-dependent
           drug binding and unbinding rates using the newly set parameters and the
           current `self.drug_concentration`.

        Updates:
            Numerous instance attributes related to channel gating and drug interaction
            parameters (e.g., `alcoeff`, `KI_inactivated`, `k_off`, `Tkel`).
            self._reusable_y0 (numpy.ndarray): Initialized for 24 states.
            Calls `self._update_drug_rates()`.
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
        """Updates the effective drug binding and unbinding rates.

        This private helper method calculates the actual, concentration-dependent
        on-rates for drug binding and sets the off-rates. It is called during
        initialization (`init_parameters`) and whenever the drug type or
        concentration is changed (`set_drug_type`, `set_drug_concentration`).

        The current implementation makes specific assumptions:
        - Drug interaction with resting states: `k_on_resting` (on-rate to resting)
          and `k_off_resting` (off-rate from resting) are set to 0. This implies
          that the model assumes negligible drug binding to, or unbinding from,
          the resting states of the channel, or these pathways are not explicitly modeled.
        - Drug interaction with inactivated states:
            - `k_on_inactivated`: The on-rate for drug binding to inactivated states
              is calculated as `self.k_on_inactivated_base * self.drug_concentration`.
              This represents a pseudo-first-order rate constant for a bimolecular
              reaction where the drug concentration is much higher than channel
              concentration or is considered constant.
            - `k_off_inactivated`: The off-rate for drug unbinding from inactivated
              states is set to `self.k_off`, which is the effective (concentration-
              independent) dissociation rate constant, typically derived from
              `self.KI_inactivated` and `self.k_on_inactivated_base` or directly
              from experimental data like `recovery_tau` in `init_parameters`.

        Updates:
            self.k_on_resting (float): Set to 0.
            self.k_on_inactivated (float): Calculated effective on-rate to inactivated states.
            self.k_off_resting (float): Set to 0.
            self.k_off_inactivated (float): Set to the effective off-rate from inactivated states (`self.k_off`).
        """
        self.k_on_resting = 0
        self.k_on_inactivated = (self.k_on_inactivated_base * self.drug_concentration)
        self.k_off_resting = 0
        self.k_off_inactivated = self.k_off

    def init_waves(self):
        """Initializes NumPy arrays for simulation and pre-computation in the 25-state model.

        This method overrides `MarkovModel.init_waves` to set up essential NumPy
        arrays (referred to as "waves") tailored for the 25-state model that
        includes drug interactions. These arrays store pre-computed values,
        simulation results, and intermediate calculations.

        Specifically, it initializes:
        - self.vt (numpy.ndarray): A voltage vector ranging from -200 mV to +200 mV
          in 1 mV steps. This vector defines the voltage points at which rate
          constants and GHK factors are pre-computed.
        - self.pop (numpy.ndarray): A zero-initialized array of size 24, intended to
          store the occupancies of the 24 independent states of the model.
        - self.dstdt (numpy.ndarray): A zero-initialized array of size 24, used to
          store the time derivatives (d(state)/dt) of the state occupancies during
          ODE solving.
        - self._reusable_y0 (numpy.ndarray): A zero-initialized array of size 24,
          pre-allocated for use as the initial condition vector `y0` in the ODE solver.
        - self.iscft (numpy.ndarray): A zero-initialized array with the same shape as
          `self.vt`, used to store the pre-computed Goldman-Hodgkin-Katz (GHK)
          current scaling factors across the voltage range.

        After initializing these arrays, it calls:
        1. `self.create_rate_waves()`: An overridden method that dynamically creates
           placeholder arrays for all intrinsic and drug-related transition rates specific
           to the 25-state model.
        2. `self.stRatesVolt()`: An overridden method that populates these rate arrays
           with their voltage-dependent values across `self.vt`.

        Updates:
            self.vt, self.pop, self.dstdt, self._reusable_y0, self.iscft (numpy.ndarray).
            Calls `self.create_rate_waves()` and `self.stRatesVolt()`.
        """
        self.vt = np.arange((- 200), 201)
        self.pop = np.zeros(24)
        self.dstdt = np.zeros(24)
        self._reusable_y0 = np.zeros(24)
        self.iscft = np.zeros_like(self.vt)
        self.create_rate_waves()
        self.stRatesVolt()

    def create_rate_waves(self):
        """Dynamically creates placeholder arrays for intrinsic transition rates.

        This method overrides `MarkovModel.create_rate_waves` to prepare for the
        expanded 25-state model, though it primarily lists rates corresponding to
        the core 12-state channel gating mechanism. Drug interaction rates are typically
        handled by `_update_drug_rates` and their voltage dependence (if any, beyond
        state preference) would be incorporated in `stRatesVolt` or `NowDerivs`.

        It iterates through a predefined list of `rate_names`, each representing an
        intrinsic transition between two states of the sodium channel model (e.g.,
        'k12dis' for C1->C2, 'k21dis' for C2->C1, 'k17dis' for C1->I1, etc.).

        For each `rate_name` in the list, this method:
        1. Constructs a new attribute name by appending '_vec' (e.g., 'k12dis_vec').
        2. Creates a NumPy array of zeros with the same shape and data type (float)
           as `self.vt` (the voltage vector).
        3. Assigns this new array to the dynamically created attribute on the instance
           (e.g., `self.k12dis_vec`).

        These arrays serve as placeholders that are subsequently populated with their
        respective voltage-dependent rate constant values by the `stRatesVolt` method.

        Updates:
            Dynamically creates numerous instance attributes, each being a NumPy array
            (e.g., `self.k12dis_vec`, `self.k23dis_vec`, etc.), initialized to zeros.
            The list of rate names includes transitions for closed states (C1-C6),
            open state (O, implicitly via transitions like k56dis to O and k65dis from O),
            and inactivated states (I1-I6).
        """
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis', 'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis', 'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis', 'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis', 'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis', 'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis', 'k98dis', 'k87dis']
        for name in rate_names:
            setattr(self, (name + '_vec'), np.zeros_like(self.vt, dtype=float))

    def stRatesVolt(self):
        """Calculates and stores all voltage-dependent intrinsic transition rates.

        This method overrides `MarkovModel.stRatesVolt` to compute the values for all
        intrinsic transition rate constants (e.g., k12dis, k21dis) across the pre-defined
        voltage vector `self.vt`. While this model includes drug interactions, this
        method primarily focuses on the voltage dependence of the channel's own gating
        transitions. Drug concentration effects on on-rates are typically applied in
        `_update_drug_rates`, while this method may incorporate voltage-dependent
        components of drug binding/unbinding or allosteric effects of drugs on
        intrinsic rates via factors like `alfac` and `btfac`.

        The method performs the following steps:
        1. Ensures `ClipRate` is initialized (defaults to 1000 if not set) to prevent
           rates from becoming excessively large.
        2. Ensures rate vectors (e.g., `k12dis_vec`) are initialized by calling
           `create_rate_waves()` if necessary.
        3. Defines core voltage-dependent functions based on model parameters:
           - `amt`: Activation-like rates (alpha_m type) using `alcoeff`, `alslp`.
           - `bmt`: Deactivation-like rates (beta_m type) using `btcoeff`, `btslp`.
           - `gmt`: Inactivation-like rates (gamma type, e.g., alpha_h) using `gmcoeff`, `gmslp`.
           - `dmt`: Recovery-like rates (delta type, e.g., beta_h) using `dlcoeff`, `dlslp`.
           - `konlo`, `kofflo`: Base voltage-dependent rates potentially related to low-affinity
             drug binding or transitions to/from certain drug-bound states.
           - `konop`, `koffop`: Base voltage-dependent rates potentially related to open-state
             drug binding or transitions.
        4. Populates each rate vector (e.g., `self.k12dis_vec`) by applying the appropriate
           voltage-dependent function (`amt`, `bmt`, etc.), often with stoichiometric factors
           (e.g., 4*amt for k12dis) and allosteric factors (`alfac`, `btfac`) for transitions
           involving drug-modified states or pathways.
        5. All calculated rates are capped at `self.ClipRate`.
        6. A specific calculation for `k1211dis_vec` involves other pre-calculated rates,
           suggesting a detailed balance constraint or specific kinetic scheme detail.
        7. Calls `self._update_scalar_rates()` to update the scalar (single-value)
           rate constants (e.g., `self.k12dis`) to their values at the current `self.vm`.

        The resulting rate vectors (e.g., `self.k12dis_vec`) store the rate constants
        for each intrinsic transition at each voltage point in `self.vt`.

        Updates:
            Populates all `_vec` attributes (e.g., `self.k12dis_vec`, `self.k71dis_vec`)
            with voltage-dependent rate values.
            Calls `self._update_scalar_rates()`.
        """
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
        """Updates scalar rate constants to values at the current membrane potential.

        This method is typically called after `stRatesVolt` (which populates the
        voltage-dependent rate vectors like `k12dis_vec`) or whenever the
        membrane potential `self.vm` changes. It ensures that the scalar
        attributes for each rate constant (e.g., `self.k12dis`, `self.k21dis`)
        reflect the rate's value at the current `self.vm`.

        The process involves:
        1. Finding the index (`vidx`) in the `self.vt` array (the pre-defined
           voltage vector for rate calculations) that corresponds most closely
           to the current `self.vm`.
        2. Iterating through a predefined list of base rate names (e.g., 'k12dis').
        3. For each base rate name, constructing the corresponding vector attribute
           name (e.g., 'k12dis_vec').
        4. Retrieving the rate vector array (e.g., `self.k12dis_vec`).
        5. Setting the scalar rate attribute (e.g., `self.k12dis`) to the value
           from the vector array at the determined `vidx`.
        6. If the rate vector does not exist, is not a NumPy array, or is too short,
           the corresponding scalar rate is set to 0.0 as a fallback.

        This mechanism allows the model's ODE solver (`NowDerivs`) to use
        up-to-date scalar rate constants that are appropriate for the
        instantaneous membrane potential during a simulation.

        Updates:
            Sets scalar attributes for all intrinsic rate constants (e.g.,
            `self.k12dis`, `self.k71dis`, etc.) based on `self.vm` and
            the pre-calculated `_vec` arrays.
        """
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
        """Calculates the GHK current scaling factor across the voltage vector.

        This method computes `self.iscft`, an array containing the Goldman-Hodgkin-Katz
        (GHK) current scaling factor for each voltage in `self.vt`. This factor is
        subsequently used to convert the channel's open probability (Po) into
        ionic current. The calculation explicitly handles the mathematical
        singularity of the GHK current equation at V=0 by applying a limiting
        form (linear approximation) when the voltage is close to zero.

        The GHK current equation is:
        I = P * (F^2 * V / (R * T)) * ([ion_out] - [ion_in] * exp(-FV/RT)) / (1 - exp(-FV/RT))

        And its limiting form as V -> 0 is:
        I_limit = P * (F^2 / (R * T)) * ([ion_in] - [ion_out])
        (Note: The provided code uses (Nai - Nao), which implies a sign convention
        or a specific definition of P or current direction.)

        The method performs the following steps:
        1. Converts `self.vt` (in mV) to volts (`v_volts`).
        2. Identifies voltages near zero to apply the GHK limit.
        3. For near-zero voltages:
           Calculates `iscft` using the linear approximation of the GHK equation.
           `du2_zero = (F^2 / (R * T))`
           `iscft = PNa * du2_zero * (Nai - Nao)`
        4. For non-zero voltages:
           Calculates `iscft` using the standard GHK current equation, rearranged
           to isolate the scaling factor.
           `du1 = (V * F) / (R * T)`
           `du3 = exp(-du1)`
           `iscft = PNa * (F * du1 * (Nai - Nao * du3)) / (1 - du3)`
        5. Stores the results in `self.iscft`.

        `self.PNasc` (scaled Na+ permeability), `self.F` (Faraday's constant),
        `self.Rgc` (gas constant), `self.Tkel` (temperature in Kelvin),
        `self.Nai` (internal Na+ concentration), and `self.Nao` (external Na+
        concentration) are model parameters used in these calculations.

        Updates:
            self.iscft (numpy.ndarray): Array of GHK current scaling factors,
                                       one for each voltage in `self.vt`.
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
        """Calculates equilibrium state occupancies at a given membrane potential.

        This method determines the steady-state probability distribution across the
        24 states of the anticonvulsant Markov model for a specified membrane
        potential `vm`. It is crucial for initializing simulations or analyzing
        the model's behavior under prolonged voltage clamp.

        The calculation assumes detailed balance and involves the following steps:
        1. Sets the current membrane potential `self.vm` to the input `vm`.
        2. Ensures rate vectors are initialized and updates all voltage-dependent
           rate constants (both vector and scalar forms) by calling
           `create_rate_waves()`, `stRatesVolt()`, and `_update_scalar_rates()`.
        3. Defines a nested helper function `safe_div` to handle potential
           divisions by zero when calculating rate ratios.
        4. Calculates ratios of forward to backward microscopic rate constants
           for sequences of transitions (e.g., `du1 = k12dis / k21dis`). These
           represent equilibrium constants for pairs of connected states.
        5. Defines drug binding factors:
           - `drug_factor_closed`: Equilibrium constant for drug binding to closed states.
             (Currently set to 0, implying negligible binding to closed states or
             that these states are not directly drug-accessible in this simplified equilibrium).
           - `drug_factor_inactivated`: Equilibrium constant for drug binding to
             inactivated states (`k_on_inactivated / k_off_inactivated`).
        6. Computes sums of products of these rate ratios (`dusuma_free`, `dusumb_free`)
           representing the relative total occupancy of all closed states and all
           inactivated states of the drug-free channel, respectively, relative to
           the first closed state (C1, state 0).
        7. Computes similar sums for drug-bound states (`dusuma_drug`, `dusumb_drug`)
           by multiplying the free-channel sums with the respective drug factors.
        8. Calculates the total sum (`dusum_total`) across all effective states.
        9. Initializes a 24-element array `pop` for state occupancies.
        10. If `dusum_total` is significantly greater than zero:
            - Calculates the occupancy of each of the 6 drug-free closed states (C1-C5, O)
              by dividing their relative occupancies (derived from `closed_products`)
              by `dusum_total`.
            - Calculates the occupancy of each of the 6 drug-free inactivated states (I1-I6)
              similarly using `inact_products`.
            - Calculates occupancies for the 6 drug-bound closed states and 6 drug-bound
              inactivated states using their respective drug factors and products.
        11. If `dusum_total` is close to zero (e.g., all rates are zero), it assigns
            a default distribution (0.98 to state 0, 0.02 to state 1) as a fallback.
            This part seems specific and might need review for general applicability.
        12. Ensures no NaN values are present in `pop`.
        13. Returns the `pop` array containing the 24 state occupancies.

        Args:
            vm (float): The membrane potential (in mV) at which to calculate
                        equilibrium occupancies.

        Returns:
            numpy.ndarray: A 24-element array representing the steady-state
                           occupancies of the model states.
        """
        self.vm = vm
        if (not hasattr(self, 'k12dis_vec')):
            self.create_rate_waves()
        self.stRatesVolt()
        self._update_scalar_rates()
        def safe_div(a, b, default=0.0):
            """Performs element-wise division, returning a default value for division by zero.

            Args:
                a (Union[float, numpy.ndarray]): Numerator(s).
                b (Union[float, numpy.ndarray]): Denominator(s).
                default (float, optional): Value to return where b is close to zero.
                                           Defaults to 0.0.

            Returns:
                Union[float, numpy.ndarray]: Result of a / b, or `default` where b is
                                             close to zero (abs(b) <= 1e-10).
            """
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
        """Calculates the time derivatives of state occupancies for the ODE solver.

        This method defines the system of ordinary differential equations (ODEs)
        for the 24-state anticonvulsant Markov model. It computes dy/dt, the rate
        of change of occupancies for each state, given the current time `t` and
        state occupancies `y`. This output is used by an ODE solver (e.g.,
        `scipy.integrate.solve_ivp`) to simulate the model's dynamics.

        The method performs the following key steps:
        1. Retrieves all voltage-dependent intrinsic rate constants (e.g., `k12dis`)
           from pre-calculated vectors (`_vec` attributes) based on the current
           membrane potential `self.vm`.
        2. Retrieves drug binding/unbinding rate constants (e.g., `k_on_closed`).
        3. Constructs the 24x24 transition rate matrix (Q-matrix):
           - Off-diagonal elements `Q[i, j]` represent the rate constant of
             transition from state `j` to state `i`.
           - Diagonal elements `Q[i, i]` are set to the negative sum of all
             rate constants for transitions leaving state `i`.
           The Q-matrix encompasses:
             - Intrinsic transitions within the 12 drug-free states (0-11: C1-O, I1-I6).
             - Intrinsic transitions within the 12 drug-bound states (12-23: DC1-DO, DI1-DI6),
               which mirror the drug-free transitions.
             - Drug association/dissociation transitions between corresponding drug-free
               and drug-bound states (e.g., C1 <-> DC1, I1 <-> DI1).
        4. Calculates the derivatives `dstdt` (dy/dt) using the matrix-vector
           multiplication: `dstdt = Q * y`.
        5. Includes input and output validation to handle potential NaN or Inf values,
           returning a zero vector in such cases to prevent solver failures.

        Args:
            t (float): Current time point (formally required by ODE solvers,
                       though not explicitly used in this time-invariant system
                       as rate constants depend on `self.vm` which is set per epoch).
            y (numpy.ndarray): A 1D array of current state occupancies (probabilities)
                               for the 24 states of the model.

        Returns:
            numpy.ndarray: A 1D array representing the time derivatives (dy/dt)
                           of the occupancies for each of the 24 states.

        State Indexing Convention:
            - States  0-5:  Drug-free (C1, C2, C3, C4, C5, O)
            - States  6-11: Drug-free Inactivated (I1, I2, I3, I4, I5, I6)
            - States 12-17: Drug-bound (DC1, DC2, DC3, DC4, DC5, DO)
            - States 18-23: Drug-bound Inactivated (DI1, DI2, DI3, DI4, DI5, DI6)
        """
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
        """Simulates a single voltage clamp sweep protocol for the 24-state model.

        This method executes a pre-defined voltage clamp sweep, identified by
        `SwpNo` in the `self.SwpSeq` array. It iteratively solves the model's
        ODEs for each voltage epoch in the sweep, calculating and storing key
        simulation outputs like ionic current, open probability, and state
        occupancy sums. This method overrides the base `MarkovModel.Sweep` to
        specifically handle the 24-state anticonvulsant model and its
        corresponding result storage methods.

        The simulation process involves:
        1. Validating the sweep number and protocol definition.
        2. Initializing NumPy arrays to store time-series results (e.g.,
           `self.SimSwp` for current, `self.SimOp` for open probability,
           `self.SimDrugBound` for total drug-bound probability).
        3. Parsing epoch voltages and durations from `self.SwpSeq`. A fixed
           sampling interval of 0.005 ms is used.
        4. Setting initial conditions:
           - The membrane potential (`self.vm`) is set to the first epoch's voltage.
           - GHK current factors are calculated via `self.CurrVolt()`.
           - Initial state occupancies (`self.pop`) are set to equilibrium values
             at this voltage using `self.EquilOccup(self.vm)`.
           - Initial results are stored using `self._store_results_24()`.
        5. Iterating through each subsequent voltage epoch:
           - `self.vm` is updated to the current epoch's voltage.
           - Scalar rate constants and GHK factors are updated via
             `self._update_scalar_rates()` and `self.CurrVolt()`.
           - Time points for ODE solution (`t_eval`) are determined for the epoch.
           - The ODE system (`self.NowDerivs`) is solved using
             `scipy.integrate.solve_ivp` (LSODA method) for the current epoch,
             with initial conditions taken from the end of the previous epoch.
           - If the ODE solution is successful, results are stored in batches
             using `self._store_results_vectorized_24()`.
           - `self.pop` is updated to the final state of the epoch.
        6. A final time vector for the entire sweep (`self.time`) is generated.
        7. Returns the time points from the last ODE solution and the array of
           simulated currents (`self.SimSwp`).

        Args:
            SwpNo (int): The index of the sweep to simulate from `self.SwpSeq`.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Time points from the last epoch's ODE solution.
                - numpy.ndarray: Array of simulated ionic currents for the entire sweep.

        Raises:
            ValueError: If `SwpNo` is invalid or the protocol definition is inconsistent.

        Updates:
            self.SimSwp, self.SimOp, self.SimIn, self.SimAv, self.SimCom,
            self.SimDrugBound, self.pop, self.vm, self.time, self.full_sol_t,
            self.full_sol_y, and GHK factors via `self.CurrVolt()`.
        """
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
        """Stores simulation results for a single time point from the 24-state model.

        This helper method is typically called to store the initial state of a
        simulation sweep or results from individual time points if not using
        vectorized storage. It calculates and stores various simulation outputs
        based on the current state occupancies (`self.pop`) and membrane
        potential (`self.vm`).

        Key calculations and stored values:
        - Ionic Current (`self.SimSwp`): Calculated using the GHK formulation,
          considering only the drug-free open state (O, state 5) as conducting.
          Uses `self.iscft`, `self.numchan`, and `self.current_scaling`.
        - Total Open Probability (`self.SimOp`): Sum of drug-free open (O, state 5)
          and drug-bound open (DO, state 17) probabilities.
        - Total Inactivated Probability (`self.SimIn`): Sum of all drug-free
          inactivated states (I1-I6, states 6-11) and drug-bound inactivated
          states (DI1-DI6, states 18-23).
        - Total Available Probability (`self.SimAv`): Sum of drug-free closed and
          open states (C1-C5, O, states 0-5).
        - Command Voltage (`self.SimCom`): The current membrane potential `self.vm`.
        - Total Drug-Bound Probability (`self.SimDrugBound`): Sum of all drug-bound
          states (DC1-DO, DI1-DI6, states 12-23).

        Args:
            idx (int): The index in the simulation result arrays (e.g., `self.SimSwp`)
                       where the current results should be stored.
            t (float): The current simulation time. (Note: `t` is not directly used
                       in the body of this specific method implementation but is
                       part of the common signature for result storage methods).

        Updates:
            self.SimSwp[idx], self.SimOp[idx], self.SimIn[idx], self.SimAv[idx],
            self.SimCom[idx], self.SimDrugBound[idx]
        """
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
        """Stores simulation results for a batch of time points from the 24-state model.

        This helper method is designed for efficient storage of results obtained
        from an ODE solver that returns solutions for multiple time points
        simultaneously (a batch). It calculates and stores various simulation
        outputs based on the provided `batch_states` and `voltage`.

        Key calculations and stored values for each time point in the batch:
        - Ionic Currents (`self.SimSwp`): Calculated using the GHK formulation,
          considering only the drug-free open state (O, state 5) from `batch_states`
          as conducting. Uses `self.iscft`, `self.numchan`, and `self.current_scaling`.
        - Total Open Probabilities (`self.SimOp`): Sum of drug-free open (O, state 5)
          and drug-bound open (DO, state 17) probabilities from `batch_states`.
        - Total Inactivated Probabilities (`self.SimIn`): Sum of all drug-free
          inactivated states (I1-I6, states 6-11) and drug-bound inactivated
          states (DI1-DI6, states 18-23) from `batch_states`.
        - Total Available Probabilities (`self.SimAv`): Sum of drug-free closed and
          open states (C1-C5, O, states 0-5) from `batch_states`.
        - Command Voltages (`self.SimCom`): The `voltage` parameter is stored for
          all specified `indices`.
        - Total Drug-Bound Probabilities (`self.SimDrugBound`): Sum of all drug-bound
          states (DC1-DO, DI1-DI6, states 12-23) from `batch_states`.

        Args:
            indices (numpy.ndarray): A 1D array of integer indices indicating where
                                     the batch results should be stored in the
                                     simulation result arrays (e.g., `self.SimSwp`).
            batch_states (numpy.ndarray): A 2D array where rows correspond to time
                                          points and columns correspond to the
                                          occupancies of the 24 model states.
            voltage (float): The membrane potential at which these states were simulated.

        Updates:
            Slices of self.SimSwp, self.SimOp, self.SimIn, self.SimAv,
            self.SimCom, self.SimDrugBound corresponding to `indices`.
        """
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
    def create_default_protocol(self, target_voltages=None, holding_potential=-80, holding_duration=98, test_duration=200, tail_duration=2):
        """Creates a standard multi-step voltage clamp protocol.

        This protocol consists of multiple sweeps, each with three epochs:
        1. Holding Epoch: At `holding_potential` for `holding_duration`.
        2. Test Epoch: At a voltage from `target_voltages` for `test_duration`.
        3. Tail Epoch: Returns to `holding_potential` for `tail_duration`.

        Each sweep tests a different voltage from the `target_voltages` list.
        The generated protocol is stored in `self.SwpSeq` and also copied to
        `self.SwpSeqMultiStepKeyVoltages`.

        Args:
            target_voltages (list, optional): A list of voltages (mV) for the
                test epoch. Defaults to `[30, 0, -20, -30, -40, -50, -60]`.
            holding_potential (float, optional): Voltage (mV) for the holding
                and tail epochs. Defaults to -80 mV.
            holding_duration (float, optional): Duration (ms) of the initial
                holding epoch. Defaults to 98 ms.
            test_duration (float, optional): Duration (ms) of the test epoch.
                Defaults to 200 ms.
            tail_duration (float, optional): Duration (ms) of the tail epoch.
                Defaults to 2 ms.

        Updates:
            self.BsNm (str): Set to 'MultiStepKeyVoltages'.
            self.NumSwps (int): Number of sweeps, equal to `len(target_voltages)`.
            self.SwpSeq (numpy.ndarray): The generated sweep protocol array.
            self.SwpSeqMultiStepKeyVoltages (numpy.ndarray): A copy of `self.SwpSeq`.
            Calls `self.CurrVolt()`.
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
    def create_inactivation_protocol(self, inactivating_voltage=-20, test_voltage=0, inactivating_duration=2000, recovery_duration=100):
        """Creates a protocol to assess channel inactivation.

        This protocol consists of a single sweep with four epochs:
        1. Initial Holding: At -80 mV for 200 ms.
        2. Inactivating Pulse: At `inactivating_voltage` for `inactivating_duration`.
        3. Test Pulse: At `test_voltage` for 5 ms to measure available current.
        4. Recovery Epoch: At -80 mV for `recovery_duration`.

        The generated protocol is stored in `self.SwpSeq` and also copied to
        `self.SwpSeqInactivationProtocol`.

        Args:
            inactivating_voltage (float, optional): Voltage (mV) of the inactivating
                pulse. Defaults to -20 mV.
            test_voltage (float, optional): Voltage (mV) of the test pulse.
                Defaults to 0 mV.
            inactivating_duration (float, optional): Duration (ms) of the
                inactivating pulse. Defaults to 2000 ms.
            recovery_duration (float, optional): Duration (ms) of the final
                recovery epoch. Defaults to 100 ms.

        Updates:
            self.BsNm (str): Set to 'InactivationProtocol'.
            self.NumSwps (int): Set to 1.
            self.SwpSeq (numpy.ndarray): The generated sweep protocol array.
            self.SwpSeqInactivationProtocol (numpy.ndarray): A copy of `self.SwpSeq`.
            Calls `self.CurrVolt()`.
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
    def create_recovery_protocol(self, target_recovery_times=None, holding_potential=-80, inactivating_voltage=-20, test_voltage=0, holding_duration=200, inactivating_duration=2000, test_duration=20, tail_duration=100):
        """Creates a protocol to measure recovery from inactivation.

        This protocol consists of multiple sweeps, each with five epochs:
        1. Initial Holding: At `holding_potential` for `holding_duration`.
        2. Inactivating Pulse: At `inactivating_voltage` for `inactivating_duration`.
        3. Recovery Interval: At `holding_potential` for a variable duration from
           `target_recovery_times`.
        4. Test Pulse: At `test_voltage` for `test_duration` to measure current.
        5. Tail Epoch: At `holding_potential` for `tail_duration`.

        Each sweep uses a different recovery interval from `target_recovery_times`.
        The generated protocol is stored in `self.SwpSeq` and also copied to
        `self.SwpSeqRecoveryFromInactivation`.

        Args:
            target_recovery_times (list, optional): A list of durations (ms) for
                the recovery interval. Defaults to `[1, 3, 10, 30, 100, 300, 1000]`.
            holding_potential (float, optional): Voltage (mV) for holding, recovery,
                and tail epochs. Defaults to -80 mV.
            inactivating_voltage (float, optional): Voltage (mV) of the inactivating
                pulse. Defaults to -20 mV.
            test_voltage (float, optional): Voltage (mV) of the test pulse.
                Defaults to 0 mV.
            holding_duration (float, optional): Duration (ms) of the initial
                holding epoch. Defaults to 200 ms.
            inactivating_duration (float, optional): Duration (ms) of the
                inactivating pulse. Defaults to 2000 ms.
            test_duration (float, optional): Duration (ms) of the test pulse.
                Defaults to 20 ms.
            tail_duration (float, optional): Duration (ms) of the tail epoch.
                Defaults to 100 ms.

        Updates:
            self.BsNm (str): Set to 'RecoveryFromInactivation'.
            self.NumSwps (int): Number of sweeps, `len(target_recovery_times)`.
            self.SwpSeq (numpy.ndarray): The generated sweep protocol array.
            self.SwpSeqRecoveryFromInactivation (numpy.ndarray): A copy of `self.SwpSeq`.
            Calls `self.CurrVolt()`.
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
    def create_steady_state_inactivation_protocol(self, test_voltages=None, holding_potential=-120, prepulse_duration=2000, test_pulse_voltage=0, test_pulse_duration=5, recovery_duration=100):
        """Creates a protocol to determine steady-state inactivation (SSI).

        Also known as an availability curve protocol. This protocol consists of
        multiple sweeps, each with four epochs:
        1. Initial Holding: At `holding_potential` for 200 ms.
        2. Pre-pulse: At a variable voltage from `test_voltages` for
           `prepulse_duration` to allow channels to reach steady-state inactivation.
        3. Test Pulse: At `test_pulse_voltage` for `test_pulse_duration` to
           measure available current.
        4. Recovery Epoch: At `holding_potential` for `recovery_duration`.

        Each sweep uses a different pre-pulse voltage from `test_voltages`.
        The generated protocol is stored in `self.SwpSeq` and also copied to
        `self.SwpSeqSteadyStateInactivation`.

        Args:
            test_voltages (list or numpy.ndarray, optional): Voltages (mV) for the
                pre-pulse. Defaults to `np.arange(-120, -15, 5)`.
            holding_potential (float, optional): Voltage (mV) for the initial holding
                and recovery epochs. Defaults to -120 mV.
            prepulse_duration (float, optional): Duration (ms) of the pre-pulse.
                Defaults to 2000 ms.
            test_pulse_voltage (float, optional): Voltage (mV) of the test pulse.
                Defaults to 0 mV.
            test_pulse_duration (float, optional): Duration (ms) of the test pulse.
                Defaults to 5 ms.
            recovery_duration (float, optional): Duration (ms) of the recovery epoch.
                Defaults to 100 ms.

        Updates:
            self.BsNm (str): Set to 'SteadyStateInactivation'.
            self.NumSwps (int): Number of sweeps, `len(test_voltages)`.
            self.SwpSeq (numpy.ndarray): The generated sweep protocol array.
            self.SwpSeqSteadyStateInactivation (numpy.ndarray): A copy of `self.SwpSeq`.
            Calls `self.CurrVolt()`.
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