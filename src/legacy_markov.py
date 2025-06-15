import numpy as np
from scipy.integrate import solve_ivp
class MarkovModel:
    """
    Implements a traditional Markov model for simulating ion channel currents,
    specifically sodium channels. This version is considered "legacy."
    The model defines a set of 13 discrete states representing different
    conformations of the ion channel (e.g., closed, open, inactivated).
    Transitions between these states are governed by voltage-dependent rate
    constants. The model simulates the time evolution of the probability
    distribution across these states in response to a voltage-clamp protocol,
    using an ODE solver.
    Key Attributes:
        NumSwps (int): Number of sweeps in the current voltage protocol.
        num_states (int): Total number of states in the model (fixed at 13).
        vm (float): Current membrane potential in mV.
        pop (np.ndarray): A 1D array (size 13) representing the probabilities
                          of the channel being in each of its states.
        SwpSeq (np.ndarray): The current voltage clamp protocol sequence.
        SimSwp (np.ndarray): Stores the simulated current for each time point
                             of the last run sweep.
        SimOp (np.ndarray): Stores the probability of the channel being in the
                            open state (state 6, index 5).
        SimIn (np.ndarray): Stores the sum of probabilities of the channel
                            being in any inactivated state (states 7-13, indices 6-12).
        SimAv (np.ndarray): Stores the sum of probabilities of the channel
                            being in an available (non-inactivated) state (states 1-6, indices 0-5).
        SimCom (np.ndarray): Stores the command voltage for each time point.
        time (np.ndarray): Time vector for the simulation.
        vt (np.ndarray): A pre-defined array of voltage points (-200mV to 200mV)
                         for which rates and currents are pre-calculated or looked up.
        iscft (np.ndarray): Current scaling factor for each voltage in `vt`,
                            derived from GHK equation.
        # ... (other biophysical parameters like alcoeff, btslp, etc.)
    The model structure involves:
    - Initialization of biophysical parameters (`init_parameters`).
    - Initialization of data structures (`init_waves`, `create_rate_waves`).
    - Calculation of voltage-dependent transition rates (`stRatesVolt`, `_update_scalar_rates`).
    - Calculation of current-voltage relationships (`CurrVolt`).
    - Calculation of equilibrium state occupancies (`EquilOccup`).
    - Simulation of sweeps using `scipy.integrate.solve_ivp` (`Sweep`, `NowDerivs`).
    - Creation of default voltage protocols (`create_default_protocol`).
    """
    def __init__(self):
        """
        Initializes the MarkovModel instance.
        Sets up default values for sweep counts, the number of states (13),
        and the initial membrane potential. It then calls a sequence of
        helper methods to:
        - Initialize all biophysical parameters (`init_parameters`).
        - Initialize data arrays (e.g., for state populations, time series)
          and pre-calculate voltage-dependent rate arrays (`init_waves`).
        - Calculate initial voltage-dependent transition rates (`stRatesVolt`).
        - Calculate initial current-voltage relationships (`CurrVolt`).
        - Create and set a default voltage-clamp protocol (`create_default_protocol`).
        """
        self.NumSwps = 0                           
        self.num_states = 13                                         
        self.vm = -80                             
        self.init_parameters()
        self.init_waves()
        self.stRatesVolt()
        self.CurrVolt()
        self.create_default_protocol()                                      
    def init_parameters(self):
        """Initialize biophysical parameters."""
        self.alcoeff = 20     
        self.alslp = 40           
        self.btcoeff = 0.3    
        self.btslp = 18.5      
        self.ConCoeff = 0.004                                          
        self.CoffCoeff = 4.5                                          
        self.ConSlp = 1e8                                           
        self.CoffSlp = 1e8                                          
        self.gmcoeff = 50      
        self.gmslp = 100       
        self.dlcoeff = 0.8
        self.dlslp = 6
        self.OpOnCoeff = 4                          
        self.OpOffCoeff = 0.008                      
        self.ConHiCoeff = 4                                
        self.CoffHiCoeff = 0.008                             
        self.OpOnSlp = 1e8                                          
        self.OpOffSlp = 1e8                                         
        self.konlo = self.kofflo = self.konhi = self.koffhi = 0
        self.konop = self.koffop = self.kdlo = self.kdhi = 0
        self.alfac = np.sqrt(np.sqrt(self.ConHiCoeff / self.ConCoeff))
        self.btfac = np.sqrt(np.sqrt(self.CoffCoeff / self.CoffHiCoeff))
        self.numchan = 1
        self.F = 96485
        self.Rgc = 8314
        self.Tkel = 295
        self.Nao, self.Nai = 150, 15
        self.ClipRate = 6000
        self.current_scaling = 0.0125
        self.vm = -80                             
        self.PNasc = 1e-5 
        self._reusable_y0 = np.zeros(12)                                                               
    def init_waves(self):
        """
        Initializes data arrays for simulation and pre-calculated values.
        This method sets up:
        - `vt`: A numpy array of voltage points from -200mV to 200mV.
        - `pop`: A numpy array (size 13) to store the probability of the channel
                 being in each of its 13 states.
        - `dstdt`: A numpy array (size 12) to store derivatives of the first 12
                   state probabilities for the ODE solver.
        - `_reusable_y0`: A numpy array (size 12) for ODE initial conditions.
        - `iscft`: An array to store GHK-derived current scaling factors for each
                   voltage in `vt`.
        Calls `create_rate_waves()` to initialize arrays for storing
        voltage-dependent transition rates and `stRatesVolt()` to populate them.
        """
        self.vt = np.arange(-200, 201)
        self.pop = np.zeros(13)  
        self.dstdt = np.zeros(12)                             
        self._reusable_y0 = np.zeros(12)
        self.iscft = np.zeros_like(self.vt)
        self.create_rate_waves()
        self.stRatesVolt()                             
    def create_rate_waves(self):
        """
        Creates numpy arrays to store pre-calculated voltage-dependent transition rates.
        For each defined transition rate in the model (e.g., 'k12dis', 'k23dis'),
        this method initializes a corresponding numpy array (e.g., `self.k12dis_vec`)
        with the same size as `self.vt`. These arrays will be populated by
        `stRatesVolt` with the rate values at each voltage in `self.vt`.
        """
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis',
                     'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis',
                     'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis',
                     'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis',
                     'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis',
                     'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis',
                     'k98dis', 'k87dis']
        for name in rate_names:
            setattr(self, name + '_vec', np.zeros_like(self.vt, dtype=float))
    def stRatesVolt(self):
        """
        Calculates and stores all voltage-dependent state transition rates.
        This method computes the rates for all defined transitions in the
        13-state Markov model across the pre-defined voltage range `self.vt`.
        It uses various biophysical parameters (e.g., `alcoeff`, `btslp`,
        `ConCoeff`) to calculate rates like alpha/beta for activation,
        kon/koff for inactivation, and other inter-state transition rates.
        The calculated rates for each transition (e.g., k12dis, k21dis) are
        stored in their respective vectorized arrays (e.g., `self.k12dis_vec`,
        `self.k21dis_vec`) for efficient lookup during simulations.
        It also calls `_update_scalar_rates()` to set the scalar rate attributes
        (e.g., `self.k12dis`) based on the current `self.vm`.
        Rates are clipped to `self.ClipRate` if they exceed it.
        """
        if not hasattr(self, 'ClipRate') or self.ClipRate is None:
            self.ClipRate = 1000
        if not hasattr(self, 'k12dis_vec'):
            self.create_rate_waves()
        vt = self.vt
        amt = self.alcoeff * np.exp(vt / self.alslp)
        bmt = self.btcoeff * np.exp(-vt / self.btslp)
        gmt = self.gmcoeff * np.exp(vt / self.gmslp)
        dmt = self.dlcoeff * np.exp(-vt / self.dlslp)
        konlo = self.ConCoeff * np.exp(vt / self.ConSlp)
        kofflo = self.CoffCoeff * np.exp(-vt / self.CoffSlp)
        konop = self.OpOnCoeff * np.exp(vt / self.OpOnSlp)
        koffop = self.OpOffCoeff * np.exp(-vt / self.OpOffSlp)
        self.k12dis_vec = np.minimum(4 * amt, self.ClipRate)
        self.k23dis_vec = np.minimum(3 * amt, self.ClipRate)
        self.k34dis_vec = np.minimum(2 * amt, self.ClipRate)
        self.k45dis_vec = np.minimum(amt, self.ClipRate)
        self.k56dis_vec = np.minimum(gmt, self.ClipRate)
        self.k65dis_vec = np.minimum(dmt, self.ClipRate)
        self.k54dis_vec = np.minimum(4 * bmt, self.ClipRate)
        self.k43dis_vec = np.minimum(3 * bmt, self.ClipRate)
        self.k32dis_vec = np.minimum(2 * bmt, self.ClipRate)
        self.k21dis_vec = np.minimum(bmt, self.ClipRate)
        dph = 1                         
        self.k17dis_vec = np.minimum(konlo * dph, self.ClipRate)
        self.k71dis_vec = np.minimum(kofflo, self.ClipRate)
        self.k28dis_vec = np.minimum(self.k17dis_vec * self.alfac, self.ClipRate)
        self.k82dis_vec = np.minimum(self.k71dis_vec / self.btfac, self.ClipRate)
        self.k39dis_vec = np.minimum(self.k17dis_vec * self.alfac**2, self.ClipRate)
        self.k93dis_vec = np.minimum(self.k71dis_vec / (self.btfac**2), self.ClipRate)
        self.k410dis_vec = np.minimum(self.k17dis_vec * self.alfac**3, self.ClipRate)
        self.k104dis_vec = np.minimum(self.k71dis_vec / (self.btfac**3), self.ClipRate)
        self.k511dis_vec = np.minimum(self.k17dis_vec * self.alfac**4, self.ClipRate)
        self.k115dis_vec = np.minimum(self.k71dis_vec / (self.btfac**4), self.ClipRate)
        self.k612dis_vec = np.minimum(konop, self.ClipRate)
        self.k126dis_vec = np.minimum(koffop, self.ClipRate)
        self.k78dis_vec = np.minimum(4 * amt * self.alfac, self.ClipRate)
        self.k89dis_vec = np.minimum(3 * amt * self.alfac, self.ClipRate)
        self.k910dis_vec = np.minimum(2 * amt * self.alfac, self.ClipRate)
        self.k1011dis_vec = np.minimum(amt * self.alfac, self.ClipRate)
        self.k1112dis_vec = np.minimum(gmt, self.ClipRate)
        self.k1110dis_vec = np.minimum(4 * bmt * (1/self.btfac), self.ClipRate)
        self.k109dis_vec = np.minimum(3 * bmt * (1/self.btfac), self.ClipRate)  
        self.k98dis_vec = np.minimum(2 * bmt * (1/self.btfac), self.ClipRate)
        self.k87dis_vec = np.minimum(bmt * (1/self.btfac), self.ClipRate)
        k115_safe = np.where(self.k115dis_vec > 0, self.k115dis_vec, 1.0)
        self.k1211dis_vec = np.minimum(
            (self.k65dis_vec * self.k511dis_vec * self.k126dis_vec) / 
            (self.k612dis_vec * k115_safe), 
            self.ClipRate
        )
        self._update_scalar_rates()
    def _update_scalar_rates(self):
        """
        Updates scalar rate attributes based on the current membrane potential `self.vm`.
        This helper method finds the closest voltage index in `self.vt` to
        the current `self.vm`. It then uses this index to look up the
        pre-calculated vectorized rates (e.g., `self.k12dis_vec`) and assigns
        them to their corresponding scalar attributes (e.g., `self.k12dis`).
        These scalar rates are used by `NowDerivs` and `EquilOccup` for
        constructing the transition matrix at a specific `self.vm`.
        """
        vidx = np.argmin(np.abs(self.vt - self.vm))
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis',
                     'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis',
                     'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis',
                     'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis',
                     'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis',
                     'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis',
                     'k98dis', 'k87dis']
        for name in rate_names:
            vec_name = name + '_vec'
            if hasattr(self, vec_name):
                vec_array = getattr(self, vec_name)
                if isinstance(vec_array, np.ndarray) and len(vec_array) > vidx:
                    setattr(self, name, vec_array[vidx])
                else:
                    setattr(self, name, 0.0)
            else:
                setattr(self, name, 0.0)
    def CurrVolt(self):
        """
        Calculates the current-voltage (I-V) relationship for the open state.
        This method computes the single-channel current (`iscft`) for each
        voltage in the `self.vt` array using the Goldman-Hodgkin-Katz (GHK)
        current equation. It considers sodium ion concentrations (`Nao`, `Nai`),
        permeability (`PNasc`), temperature (`Tkel`), and physical constants.
        The results are stored in `self.iscft`. This array is used as a scaling
        factor during simulations (`_store_results`, `_store_results_vectorized`)
        to calculate the total macroscopic current based on the probability of
        the channel being in the open state (state 6, index 5).
        Handles potential division by zero if vm is exactly 0 mV.
        """
        scaled_PNasc = self.PNasc
        v_volts = self.vt * 1e-3                                     
        near_zero = np.abs(v_volts) < 1e-6
        not_zero = ~near_zero
        self.iscft = np.zeros_like(v_volts)
        if np.any(near_zero):
            du2_zero = self.F * self.F / (self.Rgc * self.Tkel)
            self.iscft[near_zero] = scaled_PNasc * du2_zero * (self.Nai - self.Nao)
        if np.any(not_zero):
            v_nz = v_volts[not_zero]
            du1 = (v_nz * self.F) / (self.Rgc * self.Tkel)
            du3 = np.exp(-du1)
            du5_corrected = self.F * du1 * (self.Nai - self.Nao * du3) / (1 - du3)
            self.iscft[not_zero] = scaled_PNasc * du5_corrected
    def EquilOccup(self, vm):
        """
        Calculates the equilibrium state occupancies at a given membrane potential.
        This method determines the steady-state probabilities for each of the
        13 states of the Markov model at the specified voltage `vm`.
        It first updates the scalar transition rates based on `vm` using
        `_update_scalar_rates()`. Then, it constructs the transition rate
        matrix (Q matrix) and solves the system dP/dt = Q * P = 0, subject to
        sum(P) = 1, to find the equilibrium probabilities.
        The calculation involves solving a system of linear equations derived
        from the rate constants. A `safe_div` helper function is used to
        prevent division by zero errors during these calculations.
        Args:
            vm (float): The membrane potential (in mV) at which to calculate
                        equilibrium occupancies.
        Returns:
            np.ndarray: A 1D array of 13 elements representing the equilibrium
                        probabilities for each state (P1 to P13).
        """
        self.vm = vm
        if not hasattr(self, 'k12dis_vec'):
            self.create_rate_waves()
        self.stRatesVolt()
        self._update_scalar_rates()
        def safe_div(a, b, default=0.0):
            """Safely divides a by b, returning default if b is zero."""
            if np.isscalar(b):
                if abs(b) > 1e-10:
                    return a / b
                else:
                    return default
            else:
                result = np.full_like(a, default, dtype=float)
                mask = np.abs(b) > 1e-10
                if np.any(mask):
                    result[mask] = a[mask] / b[mask]
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
        dusuma = 1 + du1 + du1*du2 + du1*du2*du3 + du1*du2*du3*du4 + du1*du2*du3*du4*du5
        dusumb = du7 + du7*du8 + du7*du8*du9 + du7*du8*du9*du10 + du7*du8*du9*du10*du11 + du7*du8*du9*du10*du11*du12
        dusum = dusuma + dusumb
        pop = np.zeros(12)
        if dusum > 1e-10:
            du_products = np.array([
                1, du1, du1*du2, du1*du2*du3, du1*du2*du3*du4, 
                du1*du2*du3*du4*du5
            ])
            du7_products = np.array([
                du7, du7*du8, du7*du8*du9, du7*du8*du9*du10,
                du7*du8*du9*du10*du11, du7*du8*du9*du10*du11*du12
            ])
            pop[:6] = du_products / dusum                            
            pop[6:12] = du7_products / dusum                              
        else:
            pop[0] = 0.98                     
            pop[1] = 0.02                     
        pop = np.nan_to_num(pop, nan=0.0)
        return pop
    def NowDerivs(self, t, y):
        """
        Calculates the derivatives of state probabilities for the ODE solver.
        This function is used by `scipy.integrate.solve_ivp` during the
        simulation of a voltage sweep. It computes dP/dt for each of the first
        12 states (P1 to P12), based on the current state probabilities `y`
        and the scalar transition rates (e.g., `self.k12dis`, `self.k21dis`)
        which are set by `_update_scalar_rates()` according to `self.vm`.
        The 13th state's probability (P13) is calculated as 1 minus the sum
        of the first 12 state probabilities, ensuring conservation of probability.
        The derivatives are defined by the set of differential equations
        representing the Markov model's state transitions.
        Args:
            t (float): The current time point in the simulation (not explicitly
                       used in rate calculation as rates depend on `self.vm`,
                       which is updated by the `Sweep` method prior to calling
                       the ODE solver).
            y (np.ndarray): A 1D array of current state probabilities for the
                            first 12 states (P1 to P12).
        Returns:
            np.ndarray: A 1D array representing the derivatives (dP/dt) for
                        each of the first 12 states.
        """
        vidx = np.searchsorted(self.vt, self.vm)
        vidx = np.clip(vidx, 0, len(self.vt) - 1)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return np.zeros_like(y)
        k12dis  = self.k12dis_vec[vidx]
        k23dis  = self.k23dis_vec[vidx]
        k34dis  = self.k34dis_vec[vidx]
        k45dis  = self.k45dis_vec[vidx]
        k56dis  = self.k56dis_vec[vidx]
        k65dis  = self.k65dis_vec[vidx]
        k54dis  = self.k54dis_vec[vidx]
        k43dis  = self.k43dis_vec[vidx]
        k32dis  = self.k32dis_vec[vidx]
        k21dis  = self.k21dis_vec[vidx]
        k17dis  = self.k17dis_vec[vidx]
        k71dis  = self.k71dis_vec[vidx]
        k28dis  = self.k28dis_vec[vidx]
        k82dis  = self.k82dis_vec[vidx]
        k39dis  = self.k39dis_vec[vidx]
        k93dis  = self.k93dis_vec[vidx]
        k410dis = self.k410dis_vec[vidx]
        k104dis = self.k104dis_vec[vidx]
        k511dis = self.k511dis_vec[vidx]
        k115dis = self.k115dis_vec[vidx]
        k612dis = self.k612dis_vec[vidx]
        k126dis = self.k126dis_vec[vidx]
        k78dis  = self.k78dis_vec[vidx]
        k89dis  = self.k89dis_vec[vidx]
        k910dis = self.k910dis_vec[vidx]
        k1011dis= self.k1011dis_vec[vidx]
        k1112dis= self.k1112dis_vec[vidx]
        k1211dis= self.k1211dis_vec[vidx]
        k1110dis= self.k1110dis_vec[vidx]
        k109dis = self.k109dis_vec[vidx]
        k98dis  = self.k98dis_vec[vidx]
        k87dis  = self.k87dis_vec[vidx]
        st = y.copy()
        Q = np.zeros((12, 12))
        Q[0, 1] = k21dis
        Q[0, 6] = k71dis
        Q[1, 0] = k12dis
        Q[1, 2] = k32dis
        Q[1, 7] = k82dis
        Q[2, 1] = k23dis
        Q[2, 3] = k43dis
        Q[2, 8] = k93dis
        Q[3, 2] = k34dis
        Q[3, 4] = k54dis
        Q[3, 9] = k104dis
        Q[4, 3] = k45dis
        Q[4, 5] = k65dis
        Q[4, 10] = k115dis
        Q[5, 4] = k56dis
        Q[5, 11] = k126dis
        Q[6, 0] = k17dis
        Q[6, 7] = k87dis
        Q[7, 6] = k78dis
        Q[7, 8] = k98dis
        Q[7, 1] = k28dis
        Q[8, 7] = k89dis
        Q[8, 9] = k109dis
        Q[8, 2] = k39dis
        Q[9, 8] = k910dis
        Q[9, 10] = k1110dis
        Q[9, 3] = k410dis
        Q[10, 9] = k1011dis
        Q[10, 11] = k1211dis
        Q[10, 4] = k511dis
        Q[11, 10] = k1112dis
        Q[11, 5] = k612dis
        Q[0, 0]   = -(k12dis + k17dis)                                                             
        Q[1, 1]   = -(k21dis + k23dis + k28dis)                                                         
        Q[2, 2]   = -(k32dis + k34dis + k39dis)                                                         
        Q[3, 3]   = -(k43dis + k45dis + k410dis)                                                        
        Q[4, 4]   = -(k54dis + k56dis + k511dis)                                                       
        Q[5, 5]   = -(k65dis + k612dis)                                                             
        Q[6, 6]   = -(k71dis + k78dis)                                                             
        Q[7, 7]   = -(k82dis + k87dis + k89dis)                                                         
        Q[8, 8]   = -(k93dis + k98dis + k910dis)                                                        
        Q[9, 9]   = -(k104dis + k109dis + k1011dis)                                                     
        Q[10, 10] = -(k115dis + k1110dis + k1112dis)                                                    
        Q[11, 11] = -(k126dis + k1211dis)                                                          
        dstdt = np.zeros_like(y)
        for i in range(12):
            for j in range(12):
                dstdt[i] += Q[i, j] * st[j]
        if np.any(np.isnan(dstdt)) or np.any(np.isinf(dstdt)):
            return np.zeros_like(st)
        return dstdt
    def Sweep(self, SwpNo):
        """
        Runs a single voltage-clamp sweep simulation for the legacy Markov model.
        This method simulates the channel's response to a specific sweep (`SwpNo`)
        from the current voltage protocol (`self.SwpSeq`). The process involves:
        1. Setting initial state probabilities (`self.pop`) using `EquilOccup`
           at the holding potential of the first epoch of the sweep.
        2. Iterating through each epoch (voltage step) defined in the protocol for the sweep.
        3. For each epoch:
            a. Setting `self.vm` to the epoch's voltage.
            b. Updating scalar transition rates using `_update_scalar_rates()`.
            c. Using `scipy.integrate.solve_ivp` with `self.NowDerivs` to solve
               the system of ODEs describing state probability changes over time.
               The initial conditions for the ODE solver (`y0`) are taken from
               the first 12 states of `self.pop`.
            d. Storing the results (current, open probability, etc.) at sampled
               time points using `_store_results_vectorized`.
            e. Updating `self.pop` with the final state probabilities from the epoch.
        4. Populating `self.time` with the time vector for the simulation.
        Args:
            SwpNo (int): The sweep number (0-indexed) from the `self.SwpSeq`
                         protocol to simulate.
        Returns:
            tuple: A tuple containing:
                - t (np.ndarray): The time points at which the ODE solver evaluated
                  the solution (may not exactly match `self.time`).
                - self.SimSwp (np.ndarray): The array of simulated currents for the sweep.
        """
        if SwpNo >= self.SwpSeq.shape[1] or SwpNo < 0:
            raise ValueError(f"Invalid sweep number {SwpNo}")
        SwpSeq = self.SwpSeq
        NumEpchs = int(SwpSeq[0, SwpNo])
        if NumEpchs <= 0 or 2*NumEpchs + 1 >= SwpSeq.shape[0]:
            raise ValueError("Invalid number of epochs in protocol")
        total_points = int(SwpSeq[2*NumEpchs + 1, SwpNo]) + 1
        sampint = 0.005                          
        self.SimSwp = np.zeros(total_points)                  
        self.SimOp = np.zeros(total_points)                      
        self.SimIn = np.zeros(total_points)                             
        self.SimAv = np.zeros(total_points)                           
        self.SimCom = np.zeros(total_points)                    
        self.pop = np.zeros(13)
        self.pop[0] = 1.0                                                  
        epoch_voltages = np.zeros(NumEpchs + 1)
        epoch_end_times = np.zeros(NumEpchs + 1)
        epoch_voltages[0] = SwpSeq[2, SwpNo]
        epoch_end_times[0] = 0.0
        for e in range(1, NumEpchs + 1):
            epoch_voltages[e] = SwpSeq[2 * e, SwpNo]
            epoch_end_times[e] = int(SwpSeq[2 * e + 1, SwpNo]) * sampint
        self.vm = epoch_voltages[0]
        self.CurrVolt()
        self.pop = self.EquilOccup(self.vm)
        self._store_results(0, 0)
        current_time = 0.0
        store_idx = 1
        for epoch in range(1, NumEpchs + 1):
            self.vm = epoch_voltages[epoch]
            epoch_end_time = epoch_end_times[epoch]
            self._update_scalar_rates()
            self.CurrVolt()
            num_points = max(2, int((epoch_end_time - current_time) / sampint) + 1)
            t_eval = np.linspace(current_time, epoch_end_time, num_points)
            if len(t_eval) <= 1:
                current_time = epoch_end_time
                continue
            self._reusable_y0[:] = self.pop[:12]
            sol = solve_ivp(
                self.NowDerivs,
                [current_time, epoch_end_time],
                self._reusable_y0,
                method='LSODA',
                t_eval=t_eval,
                rtol=1e-6,
                atol=1e-8
            )
            if sol.success and len(sol.t) > 0:
                batch_size = len(sol.t)
                end_idx = min(store_idx + batch_size, total_points)
                batch_indices = np.arange(store_idx, end_idx)
                actual_batch = len(batch_indices)
                if actual_batch > 0:
                    batch_states = sol.y[:, :actual_batch].T
                    self._store_results_vectorized(
                        batch_indices,
                        batch_states,
                        self.vm
                    )
                    self.pop[:12] = sol.y[:, -1]
                    store_idx = end_idx
            current_time = epoch_end_time
        self.time = np.arange(0, total_points * sampint, sampint)[:total_points]
        return sol.t, self.SimSwp
    def _store_results(self, idx, t):
        """
        Stores the simulation results for a single time point (non-vectorized).
        This method calculates and stores the macroscopic current, open probability,
        inactivated probability, available probability, and command voltage for a
        single time point `idx` in the simulation output arrays (e.g., `self.SimSwp`).
        It uses the current state populations (`self.pop`) and the current
        membrane potential (`self.vm`) to derive these values. The current is
        calculated using the open state probability (state 6, index 5), the
        pre-calculated GHK current factor from `self.iscft`, and scaling factors.
        Args:
            idx (int): The index in the simulation output arrays where results
                       for the current time point should be stored.
            t (float): The current simulation time (not directly used for storage
                       logic itself but often available when this method is called).
        """
        vidx = np.searchsorted(self.vt, self.vm)
        vidx = np.clip(vidx, 0, len(self.vt) - 1)
        open_prob = self.pop[5]                                               
        current = open_prob * self.iscft[vidx] * self.numchan * self.current_scaling
        self.SimSwp[idx] = current
        self.SimOp[idx] = self.pop[5]                                            
        self.SimIn[idx] = np.sum(self.pop[6:])                                                                            
        self.SimAv[idx] = np.sum(self.pop[:6])                                                                    
        self.SimCom[idx] = self.vm
    def _store_results_vectorized(self, indices, batch_states, voltage):
        """
        Stores a batch of simulation results in their respective arrays (vectorized).
        Calculates and stores currents, open probabilities, inactivated probabilities,
        available probabilities, and command voltages for a batch of time points.
        This method is optimized for performance using vectorized numpy operations.
        Args:
            indices (np.ndarray or list): Array of indices in the output arrays
                                          (e.g., `self.SimSwp`) where results
                                          should be stored.
            batch_states (np.ndarray): A 2D array where each row contains the
                                       probabilities for the first 12 states
                                       at a specific time point.
            voltage (np.ndarray or float): A 1D array of membrane potentials or a
                                           single float corresponding to each row in
                                           `batch_states` or for all states if float.
        """
        if len(indices) == 0:
            return
        vidx = np.searchsorted(self.vt, voltage)
        vidx = np.clip(vidx, 0, len(self.vt) - 1)
        current_factor = self.iscft[vidx]
        open_probs = batch_states[:, 5]
        currents = open_probs * current_factor * self.numchan * self.current_scaling
        inactivated = np.sum(batch_states[:, 6:], axis=1)                                         
        available = np.sum(batch_states[:, :6], axis=1)                                    
        self.SimSwp[indices] = currents
        self.SimOp[indices] = open_probs
        self.SimIn[indices] = inactivated
        self.SimAv[indices] = available
        self.SimCom[indices] = voltage
    def create_default_protocol(self, target_voltages=None, holding_potential=-80,
                              holding_duration=98, test_duration=200, tail_duration=2):
        """
        Creates a default multi-step voltage clamp protocol for the legacy Markov model.
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
        This method populates `self.NumSwps` and `self.SwpSeq` (the protocol array).
        It also stores a copy of the protocol under an attribute named `SwpSeq{self.BsNm}`
        (where `self.BsNm` is "MultiStepKeyVoltages"). Finally, it calls
        `self.CurrVolt()` to ensure current-voltage relationships are up to date.
        """
        self.BsNm = "MultiStepKeyVoltages"
        if target_voltages is None:
            target_voltages = [30, 0, -20, -30, -40, -50, -60]
        target_voltages = np.array(target_voltages)
        self.NumSwps = len(target_voltages)
        self.SwpSeq = np.zeros((8, self.NumSwps))
        holding_samples = int(holding_duration / 0.005)
        test_samples = int(test_duration / 0.005)
        tail_samples = int(tail_duration / 0.005)
        total_samples = holding_samples + test_samples + tail_samples
        self.SwpSeq[0, :] = 3                      
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        self.SwpSeq[4, :] = target_voltages
        self.SwpSeq[5, :] = holding_samples + test_samples
        self.SwpSeq[6, :] = holding_potential
        self.SwpSeq[7, :] = total_samples
        assert self.NumSwps == len(target_voltages), "Voltage count mismatch"
        assert np.allclose(self.SwpSeq[4,:], target_voltages), "Voltage assignment error"
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        self.CurrVolt()
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
        self.NumSwps = 1
        self.SwpSeq = np.zeros((10, 1))                               
        sampint = 0.005                          
        holding_duration = 200                             
        holding_samples = int(holding_duration / sampint)
        inactivating_samples = int(inactivating_duration / sampint)  
        test_samples = int(5 / sampint)                                                         
        recovery_samples = int(recovery_duration / sampint)
        self.SwpSeq[0, 0] = 4
        self.SwpSeq[2, 0] = -80                     
        self.SwpSeq[3, 0] = holding_samples
        self.SwpSeq[4, 0] = inactivating_voltage 
        self.SwpSeq[5, 0] = holding_samples + inactivating_samples
        self.SwpSeq[6, 0] = test_voltage
        self.SwpSeq[7, 0] = holding_samples + inactivating_samples + test_samples
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
        if target_recovery_times is None:
            target_recovery_times = [1, 3, 10, 30, 100, 300, 1000]
        target_recovery_times = np.array(target_recovery_times)
        self.NumSwps = len(target_recovery_times)
        self.SwpSeq = np.zeros((12, self.NumSwps))
        sampint = 0.005        
        holding_samples = int(holding_duration / sampint)
        inactivating_samples = int(inactivating_duration / sampint)
        test_samples = int(test_duration / sampint)
        tail_samples = int(tail_duration / sampint)
        recovery_samples = (target_recovery_times / sampint).astype(int)
        self.SwpSeq[0, :] = 5                      
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        self.SwpSeq[4, :] = inactivating_voltage
        self.SwpSeq[5, :] = holding_samples + inactivating_samples
        self.SwpSeq[6, :] = holding_potential
        self.SwpSeq[7, :] = holding_samples + inactivating_samples + recovery_samples
        self.SwpSeq[8, :] = test_voltage
        self.SwpSeq[9, :] = holding_samples + inactivating_samples + recovery_samples + test_samples
        self.SwpSeq[10, :] = holding_potential
        self.SwpSeq[11, :] = holding_samples + inactivating_samples + recovery_samples + test_samples + tail_samples
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        self.CurrVolt()
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
            test_voltages = np.arange(-120, -15, 5)                                
        test_voltages = np.array(test_voltages)
        self.NumSwps = len(test_voltages)
        self.SwpSeq = np.zeros((10, self.NumSwps))
        sampint = 0.005        
        holding_samples = int(200 / sampint)                          
        prepulse_samples = int(prepulse_duration / sampint)
        test_samples = int(test_pulse_duration / sampint)
        recovery_samples = int(recovery_duration / sampint)
        self.SwpSeq[0, :] = 4                      
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        self.SwpSeq[4, :] = test_voltages
        self.SwpSeq[5, :] = holding_samples + prepulse_samples
        self.SwpSeq[6, :] = test_pulse_voltage
        self.SwpSeq[7, :] = holding_samples + prepulse_samples + test_samples
        self.SwpSeq[8, :] = holding_potential
        self.SwpSeq[9, :] = holding_samples + prepulse_samples + test_samples + recovery_samples
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        self.CurrVolt()
class AnticonvulsantMarkovModel(MarkovModel):
    """
    24-state Markov model for voltage-gated sodium channels with anticonvulsant drug binding.
    This model extends the 12-state Kuo-Bean sodium channel model to include drug binding
    states, based on the anticonvulsant binding site characterized by Kuo (1998). The model
    captures state-dependent drug affinity with ~100-fold higher affinity for inactivated
    states compared to resting states.
    State Organization:
        - States 0-5: Drug-free closed/open states (C1-C5, O)
        - States 6-11: Drug-free inactivated states (I1-I6)
        - States 12-17: Drug-bound closed/open states (DC1-DC5, DO)
        - States 18-23: Drug-bound inactivated states (DI1-DI6)
    Attributes:
        num_states (int): Total number of states (24)
        drug_concentration (float): Drug concentration in μM
        drug_type (str): Type of anticonvulsant ('CBZ', 'LTG', 'DPH', or 'MIXED')
        vm (float): Current membrane voltage in mV
        pop (numpy.ndarray): State probability vector (24 elements)
        time (numpy.ndarray): Time vector for simulation results
        SimSwp (numpy.ndarray): Simulated current trace
        SimOp (numpy.ndarray): Open state probability over time
        SimIn (numpy.ndarray): Inactivated state probability over time
        SimDrugBound (numpy.ndarray): Drug-bound fraction over time
    """
    def __init__(self, drug_concentration=0.0, drug_type='DPH'):
        """
        Initialize the anticonvulsant Markov model.
        Parameters:
            drug_concentration (float): Initial drug concentration in μM. Default is 0.0.
            drug_type (str): Type of anticonvulsant drug. Options are:
                - 'CBZ': Carbamazepine
                - 'LTG': Lamotrigine  
                - 'DPH': Phenytoin (Diphenylhydantoin)
                - 'MIXED': Average parameters of all three drugs
                Default is 'mixed'.
        """
        self.NumSwps = 0
        self.num_states = 25                         
        self.drug_concentration = drug_concentration      
        self.drug_type = drug_type.upper()
        self.vm = -80
        self.init_parameters()
        self.init_waves()
        self._update_drug_rates()                                          
        self.CurrVolt()
        self.create_default_protocol()
        self.pop = self.EquilOccup(self.vm)                                            
    def set_drug_type(self, drug_type):
        """
        Change the drug type and update all dependent parameters.
        This method reinitializes drug-specific parameters while maintaining
        the current drug concentration. The equilibrium state distribution
        is recalculated for the new drug type.
        Parameters:
            drug_type (str): New drug type ('CBZ', 'LTG', 'DPH', or 'MIXED')
        """
        self.drug_type = drug_type.upper()
        self.init_parameters()                                                         
        self._update_drug_rates()                                                
        self.pop = self.EquilOccup(self.vm)                                     
    def set_drug_concentration(self, drug_concentration):
        """
        Update the drug concentration and recalculate binding rates.
        This method updates concentration-dependent binding rates and
        recalculates the equilibrium state distribution at the current
        membrane voltage.
        Parameters:
            drug_concentration (float): New drug concentration in μM
        """
        self.drug_concentration = drug_concentration
        self._update_drug_rates()                          
        self.pop = self.EquilOccup(self.vm)                                     
    def init_parameters(self):
        """
        Initialize all model parameters including drug-specific binding kinetics.
        This method sets up:
        1. Original Kuo-Bean sodium channel gating parameters
        2. Drug-specific binding affinities and rates from Kuo 1998
        3. Physical constants and scaling factors
        The drug-specific parameters include:
        - KI_inactivated: Dissociation constant for inactivated states (μM)
        - recovery_tau: Time constant for recovery from drug block (ms)
        - k_off: Drug unbinding rate (1/ms)
        - k_on: Concentration-dependent binding rates (1/ms/μM)
        - k_off_scaling: Calibrated scaling factor to match experimental shifts
        Notes:
            Resting state affinity (KR) is set to 100x weaker than inactivated
            state affinity based on Kuo 1998 findings.
        """
        self.alcoeff = 20     
        self.alslp = 40           
        self.btcoeff = 0.3    
        self.btslp = 18.5      
        self.ConCoeff = 0.004                                          
        self.CoffCoeff = 4.5                                          
        self.ConSlp = 1e8                                           
        self.CoffSlp = 1e8                                          
        self.gmcoeff = 50      
        self.gmslp = 100       
        self.dlcoeff = 0.8
        self.dlslp = 6
        self.OpOnCoeff = 4                          
        self.OpOffCoeff = 0.008                      
        self.ConHiCoeff = 4                                
        self.CoffHiCoeff = 0.008                             
        self.OpOnSlp = 1e8                                          
        self.OpOffSlp = 1e8                                         
        self.alfac = np.sqrt(np.sqrt(self.ConHiCoeff / self.ConCoeff))
        self.btfac = np.sqrt(np.sqrt(self.CoffCoeff / self.CoffHiCoeff))
        self.drug_params = {
            'CBZ': {
                'KI_inactivated': 25.0,                                        
                'recovery_tau': 189.0,                                                     
                'k_off_base': 1.0 / 189.0,                                          
                'k_off_scaling': 0.55                                             
            },
            'LTG': {
                'KI_inactivated': 9.0,                                         
                'recovery_tau': 321.0,                                                     
                'k_off_base': 1.0 / 321.0,                                          
                'k_off_scaling': 0.42                                             
            },
            'DPH': {
                'KI_inactivated': 9.0,                                         
                'recovery_tau': 600.0,                                                         
                'k_off_base': 1.0 / 600.0,                                          
                'k_off_scaling': 0.50                                             
            }
        }
        if self.drug_type in self.drug_params:
            params = self.drug_params[self.drug_type]
        else:
            print(f"Warning: Unknown drug type '{self.drug_type}' (or default 'DPH'), using DPH parameters as fallback.")
            params = self.drug_params['DPH']
        self.KI_inactivated = params['KI_inactivated']
        self.recovery_tau = params['recovery_tau']
        self.k_off = params['k_off_base'] * params.get('k_off_scaling', 1.0)
        self.KR_resting = self.KI_inactivated * 100.0
        self.k_on_inactivated_base = self.k_off / self.KI_inactivated
        self.k_on_resting_base = self.k_off / self.KR_resting
        self.numchan = 1
        self.F = 96485
        self.Rgc = 8314
        self.Tkel = 298
        self.Nao, self.Nai = 150, 15
        self.ClipRate = 6000
        self.current_scaling = 0.0125
        self.PNasc = 1e-5 
        self._reusable_y0 = np.zeros(24)
        self._update_drug_rates()
    def _update_drug_rates(self):
        """
        Update concentration-dependent drug binding rates.
        This private method calculates the actual binding rates based on
        the current drug concentration. The binding rate k_on is proportional
        to drug concentration, while k_off is concentration-independent.
        Updates:
            k_on_resting: Binding rate to resting/closed states (1/ms)
            k_on_inactivated: Binding rate to inactivated states (1/ms)
            k_off_resting: Unbinding rate from resting states (1/ms)
            k_off_inactivated: Unbinding rate from inactivated states (1/ms)
        """
        self.k_on_resting = 0
        self.k_on_inactivated = self.k_on_inactivated_base * self.drug_concentration
        self.k_off_resting = 0
        self.k_off_inactivated = self.k_off
    def init_waves(self):
        """
        Initialize arrays for voltage-dependent rates and state variables.
        This method pre-allocates arrays for:
        - Voltage vector (-200 to +200 mV)
        - State probability vector (24 states)
        - Rate constant arrays for all transitions
        - Current-voltage relationship
        It also calls methods to calculate voltage-dependent rates and
        create the default voltage protocol.
        """
        self.vt = np.arange(-200, 201)
        self.pop = np.zeros(24)  
        self.dstdt = np.zeros(24)
        self._reusable_y0 = np.zeros(24)
        self.iscft = np.zeros_like(self.vt)
        self.create_rate_waves()
        self.stRatesVolt()
    def create_rate_waves(self):
        """
        Create arrays for all voltage-dependent transition rates.
        This method pre-allocates numpy arrays for each transition rate
        in the 24-state model. Rate names follow the convention:
        k[from][to]dis for transitions between states.
        Notes:
            The '_vec' suffix indicates these are vectorized arrays
            containing rates at all voltages in self.vt.
        """
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis',
                     'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis',
                     'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis',
                     'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis',
                     'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis',
                     'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis',
                     'k98dis', 'k87dis']
        for name in rate_names:
            setattr(self, name + '_vec', np.zeros_like(self.vt, dtype=float))
    def stRatesVolt(self):
        """
        Calculate voltage-dependent transition rates for all voltages.
        This method computes all transition rates based on the Kuo-Bean
        formulation with exponential voltage dependence. Rates are
        calculated for the entire voltage range and stored in vectorized
        arrays for efficient lookup during simulations.
        The rates include:
        - Activation/deactivation transitions
        - Fast inactivation/recovery transitions  
        - State-dependent coupling factors
        All rates are clipped at ClipRate (default 6000/ms) to prevent
        numerical instability.
        """
        if not hasattr(self, 'ClipRate') or self.ClipRate is None:
            self.ClipRate = 1000
        if not hasattr(self, 'k12dis_vec'):
            self.create_rate_waves()
        vt = self.vt
        amt = self.alcoeff * np.exp(vt / self.alslp)
        bmt = self.btcoeff * np.exp(-vt / self.btslp)
        gmt = self.gmcoeff * np.exp(vt / self.gmslp)
        dmt = self.dlcoeff * np.exp(-vt / self.dlslp)
        konlo = self.ConCoeff * np.exp(vt / self.ConSlp)
        kofflo = self.CoffCoeff * np.exp(-vt / self.CoffSlp)
        konop = self.OpOnCoeff * np.exp(vt / self.OpOnSlp)
        koffop = self.OpOffCoeff * np.exp(-vt / self.OpOffSlp)
        self.k12dis_vec = np.minimum(4 * amt, self.ClipRate)
        self.k23dis_vec = np.minimum(3 * amt, self.ClipRate)
        self.k34dis_vec = np.minimum(2 * amt, self.ClipRate)
        self.k45dis_vec = np.minimum(amt, self.ClipRate)
        self.k56dis_vec = np.minimum(gmt, self.ClipRate)
        self.k65dis_vec = np.minimum(dmt, self.ClipRate)
        self.k54dis_vec = np.minimum(4 * bmt, self.ClipRate)
        self.k43dis_vec = np.minimum(3 * bmt, self.ClipRate)
        self.k32dis_vec = np.minimum(2 * bmt, self.ClipRate)
        self.k21dis_vec = np.minimum(bmt, self.ClipRate)
        dph = 1
        self.k17dis_vec = np.minimum(konlo * dph, self.ClipRate)
        self.k71dis_vec = np.minimum(kofflo, self.ClipRate)
        self.k28dis_vec = np.minimum(self.k17dis_vec * self.alfac, self.ClipRate)
        self.k82dis_vec = np.minimum(self.k71dis_vec / self.btfac, self.ClipRate)
        self.k39dis_vec = np.minimum(self.k17dis_vec * self.alfac**2, self.ClipRate)
        self.k93dis_vec = np.minimum(self.k71dis_vec / (self.btfac**2), self.ClipRate)
        self.k410dis_vec = np.minimum(self.k17dis_vec * self.alfac**3, self.ClipRate)
        self.k104dis_vec = np.minimum(self.k71dis_vec / (self.btfac**3), self.ClipRate)
        self.k511dis_vec = np.minimum(self.k17dis_vec * self.alfac**4, self.ClipRate)
        self.k115dis_vec = np.minimum(self.k71dis_vec / (self.btfac**4), self.ClipRate)
        self.k612dis_vec = np.minimum(konop, self.ClipRate)
        self.k126dis_vec = np.minimum(koffop, self.ClipRate)
        self.k78dis_vec = np.minimum(4 * amt * self.alfac, self.ClipRate)
        self.k89dis_vec = np.minimum(3 * amt * self.alfac, self.ClipRate)
        self.k910dis_vec = np.minimum(2 * amt * self.alfac, self.ClipRate)
        self.k1011dis_vec = np.minimum(amt * self.alfac, self.ClipRate)
        self.k1112dis_vec = np.minimum(gmt, self.ClipRate)
        self.k1110dis_vec = np.minimum(4 * bmt * (1/self.btfac), self.ClipRate)
        self.k109dis_vec = np.minimum(3 * bmt * (1/self.btfac), self.ClipRate)
        self.k98dis_vec = np.minimum(2 * bmt * (1/self.btfac), self.ClipRate)
        self.k87dis_vec = np.minimum(bmt * (1/self.btfac), self.ClipRate)
        k115_safe = np.where(self.k115dis_vec > 0, self.k115dis_vec, 1.0)
        self.k1211dis_vec = np.minimum(
            (self.k65dis_vec * self.k511dis_vec * self.k126dis_vec) / 
            (self.k612dis_vec * k115_safe), 
            self.ClipRate
        )
        self._update_scalar_rates()
    def _update_scalar_rates(self):
        """
        Extract scalar rate values at the current membrane voltage.
        This private method looks up the transition rates at the current
        voltage (self.vm) from the pre-calculated rate arrays and stores
        them as scalar attributes for use in the ODE solver.
        """
        vidx = np.argmin(np.abs(self.vt - self.vm))
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis',
                     'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis',
                     'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis',
                     'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis',
                     'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis',
                     'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis',
                     'k98dis', 'k87dis']
        for name in rate_names:
            vec_name = name + '_vec'
            if hasattr(self, vec_name):
                vec_array = getattr(self, vec_name)
                if isinstance(vec_array, np.ndarray) and len(vec_array) > vidx:
                    setattr(self, name, vec_array[vidx])
                else:
                    setattr(self, name, 0.0)
            else:
                setattr(self, name, 0.0)
    def CurrVolt(self):
        """
        Calculate the current-voltage relationship using GHK equation.
        This method computes the single-channel current at each voltage
        using the Goldman-Hodgkin-Katz (GHK) current equation. The
        calculation accounts for the sodium gradient and temperature.
        Updates:
            self.iscft: Array of single-channel currents at each voltage
        Notes:
            Special handling is included for voltages near 0 mV to avoid
            numerical issues with the GHK equation.
        """
        scaled_PNasc = self.PNasc
        v_volts = self.vt * 1e-3
        near_zero = np.abs(v_volts) < 1e-6
        not_zero = ~near_zero
        self.iscft = np.zeros_like(v_volts)
        if np.any(near_zero):
            du2_zero = self.F * self.F / (self.Rgc * self.Tkel)
            self.iscft[near_zero] = scaled_PNasc * du2_zero * (self.Nai - self.Nao)
        if np.any(not_zero):
            v_nz = v_volts[not_zero]
            du1 = (v_nz * self.F) / (self.Rgc * self.Tkel)
            du3 = np.exp(-du1)
            du5_corrected = self.F * du1 * (self.Nai - self.Nao * du3) / (1 - du3)
            self.iscft[not_zero] = scaled_PNasc * du5_corrected
    def EquilOccup(self, vm):
        """
        Calculate equilibrium state occupancies at a given voltage.
        This method computes the steady-state probability distribution
        across all 24 states at the specified membrane voltage. It uses
        the principle of detailed balance to calculate equilibrium
        occupancies analytically.
        Parameters:
            vm (float): Membrane voltage in mV
        Returns:
            numpy.ndarray: State probability vector (24 elements) at equilibrium
        Notes:
            The calculation accounts for both intrinsic gating equilibria
            and drug binding equilibria. Drug binding factors depend on
            the current drug concentration.
        """
        self.vm = vm
        if not hasattr(self, 'k12dis_vec'):
            self.create_rate_waves()
        self.stRatesVolt()
        self._update_scalar_rates()
        def safe_div(a, b, default=0.0):
            if np.isscalar(b):
                if abs(b) > 1e-10:
                    return a / b
                else:
                    return default
            else:
                result = np.full_like(a, default, dtype=float)
                mask = np.abs(b) > 1e-10
                if np.any(mask):
                    result[mask] = a[mask] / b[mask]
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
        drug_factor_inactivated = self.k_on_inactivated / self.k_off_inactivated
        dusuma_free = 1 + du1 + du1*du2 + du1*du2*du3 + du1*du2*du3*du4 + du1*du2*du3*du4*du5
        dusumb_free = du7 + du7*du8 + du7*du8*du9 + du7*du8*du9*du10 + du7*du8*du9*du10*du11 + du7*du8*du9*du10*du11*du12
        dusuma_drug = drug_factor_closed * dusuma_free
        dusumb_drug = drug_factor_inactivated * dusumb_free
        dusum_total = dusuma_free + dusumb_free + dusuma_drug + dusumb_drug
        pop = np.zeros(24)
        if dusum_total > 1e-10:
            closed_products = np.array([1, du1, du1*du2, du1*du2*du3, du1*du2*du3*du4, du1*du2*du3*du4*du5])
            pop[0:6] = closed_products / dusum_total
            inact_products = np.array([du7, du7*du8, du7*du8*du9, du7*du8*du9*du10, du7*du8*du9*du10*du11, du7*du8*du9*du10*du11*du12])
            pop[6:12] = inact_products / dusum_total
            pop[12:18] = drug_factor_closed * closed_products / dusum_total
            pop[18:24] = drug_factor_inactivated * inact_products / dusum_total
        else:
            pop[0] = 0.98                           
            pop[1] = 0.02
        pop = np.nan_to_num(pop, nan=0.0)
        return pop
    def NowDerivs(self, t, y):
        """
        Calculate state derivatives for the ODE solver.
        This method computes dy/dt for all 24 states based on the
        transition rate matrix Q. It constructs the full Q matrix
        including intrinsic gating transitions and drug binding/unbinding
        transitions.
        Parameters:
            t (float): Current time (not used, but required by ODE solver)
            y (numpy.ndarray): Current state probability vector (24 elements)
        Returns:
            numpy.ndarray: State derivatives dy/dt (24 elements)
        Notes:
            The Q matrix structure:
            - Q[i,j] = rate from state j to state i (off-diagonal)
            - Q[i,i] = negative sum of all rates leaving state i (diagonal)
            Drug-bound open state (state 17) is non-conducting (blocked).
        """
        vidx = np.searchsorted(self.vt, self.vm)
        vidx = np.clip(vidx, 0, len(self.vt) - 1)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return np.zeros_like(y)
        k12dis  = self.k12dis_vec[vidx]; k21dis  = self.k21dis_vec[vidx]
        k23dis  = self.k23dis_vec[vidx]; k32dis  = self.k32dis_vec[vidx]
        k34dis  = self.k34dis_vec[vidx]; k43dis  = self.k43dis_vec[vidx]
        k45dis  = self.k45dis_vec[vidx]; k54dis  = self.k54dis_vec[vidx]
        k56dis  = self.k56dis_vec[vidx]; k65dis  = self.k65dis_vec[vidx]
        k17dis  = self.k17dis_vec[vidx]; k71dis  = self.k71dis_vec[vidx]
        k28dis  = self.k28dis_vec[vidx]; k82dis  = self.k82dis_vec[vidx]
        k39dis  = self.k39dis_vec[vidx]; k93dis  = self.k93dis_vec[vidx]
        k410dis = self.k410dis_vec[vidx]; k104dis = self.k104dis_vec[vidx]
        k511dis = self.k511dis_vec[vidx]; k115dis = self.k115dis_vec[vidx]
        k612dis = self.k612dis_vec[vidx]; k126dis = self.k126dis_vec[vidx]
        k78dis  = self.k78dis_vec[vidx]; k87dis  = self.k87dis_vec[vidx]
        k89dis  = self.k89dis_vec[vidx]; k98dis  = self.k98dis_vec[vidx]
        k910dis = self.k910dis_vec[vidx]; k109dis = self.k109dis_vec[vidx]
        k1011dis= self.k1011dis_vec[vidx]; k1110dis= self.k1110dis_vec[vidx]
        k1112dis= self.k1112dis_vec[vidx]; k1211dis= self.k1211dis_vec[vidx]
        k_on_closed = self.k_on_resting
        k_off_closed = self.k_off_resting
        k_on_inact = self.k_on_inactivated
        k_off_inact = self.k_off_inactivated
        st = y.copy()
        Q = np.zeros((24, 24))
        Q[0, 1] = k21dis; Q[0, 6] = k71dis
        Q[1, 0] = k12dis; Q[1, 2] = k32dis; Q[1, 7] = k82dis
        Q[2, 1] = k23dis; Q[2, 3] = k43dis; Q[2, 8] = k93dis
        Q[3, 2] = k34dis; Q[3, 4] = k54dis; Q[3, 9] = k104dis
        Q[4, 3] = k45dis; Q[4, 5] = k65dis; Q[4, 10] = k115dis
        Q[5, 4] = k56dis; Q[5, 11] = k126dis
        Q[6, 0] = k17dis; Q[6, 7] = k87dis
        Q[7, 1] = k28dis; Q[7, 6] = k78dis; Q[7, 8] = k98dis
        Q[8, 2] = k39dis; Q[8, 7] = k89dis; Q[8, 9] = k109dis
        Q[9, 3] = k410dis; Q[9, 8] = k910dis; Q[9, 10] = k1110dis
        Q[10, 4] = k511dis; Q[10, 9] = k1011dis; Q[10, 11] = k1211dis
        Q[11, 5] = k612dis; Q[11, 10] = k1112dis
        Q[12, 13] = k21dis; Q[12, 18] = k71dis
        Q[13, 12] = k12dis; Q[13, 14] = k32dis; Q[13, 19] = k82dis
        Q[14, 13] = k23dis; Q[14, 15] = k43dis; Q[14, 20] = k93dis
        Q[15, 14] = k34dis; Q[15, 16] = k54dis; Q[15, 21] = k104dis
        Q[16, 15] = k45dis; Q[16, 17] = k65dis; Q[16, 22] = k115dis
        Q[17, 16] = k56dis; Q[17, 23] = k126dis
        Q[18, 12] = k17dis; Q[18, 19] = k87dis
        Q[19, 13] = k28dis; Q[19, 18] = k78dis; Q[19, 20] = k98dis
        Q[20, 14] = k39dis; Q[20, 19] = k89dis; Q[20, 21] = k109dis
        Q[21, 15] = k410dis; Q[21, 20] = k910dis; Q[21, 22] = k1110dis
        Q[22, 16] = k511dis; Q[22, 21] = k1011dis; Q[22, 23] = k1211dis
        Q[23, 17] = k612dis; Q[23, 22] = k1112dis
        Q[12, 0] = k_on_closed; Q[0, 12] = k_off_closed             
        Q[13, 1] = k_on_closed; Q[1, 13] = k_off_closed             
        Q[14, 2] = k_on_closed; Q[2, 14] = k_off_closed             
        Q[15, 3] = k_on_closed; Q[3, 15] = k_off_closed             
        Q[16, 4] = k_on_closed; Q[4, 16] = k_off_closed             
        Q[17, 5] = k_on_closed; Q[5, 17] = k_off_closed            
        Q[18, 6] = k_on_inact; Q[6, 18] = k_off_inact               
        Q[19, 7] = k_on_inact; Q[7, 19] = k_off_inact               
        Q[20, 8] = k_on_inact; Q[8, 20] = k_off_inact               
        Q[21, 9] = k_on_inact; Q[9, 21] = k_off_inact               
        Q[22, 10] = k_on_inact; Q[10, 22] = k_off_inact             
        Q[23, 11] = k_on_inact; Q[11, 23] = k_off_inact             
        Q[0,0]   = -(k12dis + k17dis + k_on_closed)
        Q[1,1]   = -(k21dis + k23dis + k28dis + k_on_closed)
        Q[2,2]   = -(k32dis + k34dis + k39dis + k_on_closed)
        Q[3,3]   = -(k43dis + k45dis + k410dis + k_on_closed)
        Q[4,4]   = -(k54dis + k56dis + k511dis + k_on_closed)
        Q[5,5]   = -(k65dis + k612dis + k_on_closed)
        Q[6,6]   = -(k71dis + k78dis + k_on_inact)
        Q[7,7]   = -(k82dis + k87dis + k89dis + k_on_inact)
        Q[8,8]   = -(k93dis + k98dis + k910dis + k_on_inact)
        Q[9,9]   = -(k104dis + k109dis + k1011dis + k_on_inact)
        Q[10,10] = -(k115dis + k1110dis + k1112dis + k_on_inact)
        Q[11,11] = -(k126dis + k1211dis + k_on_inact)
        Q[12,12] = -(k12dis + k17dis + k_off_closed) 
        Q[13,13] = -(k21dis + k23dis + k28dis + k_off_closed) 
        Q[14,14] = -(k32dis + k34dis + k39dis + k_off_closed) 
        Q[15,15] = -(k43dis + k45dis + k410dis + k_off_closed) 
        Q[16,16] = -(k54dis + k56dis + k511dis + k_off_closed) 
        Q[17,17] = -(k65dis + k612dis + k_off_closed) 
        Q[18,18] = -(k71dis + k78dis + k_off_inact) 
        Q[19,19] = -(k82dis + k87dis + k89dis + k_off_inact) 
        Q[20,20] = -(k93dis + k98dis + k910dis + k_off_inact) 
        Q[21,21] = -(k104dis + k109dis + k1011dis + k_off_inact) 
        Q[22,22] = -(k115dis + k1110dis + k1112dis + k_off_inact) 
        Q[23,23] = -(k126dis + k1211dis + k_off_inact) 
        dstdt = np.zeros_like(y)
        for i in range(24):
            for j in range(24):
                dstdt[i] += Q[i, j] * st[j]
        if np.any(np.isnan(dstdt)) or np.any(np.isinf(dstdt)):
            return np.zeros_like(st)
        return dstdt
    def Sweep(self, SwpNo):
        """
        Execute a single voltage-clamp sweep from the protocol.
        This method simulates the response to a voltage protocol sweep,
        solving the differential equations for state evolution and
        calculating the resulting current trace.
        Parameters:
            SwpNo (int): Sweep number to execute (0-indexed)
        Returns:
            tuple: (time_points, current_trace)
                - time_points: Time vector from ODE solver
                - current_trace: Simulated current trace
        Updates:
            self.SimSwp: Complete current trace
            self.SimOp: Open state probability over time
            self.SimIn: Inactivated state probability over time
            self.SimAv: Available (closed) state probability over time
            self.SimDrugBound: Drug-bound fraction over time
            self.time: Time vector for the sweep
        Notes:
            Uses scipy's solve_ivp with LSODA method for efficient
            integration of the stiff ODE system.
        """
        if SwpNo >= self.SwpSeq.shape[1] or SwpNo < 0:
            raise ValueError(f"Invalid sweep number {SwpNo}")
        SwpSeq = self.SwpSeq
        NumEpchs = int(SwpSeq[0, SwpNo])
        if NumEpchs <= 0 or 2*NumEpchs + 1 >= SwpSeq.shape[0]:
            raise ValueError("Invalid number of epochs in protocol")
        total_points = int(SwpSeq[2*NumEpchs + 1, SwpNo]) + 1
        sampint = 0.005
        self.SimSwp = np.zeros(total_points)
        self.SimOp = np.zeros(total_points)
        self.SimIn = np.zeros(total_points)
        self.SimAv = np.zeros(total_points)
        self.SimCom = np.zeros(total_points)
        self.SimDrugBound = np.zeros(total_points)                             
        epoch_voltages = np.zeros(NumEpchs + 1)
        epoch_end_times = np.zeros(NumEpchs + 1)
        epoch_voltages[0] = SwpSeq[2, SwpNo]
        epoch_end_times[0] = 0.0
        for e in range(1, NumEpchs + 1):
            epoch_voltages[e] = SwpSeq[2 * e, SwpNo]
            epoch_end_times[e] = int(SwpSeq[2 * e + 1, SwpNo]) * sampint
        self.vm = epoch_voltages[0]
        self.CurrVolt()
        self.pop = self.EquilOccup(self.vm)
        self._store_results_24(0, 0)
        current_time = 0.0
        store_idx = 1
        for epoch in range(1, NumEpchs + 1):
            self.vm = epoch_voltages[epoch]
            epoch_end_time = epoch_end_times[epoch]
            self._update_scalar_rates()
            self.CurrVolt()
            num_points = max(2, int((epoch_end_time - current_time) / sampint) + 1)
            t_eval = np.linspace(current_time, epoch_end_time, num_points)
            if len(t_eval) <= 1:
                current_time = epoch_end_time
                continue
            self._reusable_y0[:] = self.pop[:24]
            sol = solve_ivp(
                self.NowDerivs,
                [current_time, epoch_end_time],
                self._reusable_y0,
                method='LSODA',
                t_eval=t_eval,
                rtol=1e-6,
                atol=1e-8
            )
            self.full_sol_t = sol.t
            self.full_sol_y = sol.y
            if sol.success and len(sol.t) > 0:
                batch_size = len(sol.t)
                end_idx = min(store_idx + batch_size, total_points)
                batch_indices = np.arange(store_idx, end_idx)
                actual_batch = len(batch_indices)
                if actual_batch > 0:
                    batch_states = sol.y[:, :actual_batch].T
                    self._store_results_vectorized_24(batch_indices, batch_states, self.vm)
                    self.pop[:24] = sol.y[:, -1]
                    store_idx = end_idx
            current_time = epoch_end_time
        self.time = np.arange(0, total_points * sampint, sampint)[:total_points]
        return sol.t, self.SimSwp
    def _store_results_24(self, idx, t):
        """
        Store simulation results for a single time point (24-state model).
        This private method calculates and stores the current and state
        probabilities at a single time point. Only drug-free open channels
        contribute to current as drug-bound channels are blocked.
        Parameters:
            idx (int): Index in result arrays
            t (float): Current time (not used but kept for compatibility)
        """
        vidx = np.searchsorted(self.vt, self.vm)
        vidx = np.clip(vidx, 0, len(self.vt) - 1)
        open_prob_free = self.pop[5]                          
        open_prob_drug = self.pop[17]                                    
        conducting_open_prob = open_prob_free
        current = conducting_open_prob * self.iscft[vidx] * self.numchan * self.current_scaling
        self.SimSwp[idx] = current
        self.SimOp[idx] = open_prob_free + open_prob_drug                          
        self.SimIn[idx] = np.sum(self.pop[6:12]) + np.sum(self.pop[18:24])                   
        self.SimAv[idx] = np.sum(self.pop[:6])                   
        self.SimCom[idx] = self.vm
        self.SimDrugBound[idx] = np.sum(self.pop[12:24])                         
    def _store_results_vectorized_24(self, indices, batch_states, voltage):
        """
        Store simulation results for multiple time points (vectorized).
        This private method efficiently processes and stores results for
        multiple time points simultaneously using numpy vectorization.
        Parameters:
            indices (numpy.ndarray): Indices in result arrays
            batch_states (numpy.ndarray): State probabilities (n_points x 24)
            voltage (float): Membrane voltage for this epoch
        """
        if len(indices) == 0:
            return
        vidx = np.searchsorted(self.vt, voltage)
        vidx = np.clip(vidx, 0, len(self.vt) - 1)
        current_factor = self.iscft[vidx]
        conducting_open_probs = batch_states[:, 5]                             
        total_open_probs = batch_states[:, 5] + batch_states[:, 17]                           
        currents = conducting_open_probs * current_factor * self.numchan * self.current_scaling
        inactivated = np.sum(batch_states[:, 6:12], axis=1) + np.sum(batch_states[:, 18:24], axis=1)
        available = np.sum(batch_states[:, :6], axis=1)
        drug_bound = np.sum(batch_states[:, 12:24], axis=1)
        self.SimSwp[indices] = currents
        self.SimOp[indices] = total_open_probs                                
        self.SimIn[indices] = inactivated
        self.SimAv[indices] = available
        self.SimCom[indices] = voltage
        self.SimDrugBound[indices] = drug_bound
    def create_default_protocol(self, target_voltages=None, holding_potential=-80,
                              holding_duration=98, test_duration=200, tail_duration=2):
        """
        Create a standard voltage-clamp protocol for activation curves.
        This method creates a multi-sweep protocol with test pulses to
        different voltages, suitable for measuring activation curves and
        peak current-voltage relationships.
        Parameters:
            target_voltages (list or None): Test pulse voltages in mV.
                Default: [30, 0, -20, -30, -40, -50, -60]
            holding_potential (float): Holding voltage in mV. Default: -80
            holding_duration (float): Initial holding period in ms. Default: 98
            test_duration (float): Test pulse duration in ms. Default: 200
            tail_duration (float): Final recovery period in ms. Default: 2
        Updates:
            self.SwpSeq: Protocol array defining all sweeps
            self.NumSwps: Number of sweeps in protocol
        """
        self.BsNm = "AnticonvulsantProtocol"
        if target_voltages is None:
            target_voltages = [30, 0, -20, -30, -40, -50, -60]
        target_voltages = np.array(target_voltages)
        self.NumSwps = len(target_voltages)
        self.SwpSeq = np.zeros((8, self.NumSwps))
        holding_samples = int(holding_duration / 0.005)
        test_samples = int(test_duration / 0.005)
        tail_samples = int(tail_duration / 0.005)
        self.SwpSeq[0, :] = 3
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        self.SwpSeq[4, :] = target_voltages
        self.SwpSeq[5, :] = holding_samples + test_samples
        self.SwpSeq[6, :] = holding_potential
        self.SwpSeq[7, :] = holding_samples + test_samples + tail_samples
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        self.CurrVolt()
    def create_inactivation_protocol(self, inactivating_voltage=-20, test_voltage=0, 
                                inactivating_duration=2000, recovery_duration=100):
        """
        Create a protocol optimized to show anticonvulsant effects on inactivation.
        This protocol is designed to maximize drug binding during a long
        inactivating prepulse, followed by a brief test pulse to measure
        the remaining available current. The 2-second default duration is
        critical for reaching steady-state drug binding.
        Parameters:
            inactivating_voltage (float): Voltage for inactivating prepulse in mV. 
                Default: -20 (promotes strong inactivation)
            test_voltage (float): Voltage for test pulse in mV. 
                Default: 0 (maximal channel opening)
            inactivating_duration (float): Duration of inactivating pulse in ms.
                Default: 2000 (2 seconds for complete drug equilibration)
                WARNING: Shorter durations will underestimate drug effects!
            recovery_duration (float): Final recovery period in ms. Default: 100
        Protocol structure:
            1. Hold at -80 mV (200 ms) - equilibration
            2. Long step to inactivating voltage - promotes drug binding
            3. Brief test pulse (5 ms) - measures available current
            4. Return to holding potential - recovery
        Notes:
            Kuo 1998 shows drug binding requires >1 second to reach
            steady state. The 2-second default is essential for accurate
            measurement of anticonvulsant potency.
        """
        self.BsNm = "InactivationProtocol"
        self.NumSwps = 1
        self.SwpSeq = np.zeros((10, 1))                               
        sampint = 0.005                          
        holding_duration = 200                             
        holding_samples = int(holding_duration / sampint)
        inactivating_samples = int(inactivating_duration / sampint)  
        test_samples = int(5 / sampint)                                                         
        recovery_samples = int(recovery_duration / sampint)
        self.SwpSeq[0, 0] = 4
        self.SwpSeq[2, 0] = -80                     
        self.SwpSeq[3, 0] = holding_samples
        self.SwpSeq[4, 0] = inactivating_voltage 
        self.SwpSeq[5, 0] = holding_samples + inactivating_samples
        self.SwpSeq[6, 0] = test_voltage
        self.SwpSeq[7, 0] = holding_samples + inactivating_samples + test_samples
        self.SwpSeq[8, 0] = -80
        self.SwpSeq[9, 0] = holding_samples + inactivating_samples + test_samples + recovery_samples
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        self.CurrVolt()
    def create_recovery_protocol(self, target_recovery_times=None, holding_potential=-80,
                        inactivating_voltage=-20, test_voltage=0,
                        holding_duration=200, inactivating_duration=2000, 
                        test_duration=20, tail_duration=100):
        """
        Create a protocol to measure recovery from drug-induced inactivation.
        This protocol applies a long inactivating pulse to promote drug
        binding, then varies the recovery interval before a test pulse.
        This reveals the slow recovery kinetics characteristic of
        anticonvulsant unbinding.
        Parameters:
            target_recovery_times (list or None): Recovery intervals in ms.
                Default: [1, 3, 10, 30, 100, 300, 1000] (log spacing)
            holding_potential (float): Recovery voltage in mV. Default: -80
            inactivating_voltage (float): Inactivating voltage in mV. Default: -20
            test_voltage (float): Test pulse voltage in mV. Default: 0
            holding_duration (float): Initial equilibration in ms. Default: 200
            inactivating_duration (float): Inactivating pulse in ms. Default: 2000
            test_duration (float): Test pulse duration in ms. Default: 20
            tail_duration (float): Final recovery period in ms. Default: 100
        Protocol structure (per sweep):
            1. Initial holding period
            2. Long inactivating pulse (drug binding)
            3. Variable recovery interval at holding potential
            4. Test pulse to measure recovery
            5. Final tail period
        Notes:
            Recovery time constants differ between drugs:
            - CBZ: ~189 ms
            - LTG: ~321 ms (slowest)
            - DPH: ~189 ms
        """
        self.BsNm = "RecoveryFromInactivation"
        if target_recovery_times is None:
            target_recovery_times = [1, 3, 10, 30, 100, 300, 1000]
        target_recovery_times = np.array(target_recovery_times)
        self.NumSwps = len(target_recovery_times)
        self.SwpSeq = np.zeros((12, self.NumSwps))
        sampint = 0.005        
        holding_samples = int(holding_duration / sampint)
        inactivating_samples = int(inactivating_duration / sampint)
        test_samples = int(test_duration / sampint)
        tail_samples = int(tail_duration / sampint)
        recovery_samples = (target_recovery_times / sampint).astype(int)
        self.SwpSeq[0, :] = 5                      
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        self.SwpSeq[4, :] = inactivating_voltage
        self.SwpSeq[5, :] = holding_samples + inactivating_samples
        self.SwpSeq[6, :] = holding_potential
        self.SwpSeq[7, :] = holding_samples + inactivating_samples + recovery_samples
        self.SwpSeq[8, :] = test_voltage
        self.SwpSeq[9, :] = holding_samples + inactivating_samples + recovery_samples + test_samples
        self.SwpSeq[10, :] = holding_potential
        self.SwpSeq[11, :] = holding_samples + inactivating_samples + recovery_samples + test_samples + tail_samples
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        self.CurrVolt()
    def create_steady_state_inactivation_protocol(self, test_voltages=None, 
                                                holding_potential=-120,
                                                prepulse_duration=2000,
                                                test_pulse_voltage=0,
                                                test_pulse_duration=5,
                                                recovery_duration=100):
        """
        Create a protocol for steady-state inactivation curves.
        This protocol varies the prepulse voltage to achieve different
        levels of steady-state inactivation, then applies a standard
        test pulse. The resulting curve shows voltage-dependent channel
        availability and the shift caused by drug binding.
        Parameters:
            test_voltages (numpy.ndarray or None): Prepulse voltages in mV.
                Default: -120 to -20 mV in 5 mV steps
            holding_potential (float): Initial holding voltage in mV. Default: -120
            prepulse_duration (float): Prepulse duration in ms. Default: 2000
            test_pulse_voltage (float): Test voltage in mV. Default: 0
            test_pulse_duration (float): Test duration in ms. Default: 5
            recovery_duration (float): Final recovery in ms. Default: 100
        Protocol structure (per sweep):
            1. Initial holding at very negative potential
            2. Long prepulse at variable voltage
            3. Brief test pulse at depolarized potential
            4. Recovery period
        Notes:
            The 2-second prepulse duration ensures steady-state drug
            binding at each voltage. Shorter durations will not reveal
            the full magnitude of the drug-induced shift.
        """
        self.BsNm = "SteadyStateInactivation"
        if test_voltages is None:
            test_voltages = np.arange(-120, -15, 5)                                
        test_voltages = np.array(test_voltages)
        self.NumSwps = len(test_voltages)
        self.SwpSeq = np.zeros((10, self.NumSwps))
        sampint = 0.005        
        holding_samples = int(200 / sampint)                          
        prepulse_samples = int(prepulse_duration / sampint)
        test_samples = int(test_pulse_duration / sampint)
        recovery_samples = int(recovery_duration / sampint)
        self.SwpSeq[0, :] = 4                      
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        self.SwpSeq[4, :] = test_voltages
        self.SwpSeq[5, :] = holding_samples + prepulse_samples
        self.SwpSeq[6, :] = test_pulse_voltage
        self.SwpSeq[7, :] = holding_samples + prepulse_samples + test_samples
        self.SwpSeq[8, :] = holding_potential
        self.SwpSeq[9, :] = holding_samples + prepulse_samples + test_samples + recovery_samples
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        self.CurrVolt()