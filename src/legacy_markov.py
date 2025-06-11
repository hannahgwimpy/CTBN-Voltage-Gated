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
        self.NumSwps = 0  # Initialize with default
        self.num_states = 13 # Explicitly define the number of states
        
        # Initialize membrane voltage
        self.vm = -80  # Default holding potential
        
        # Initialize parameters and waves
        self.init_parameters()
        self.init_waves()
        
        # Calculate rates and currents
        self.stRatesVolt()
        self.CurrVolt()
        self.create_default_protocol()  # Generate protocol on instantiation
    
    def init_parameters(self):
        """Initialize biophysical parameters."""
        
        # Original Kuo-Bean parameters
        self.alcoeff = 150      
        self.alslp = 20           
        self.btcoeff = 3      
        self.btslp = 20       
        
        # Inactivation parameters 
        self.ConCoeff = 0.005
        self.CoffCoeff = 0.5
        self.ConSlp = 1e8       
        self.CoffSlp = 1e8      
        
        # Transition rates between states
        self.gmcoeff = 150      
        self.gmslp = 1e12       
        self.dlcoeff = 40       
        self.dlslp = 1e12       
        self.epcoeff = 1.75   
        self.epslp = 1e12       
        self.ztcoeff = 0.03    
        self.ztslp = 25       
        
        # Open state transitions
        self.OpOnCoeff = 0.75   
        self.OpOffCoeff = 0.005 
        self.ConHiCoeff = 0.75   
        self.CoffHiCoeff = 0.005
        self.OpOnSlp = 1e8      
        self.OpOffSlp = 1e8  
        
        # Initialize derived parameters
        self.konlo = self.kofflo = self.konhi = self.koffhi = 0
        self.konop = self.koffop = self.kdlo = self.kdhi = 0
        
        # Calculate alfac and btfac as in Igor code
        self.alfac = np.sqrt(np.sqrt(self.ConHiCoeff / self.ConCoeff))
        self.btfac = np.sqrt(np.sqrt(self.CoffCoeff / self.CoffHiCoeff))
        
        # Other model parameters
        self.numchan = 1
        self.cm = 30
        self.F = 96480
        self.Rgc = 8314
        self.Tkel = 298
        self.Nao, self.Nai = 155, 15
        self.ClipRate = 6000
        
        # Current scaling factor to match HH model
        self.current_scaling = 0.0117

        # Initialize membrane voltage
        self.vm = -80  # Default holding potential
        
        self.PNasc = 1e-5 
        
        # Pre-allocate reusable arrays
        self._reusable_y0 = np.zeros(12)  # For ODE initial conditions - 12 states as used in NowDerivs
    
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
        
        # Create state array with 13 elements (indices 0-12) to match NowDerivs
        # Where 0-11 are the actual states used in ODE solver
        self.pop = np.zeros(13)  
        self.dstdt = np.zeros(12)  # 12 states for derivatives
        
        # Pre-allocate reusable array for ODE solver with 12 states
        # exactly matching what NowDerivs expects
        self._reusable_y0 = np.zeros(12)
        
        # Initialize current arrays
        # Only keep sodium current
        self.iscft = np.zeros_like(self.vt)
        
        self.create_rate_waves()
        self.stRatesVolt()  # Initialize rate constants

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
        
        # Create vectorized rate arrays
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

        # Set default ClipRate if not present
        if not hasattr(self, 'ClipRate') or self.ClipRate is None:
            self.ClipRate = 1000
        
        # First, ensure rate wave arrays exist
        if not hasattr(self, 'k12dis_vec'):
            self.create_rate_waves()
        
        # Use all voltages for vectorized computation
        vt = self.vt
        
        # Vectorized rate calculations
        amt = self.alcoeff * np.exp(vt / self.alslp)
        bmt = self.btcoeff * np.exp(-vt / self.btslp)
        gmt = self.gmcoeff * np.exp(vt / self.gmslp)
        dmt = self.dlcoeff * np.exp(-vt / self.dlslp)
        emt = self.epcoeff * np.exp(vt / self.epslp)
        zmt = self.ztcoeff * np.exp(-vt / self.ztslp)
        
        # Vectorized inactivation rates
        konlo = self.ConCoeff * np.exp(vt / self.ConSlp)
        kofflo = self.CoffCoeff * np.exp(-vt / self.CoffSlp)
        konop = self.OpOnCoeff * np.exp(vt / self.OpOnSlp)
        koffop = self.OpOffCoeff * np.exp(-vt / self.OpOffSlp)
        
        # Vectorized clipping using np.minimum
        # Forward rates (activation)
        self.k12dis_vec = np.minimum(4 * amt, self.ClipRate)
        self.k23dis_vec = np.minimum(3 * amt, self.ClipRate)
        self.k34dis_vec = np.minimum(2 * amt, self.ClipRate)
        self.k45dis_vec = np.minimum(amt, self.ClipRate)
        self.k56dis_vec = np.minimum(gmt, self.ClipRate)
        
        # Backward rates (deactivation)
        self.k65dis_vec = np.minimum(dmt, self.ClipRate)
        self.k54dis_vec = np.minimum(4 * bmt, self.ClipRate)
        self.k43dis_vec = np.minimum(3 * bmt, self.ClipRate)
        self.k32dis_vec = np.minimum(2 * bmt, self.ClipRate)
        self.k21dis_vec = np.minimum(bmt, self.ClipRate)
        
        # Inactivation transitions
        dph = 1  # Inactivating particle
        self.k17dis_vec = np.minimum(konlo * dph, self.ClipRate)
        self.k71dis_vec = np.minimum(kofflo, self.ClipRate)
        
        # Vectorized alfac/btfac scaling
        self.k28dis_vec = np.minimum(self.k17dis_vec * self.alfac, self.ClipRate)
        self.k82dis_vec = np.minimum(self.k71dis_vec / self.btfac, self.ClipRate)
        self.k39dis_vec = np.minimum(self.k17dis_vec * self.alfac**2, self.ClipRate)
        self.k93dis_vec = np.minimum(self.k71dis_vec / (self.btfac**2), self.ClipRate)
        self.k410dis_vec = np.minimum(self.k17dis_vec * self.alfac**3, self.ClipRate)
        self.k104dis_vec = np.minimum(self.k71dis_vec / (self.btfac**3), self.ClipRate)
        self.k511dis_vec = np.minimum(self.k17dis_vec * self.alfac**4, self.ClipRate)
        self.k115dis_vec = np.minimum(self.k71dis_vec / (self.btfac**4), self.ClipRate)
        
        # Open state transitions
        self.k612dis_vec = np.minimum(konop, self.ClipRate)
        self.k126dis_vec = np.minimum(koffop, self.ClipRate)
        
        # Inactivated state transitions
        self.k78dis_vec = np.minimum(4 * amt * self.alfac, self.ClipRate)
        self.k89dis_vec = np.minimum(3 * amt * self.alfac, self.ClipRate)
        self.k910dis_vec = np.minimum(2 * amt * self.alfac, self.ClipRate)
        self.k1011dis_vec = np.minimum(amt * self.alfac, self.ClipRate)
        self.k1112dis_vec = np.minimum(gmt, self.ClipRate)
        
        # Backward transitions in inactivated states
        self.k1110dis_vec = np.minimum(4 * bmt * (1/self.btfac), self.ClipRate)
        self.k109dis_vec = np.minimum(3 * bmt * (1/self.btfac), self.ClipRate)
        self.k98dis_vec = np.minimum(2 * bmt * (1/self.btfac), self.ClipRate)
        self.k87dis_vec = np.minimum(bmt * (1/self.btfac), self.ClipRate)
        
        # Vectorized k1211dis calculation with safety check
        k115_safe = np.where(self.k115dis_vec > 0, self.k115dis_vec, 1.0)
        self.k1211dis_vec = np.minimum(
            (self.k65dis_vec * self.k511dis_vec * self.k126dis_vec) / 
            (self.k612dis_vec * k115_safe), 
            self.ClipRate
        )
        

        
        # Update scalar rates for current voltage
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

        # Find the closest voltage index
        vidx = np.argmin(np.abs(self.vt - self.vm))
        
        # Update all scalar rates from vectorized arrays
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
                    # Set a default value if the array doesn't exist or is invalid
                    setattr(self, name, 0.0)
            else:
                # Set a default value if the vectorized version doesn't exist
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

        # Set temperature in Kelvin to fixed reference value (22°C)
        self.Tkel = 273.15 + 22.0
        
        # No temperature scaling for permeability
        scaled_PNasc = self.PNasc
        
        # Vectorized voltage processing
        v_volts = self.vt * 1e-3  # Convert all voltages from mV to V
        
        # Create masks for near-zero voltages
        near_zero = np.abs(v_volts) < 1e-6
        not_zero = ~near_zero
        
        # Sodium channel currents only - vectorized
        self.iscft = np.zeros_like(v_volts)
        
        # Near zero voltages - using L'Hôpital's rule
        if np.any(near_zero):
            du2_zero = self.F * self.F / (self.Rgc * self.Tkel)
            self.iscft[near_zero] = scaled_PNasc * du2_zero * (self.Nai - self.Nao)
        
        # Non-zero voltages - using standard GHK equation
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
        
        # Ensure rate vectors are created first
        if not hasattr(self, 'k12dis_vec'):
            self.create_rate_waves()
            
        # Calculate voltage-dependent rates
        self.stRatesVolt()
        
        # Update scalar rates for this voltage
        self._update_scalar_rates()
        
        # Safe division function to prevent NaN values
        def safe_div(a, b, default=0.0):
            """Safely divides a by b, returning default if b is zero."""

            if np.isscalar(b):
                if abs(b) > 1e-10:
                    return a / b
                else:
                    return default
            else:
                # For arrays, use masking instead of np.where to avoid evaluation errors
                result = np.full_like(a, default, dtype=float)
                mask = np.abs(b) > 1e-10
                if np.any(mask):
                    result[mask] = a[mask] / b[mask]
                return result
        
        # Vectorized rate constant ratios
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
        
        # Calculate denominators for normalization
        dusuma = 1 + du1 + du1*du2 + du1*du2*du3 + du1*du2*du3*du4 + du1*du2*du3*du4*du5
        dusumb = du7 + du7*du8 + du7*du8*du9 + du7*du8*du9*du10 + du7*du8*du9*du10*du11 + du7*du8*du9*du10*du11*du12
        dusum = dusuma + dusumb
        
        # Create state vector using 0-based indexing (12 states, indices 0-11)
        pop = np.zeros(12)
        
        # Vectorized state occupancy calculation
        if dusum > 1e-10:
            # Pre-compute common products for states 1-6 (indices 0-5)
            du_products = np.array([
                1, du1, du1*du2, du1*du2*du3, du1*du2*du3*du4, 
                du1*du2*du3*du4*du5
            ])
            
            # Pre-compute common products for states 7-12 (indices 6-11)
            du7_products = np.array([
                du7, du7*du8, du7*du8*du9, du7*du8*du9*du10,
                du7*du8*du9*du10*du11, du7*du8*du9*du10*du11*du12
            ])
            
            # Assign values using 0-based indexing
            pop[:6] = du_products / dusum  # States 1-6 (indices 0-5)
            pop[6:12] = du7_products / dusum  # States 7-12 (indices 6-11)
        else:
            # Fallback distribution using 0-based indexing
            pop[0] = 0.98  # State 1 (index 0)
            pop[1] = 0.02  # State 2 (index 1)
        
        # Ensure no NaN values
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

        # Fetch all transition rates at once
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

        # Off-diagonal entries: Q[i, j] = rate from state j → state i
        # State 1 (index 0)
        Q[0, 1] = k21dis
        Q[0, 6] = k71dis

        # State 2 (index 1)
        Q[1, 0] = k12dis
        Q[1, 2] = k32dis
        Q[1, 7] = k82dis

        # State 3 (index 2)
        Q[2, 1] = k23dis
        Q[2, 3] = k43dis
        Q[2, 8] = k93dis

        # State 4 (index 3)
        Q[3, 2] = k34dis
        Q[3, 4] = k54dis
        Q[3, 9] = k104dis

        # State 5 (index 4)
        Q[4, 3] = k45dis
        Q[4, 5] = k65dis
        Q[4, 10] = k115dis

        # State 6 (index 5)
        Q[5, 4] = k56dis
        Q[5, 11] = k126dis

        # State 7 (index 6)
        Q[6, 0] = k17dis
        Q[6, 7] = k87dis

        # State 8 (index 7)
        Q[7, 6] = k78dis
        Q[7, 8] = k98dis
        Q[7, 1] = k28dis

        # State 9 (index 8)
        Q[8, 7] = k89dis
        Q[8, 9] = k109dis
        Q[8, 2] = k39dis

        # State 10 (index 9)
        Q[9, 8] = k910dis
        Q[9, 10] = k1110dis
        Q[9, 3] = k410dis

        # State 11 (index 10)
        Q[10, 9] = k1011dis
        Q[10, 11] = k1211dis
        Q[10, 4] = k511dis

        # State 12 (index 11)
        Q[11, 10] = k1112dis
        Q[11, 5] = k612dis

        # Diagonal entries: negative sum of outgoing rates from each state
        Q[0, 0]   = -(k12dis + k17dis)                                     # State 1 (C1) -> C2, I1
        Q[1, 1]   = -(k21dis + k23dis + k28dis)                             # State 2 (C2) -> C1, C3, I2
        Q[2, 2]   = -(k32dis + k34dis + k39dis)                             # State 3 (C3) -> C2, C4, I3
        Q[3, 3]   = -(k43dis + k45dis + k410dis)                            # State 4 (C4) -> C3, C5, I4
        Q[4, 4]   = -(k54dis + k56dis + k511dis)                            # State 5 (C5) -> C4, O, I5
        Q[5, 5]   = -(k65dis + k612dis)                                     # State 6 (O)  -> C5, IO
        Q[6, 6]   = -(k71dis + k78dis)                                     # State 7 (I1) -> C1, I2
        Q[7, 7]   = -(k82dis + k87dis + k89dis)                             # State 8 (I2) -> C2, I1, I3
        Q[8, 8]   = -(k93dis + k98dis + k910dis)                            # State 9 (I3) -> C3, I2, I4
        Q[9, 9]   = -(k104dis + k109dis + k1011dis)                         # State 10 (I4)-> C4, I3, I5
        Q[10, 10] = -(k115dis + k1110dis + k1112dis)                        # State 11 (I5)-> C5, I4, IO
        Q[11, 11] = -(k126dis + k1211dis)                                   # State 12 (IO)-> O, I5

        # Full‐matrix multiplication (12×12)
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

        # Get sweep parameters with bounds checking
        if SwpNo >= self.SwpSeq.shape[1] or SwpNo < 0:
            raise ValueError(f"Invalid sweep number {SwpNo}")
        
        SwpSeq = self.SwpSeq
        NumEpchs = int(SwpSeq[0, SwpNo])
        
        if NumEpchs <= 0 or 2*NumEpchs + 1 >= SwpSeq.shape[0]:
            raise ValueError("Invalid number of epochs in protocol")
        
        # Calculate total simulation points
        total_points = int(SwpSeq[2*NumEpchs + 1, SwpNo]) + 1
        sampint = 0.005  # 5 μs sampling interval
        
        # Pre-allocate all result arrays
        self.SimSwp = np.zeros(total_points)   # Current trace
        self.SimOp = np.zeros(total_points)    # Open probability
        self.SimIn = np.zeros(total_points)    # Inactivated probability
        self.SimAv = np.zeros(total_points)    # Available probability
        self.SimCom = np.zeros(total_points)   # Command voltage
        
        # Initialize state vector using 0-based indexing
        self.pop = np.zeros(13)
        self.pop[0] = 1.0  # Start in state 1 (index 0 in 0-based indexing)
        
        # Pre-extract all epoch parameters for vectorized access
        epoch_voltages = np.zeros(NumEpchs + 1)
        epoch_end_times = np.zeros(NumEpchs + 1)
        
        # Initial voltage
        epoch_voltages[0] = SwpSeq[2, SwpNo]
        epoch_end_times[0] = 0.0
        
        for e in range(1, NumEpchs + 1):
            epoch_voltages[e] = SwpSeq[2 * e, SwpNo]
            epoch_end_times[e] = int(SwpSeq[2 * e + 1, SwpNo]) * sampint
        
        # Set initial voltage
        self.vm = epoch_voltages[0]
        
        # Calculate equilibrium at holding potential
        self.CurrVolt()
        self.pop = self.EquilOccup(self.vm)
        
        # Store initial values
        self._store_results(0, 0)
        
        # Current time and storage index
        current_time = 0.0
        store_idx = 1
        
        # Process each epoch
        for epoch in range(1, NumEpchs + 1):
            # Set voltage for this epoch
            self.vm = epoch_voltages[epoch]
            epoch_end_time = epoch_end_times[epoch]
            
            # Update voltage-dependent parameters
            self._update_scalar_rates()
            self.CurrVolt()
            
            # Create time evaluation points
            num_points = max(2, int((epoch_end_time - current_time) / sampint) + 1)
            t_eval = np.linspace(current_time, epoch_end_time, num_points)
            
            if len(t_eval) <= 1:
                current_time = epoch_end_time
                continue
            
            # Use pre-allocated array for initial conditions
            # NowDerivs expects 12 states (indexed 0-11 in Python)
            self._reusable_y0[:] = self.pop[:12]
            
            # Solve ODE system
            sol = solve_ivp(
                self.NowDerivs,
                [current_time, epoch_end_time],
                self._reusable_y0,
                method='LSODA',
                t_eval=t_eval,
                rtol=1e-6,
                atol=1e-8
            )
            
            # Vectorized result storage
            if sol.success and len(sol.t) > 0:
                # Calculate batch indices
                batch_size = len(sol.t)
                end_idx = min(store_idx + batch_size, total_points)
                batch_indices = np.arange(store_idx, end_idx)
                actual_batch = len(batch_indices)
                
                if actual_batch > 0:
                    # Extract batch of states
                    batch_states = sol.y[:, :actual_batch].T
                    
                    # Store results in vectorized fashion
                    self._store_results_vectorized(
                        batch_indices,
                        batch_states,
                        self.vm
                    )
                    
                    # Update state for next epoch using 0-based indexing
                    self.pop[:12] = sol.y[:, -1]
                    store_idx = end_idx
            
            current_time = epoch_end_time
        
        # Create time vector
        self.time = np.arange(0, total_points * sampint, sampint)[:total_points]
        
        # Return peak current (most negative for inward current)
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
        
        # Calculate current from open state and GHK current
        open_prob = self.pop[5]  # Open state is at index 5 (0-based indexing)
        current = open_prob * self.iscft[vidx] * self.numchan * self.current_scaling
        
        # Store results
        self.SimSwp[idx] = current
        self.SimOp[idx] = self.pop[5]  # Open state (index 5 in 0-based indexing)
        self.SimIn[idx] = np.sum(self.pop[6:])  # All inactivated states (states 7-13 in 1-based, indices 6-12 in 0-based)
        self.SimAv[idx] = np.sum(self.pop[:6])  # Available states (states 1-6 in 1-based, indices 0-5 in 0-based)
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
        
        # Find voltage index
        vidx = np.searchsorted(self.vt, voltage)
        vidx = np.clip(vidx, 0, len(self.vt) - 1)
        
        # Extract current factor for this voltage
        current_factor = self.iscft[vidx]
        
        # Vectorized calculations using 0-based indexing throughout
        # The batch_states array already has 12 states (indices 0-11) as returned by ODE solver
        
        # Extract open probabilities (state 6 in 1-based indexing, index 5 in 0-based indexing)
        open_probs = batch_states[:, 5]
        
        # Calculate currents
        currents = open_probs * current_factor * self.numchan * self.current_scaling
        
        # Calculate aggregate probabilities
        inactivated = np.sum(batch_states[:, 6:], axis=1)  # All inactivated states (indices 6-11)
        available = np.sum(batch_states[:, :6], axis=1)    # Available states (indices 0-5)
        
        # Store all results at once
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
        
        # Use default target voltages if none provided
        if target_voltages is None:
            target_voltages = [30, 0, -20, -30, -40, -50, -60]
        
        # Convert to numpy array for vectorized operations
        target_voltages = np.array(target_voltages)
        self.NumSwps = len(target_voltages)
        
        # Initialize protocol array
        self.SwpSeq = np.zeros((8, self.NumSwps))
        
        # Calculate time points in samples
        holding_samples = int(holding_duration / 0.005)
        test_samples = int(test_duration / 0.005)
        tail_samples = int(tail_duration / 0.005)
        total_samples = holding_samples + test_samples + tail_samples
        
        # Vectorized protocol setup
        self.SwpSeq[0, :] = 3  # 3 epochs per sweep
        
        # Epoch 1: Holding period
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        
        # Epoch 2: Step to target voltages (vectorized assignment)
        self.SwpSeq[4, :] = target_voltages
        self.SwpSeq[5, :] = holding_samples + test_samples
        
        # Epoch 3: Tail period
        self.SwpSeq[6, :] = holding_potential
        self.SwpSeq[7, :] = total_samples
        
        # Validation
        assert self.NumSwps == len(target_voltages), "Voltage count mismatch"
        assert np.allclose(self.SwpSeq[4,:], target_voltages), "Voltage assignment error"
        
        # Store protocol
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        
        # Recalculate voltage-dependent currents
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
        
class AnticonvulsantMarkovModel:
    """
    Extended Markov model with anticonvulsant drug binding based on Kuo (1998).
    
    This extends the original 12-state Kuo-Bean model to 24 states by adding
    drug-bound versions of each state. Key findings from Kuo (1998):
    - Common external binding site for phenytoin, carbamazepine, lamotrigine
    - Much higher affinity for inactivated states (KI ~9-25 μM) vs resting states (KR ~mM range)  
    - Single occupancy: one drug molecule per channel
    - External application only
    - Drug-specific recovery kinetics: CBZ τ~180ms, LTG τ~320ms, DPH τ~180ms
    
    States 0-11: Drug-free states (original model)
    States 12-23: Drug-bound states
    """
    
    def __init__(self, drug_concentration=0.0, drug_type='mixed'):
        """
        Initialize the anticonvulsant-extended Markov model.

        Args:
            drug_concentration (float): Anticonvulsant concentration in μM
            drug_type (str): 'CBZ', 'LTG', 'DPH', or 'mixed' for average parameters
        """
        self.NumSwps = 0
        self.num_states = 25  # Extended to 24 states
        self.drug_concentration = drug_concentration  # μM
        self.drug_type = drug_type.upper()

        # Initialize membrane voltage
        self.vm = -80

        # Initialize parameters and waves
        self.init_parameters()
        self.init_waves()

        # Calculate rates and currents
        self._update_drug_rates()  # Apply drug concentration to base rates
        self.CurrVolt()
        self.create_default_protocol()
        self.pop = self.EquilOccup(self.vm)  # Set initial state based on concentration

    def set_drug_type(self, drug_type):
        """
        Set the type of anticonvulsant drug and update model parameters.

        Args:
            drug_type (str): 'CBZ', 'LTG', 'DPH', or 'mixed'.
        """
        self.drug_type = drug_type.upper()
        self.init_parameters()  # Re-initialize drug-specific base affinities and rates
        self._update_drug_rates() # Apply current concentration to new base rates
        self.pop = self.EquilOccup(self.vm) # Re-calculate equilibrium occupancy

    def set_drug_concentration(self, drug_concentration):
        """
        Set the anticonvulsant drug concentration and update model parameters.

        Args:
            drug_concentration (float): Concentration in μM.
        """
        self.drug_concentration = drug_concentration
        self._update_drug_rates() # Apply new concentration
        self.pop = self.EquilOccup(self.vm) # Re-calculate equilibrium occupancy
    
    def init_parameters(self):
        """Initialize biophysical parameters with Kuo 1998-consistent kinetics."""
        
        # Original Kuo-Bean parameters (unchanged)
        self.alcoeff = 150      
        self.alslp = 20           
        self.btcoeff = 3      
        self.btslp = 20       
        
        # Inactivation parameters (unchanged)
        self.ConCoeff = 0.005
        self.CoffCoeff = 0.5
        self.ConSlp = 1e8       
        self.CoffSlp = 1e8      
        
        # Transition rates between states (unchanged)
        self.gmcoeff = 150      
        self.gmslp = 1e12       
        self.dlcoeff = 40       
        self.dlslp = 1e12       
        self.epcoeff = 1.75   
        self.epslp = 1e12       
        self.ztcoeff = 0.03    
        self.ztslp = 25       
        
        # Open state transitions (unchanged)
        self.OpOnCoeff = 0.75   
        self.OpOffCoeff = 0.005 
        self.ConHiCoeff = 0.75   
        self.CoffHiCoeff = 0.005
        self.OpOnSlp = 1e8      
        self.OpOffSlp = 1e8  
        
        # Calculate alfac and btfac (unchanged)
        self.alfac = np.sqrt(np.sqrt(self.ConHiCoeff / self.ConCoeff))
        self.btfac = np.sqrt(np.sqrt(self.CoffCoeff / self.CoffHiCoeff))
        
        # ========== CORRECTED: Kuo 1998-consistent anticonvulsant parameters ==========
        
        # Drug-specific binding affinities from Kuo 1998
        self.drug_params = {
            'CBZ': {
                'KI_inactivated': 25.0,  # μM - from Kuo 1998
                'recovery_tau': 189.0,   # ms - average of 180-197 ms
                'k_off': 1.0 / 189.0     # /ms
            },
            'LTG': {
                'KI_inactivated': 9.0,   # μM - from Kuo 1998  
                'recovery_tau': 321.0,   # ms - average of 317-325 ms
                'k_off': 1.0 / 321.0     # /ms
            },
            'DPH': {
                'KI_inactivated': 9.0,   # μM - from Kuo 1998
                'recovery_tau': 189.0,   # ms - similar to CBZ based on Kuo 1998
                'k_off': 1.0 / 189.0     # /ms
            },
            'MIXED': {
                'KI_inactivated': 15.0,  # μM - average of the three drugs
                'recovery_tau': 233.0,   # ms - average recovery time
                'k_off': 1.0 / 233.0     # /ms
            }
        }
        
        # Select drug-specific parameters
        if self.drug_type in self.drug_params:
            params = self.drug_params[self.drug_type]
        else:
            print(f"Warning: Unknown drug type '{self.drug_type}', using mixed parameters")
            params = self.drug_params['MIXED']
        
        # Set drug-specific parameters
        self.KI_inactivated = params['KI_inactivated']
        self.recovery_tau = params['recovery_tau']
        self.k_off = params['k_off']
        
        # Resting state affinity - 100x weaker than inactivated (Kuo 1998: "mM range")
        self.KR_resting = self.KI_inactivated * 100.0
        
        # Calculate binding rates: Kd = k_off / k_on, so k_on = k_off / Kd
        self.k_on_inactivated_base = self.k_off / self.KI_inactivated
        self.k_on_resting_base = self.k_off / self.KR_resting
        
        # Other model parameters (unchanged)
        self.numchan = 1
        self.cm = 30
        self.F = 96480
        self.Rgc = 8314
        self.Tkel = 298
        self.Nao, self.Nai = 155, 15
        self.ClipRate = 6000
        self.current_scaling = 0.0117
        self.PNasc = 1e-5 
        
        # Pre-allocate for 24 states
        self._reusable_y0 = np.zeros(24)
        
        # Update drug-dependent rates
        self._update_drug_rates()
    
    def _update_drug_rates(self):
        """Update drug binding rates based on current concentration."""
        # Calculate concentration-dependent binding rates
        self.k_on_resting = self.k_on_resting_base * self.drug_concentration
        self.k_on_inactivated = self.k_on_inactivated_base * self.drug_concentration
        
        # k_off is the same for both states (drug-specific)
        self.k_off_resting = self.k_off
        self.k_off_inactivated = self.k_off
    
    def init_waves(self):
        """Initialize data arrays for 24-state simulation."""
        self.vt = np.arange(-200, 201)
        
        # Create state array with 24 elements
        self.pop = np.zeros(24)  
        self.dstdt = np.zeros(24)
        
        # Pre-allocate reusable array for ODE solver with 24 states
        self._reusable_y0 = np.zeros(24)
        
        # Initialize current arrays
        self.iscft = np.zeros_like(self.vt)
        
        self.create_rate_waves()
        self.stRatesVolt()

    def create_rate_waves(self):
        """Create arrays for voltage-dependent rates."""
        
        # Original transition rates
        rate_names = ['k12dis', 'k23dis', 'k34dis', 'k45dis', 'k56dis',
                     'k65dis', 'k54dis', 'k43dis', 'k32dis', 'k21dis',
                     'k17dis', 'k71dis', 'k28dis', 'k82dis', 'k39dis',
                     'k93dis', 'k410dis', 'k104dis', 'k511dis', 'k115dis',
                     'k612dis', 'k126dis', 'k78dis', 'k89dis', 'k910dis',
                     'k1011dis', 'k1112dis', 'k1211dis', 'k1110dis', 'k109dis',
                     'k98dis', 'k87dis']
        
        # Create vectorized rate arrays for original states
        for name in rate_names:
            setattr(self, name + '_vec', np.zeros_like(self.vt, dtype=float))
    
    def stRatesVolt(self):
        """Calculate voltage-dependent rates for original states."""
        
        if not hasattr(self, 'ClipRate') or self.ClipRate is None:
            self.ClipRate = 1000
        
        if not hasattr(self, 'k12dis_vec'):
            self.create_rate_waves()
        
        vt = self.vt
        
        # Original rate calculations (same as before)
        amt = self.alcoeff * np.exp(vt / self.alslp)
        bmt = self.btcoeff * np.exp(-vt / self.btslp)
        gmt = self.gmcoeff * np.exp(vt / self.gmslp)
        dmt = self.dlcoeff * np.exp(-vt / self.dlslp)
        emt = self.epcoeff * np.exp(vt / self.epslp)
        zmt = self.ztcoeff * np.exp(-vt / self.ztslp)
        
        konlo = self.ConCoeff * np.exp(vt / self.ConSlp)
        kofflo = self.CoffCoeff * np.exp(-vt / self.CoffSlp)
        konop = self.OpOnCoeff * np.exp(vt / self.OpOnSlp)
        koffop = self.OpOffCoeff * np.exp(-vt / self.OpOffSlp)
        
        # Forward rates (activation)
        self.k12dis_vec = np.minimum(4 * amt, self.ClipRate)
        self.k23dis_vec = np.minimum(3 * amt, self.ClipRate)
        self.k34dis_vec = np.minimum(2 * amt, self.ClipRate)
        self.k45dis_vec = np.minimum(amt, self.ClipRate)
        self.k56dis_vec = np.minimum(gmt, self.ClipRate)
        
        # Backward rates (deactivation)
        self.k65dis_vec = np.minimum(dmt, self.ClipRate)
        self.k54dis_vec = np.minimum(4 * bmt, self.ClipRate)
        self.k43dis_vec = np.minimum(3 * bmt, self.ClipRate)
        self.k32dis_vec = np.minimum(2 * bmt, self.ClipRate)
        self.k21dis_vec = np.minimum(bmt, self.ClipRate)
        
        # Inactivation transitions
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
        
        # Open state transitions
        self.k612dis_vec = np.minimum(konop, self.ClipRate)
        self.k126dis_vec = np.minimum(koffop, self.ClipRate)
        
        # Inactivated state transitions
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
        """Update scalar rates for current voltage."""
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
        """Calculate I-V relationship (same as original)."""
        self.Tkel = 273.15 + 22.0
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
        """Calculate equilibrium occupancies for 24-state model."""
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
        
        # Calculate equilibrium for drug-free states (same as original)
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
        
        # Drug binding equilibrium factors
        # These represent the equilibrium constants for drug binding
        drug_factor_closed = self.k_on_resting / self.k_off_resting
        drug_factor_inactivated = self.k_on_inactivated / self.k_off_inactivated
        
        # Calculate unnormalized probabilities
        # Drug-free closed states (0-5)
        dusuma_free = 1 + du1 + du1*du2 + du1*du2*du3 + du1*du2*du3*du4 + du1*du2*du3*du4*du5
        # Drug-free inactivated states (6-11) 
        dusumb_free = du7 + du7*du8 + du7*du8*du9 + du7*du8*du9*du10 + du7*du8*du9*du10*du11 + du7*du8*du9*du10*du11*du12
        
        # Drug-bound closed states (12-17)
        dusuma_drug = drug_factor_closed * dusuma_free
        # Drug-bound inactivated states (18-23)
        dusumb_drug = drug_factor_inactivated * dusumb_free
            
        dusum_total = dusuma_free + dusumb_free + dusuma_drug + dusumb_drug
        
        pop = np.zeros(24)
        
        if dusum_total > 1e-10:
            # Drug-free closed states (0-5)
            closed_products = np.array([1, du1, du1*du2, du1*du2*du3, du1*du2*du3*du4, du1*du2*du3*du4*du5])
            pop[0:6] = closed_products / dusum_total
            
            # Drug-free inactivated states (6-11)
            inact_products = np.array([du7, du7*du8, du7*du8*du9, du7*du8*du9*du10, du7*du8*du9*du10*du11, du7*du8*du9*du10*du11*du12])
            pop[6:12] = inact_products / dusum_total
            
            # Drug-bound closed states (12-17)
            pop[12:18] = drug_factor_closed * closed_products / dusum_total
            
            # Drug-bound inactivated states (18-23)
            pop[18:24] = drug_factor_inactivated * inact_products / dusum_total
        else:
            pop[0] = 0.98  # Start in closed state 1
            pop[1] = 0.02
        
        pop = np.nan_to_num(pop, nan=0.0)
        
        return pop

    def NowDerivs(self, t, y):
        """Calculate derivatives for the 24-state anticonvulsant model."""
        vidx = np.searchsorted(self.vt, self.vm)
        vidx = np.clip(vidx, 0, len(self.vt) - 1)

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return np.zeros_like(y)

        # Fetch all voltage-dependent transition rates
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
        
        # Drug binding/unbinding rates 
        k_on_closed = self.k_on_resting
        k_off_closed = self.k_off_resting
        k_on_inact = self.k_on_inactivated
        k_off_inact = self.k_off_inactivated

        st = y.copy()
        Q = np.zeros((24, 24))

        # --- Off-diagonal entries: Q[to, from] = rate from state 'from' → state 'to' ---

        # Region 1: Drug-Free States (0-11) - Intrinsic Gating
        # State 0 (C1)
        Q[0, 1] = k21dis; Q[0, 6] = k71dis
        # State 1 (C2)
        Q[1, 0] = k12dis; Q[1, 2] = k32dis; Q[1, 7] = k82dis
        # State 2 (C3)
        Q[2, 1] = k23dis; Q[2, 3] = k43dis; Q[2, 8] = k93dis
        # State 3 (C4)
        Q[3, 2] = k34dis; Q[3, 4] = k54dis; Q[3, 9] = k104dis
        # State 4 (C5)
        Q[4, 3] = k45dis; Q[4, 5] = k65dis; Q[4, 10] = k115dis
        # State 5 (O)
        Q[5, 4] = k56dis; Q[5, 11] = k126dis
        # State 6 (I1)
        Q[6, 0] = k17dis; Q[6, 7] = k87dis
        # State 7 (I2)
        Q[7, 1] = k28dis; Q[7, 6] = k78dis; Q[7, 8] = k98dis
        # State 8 (I3)
        Q[8, 2] = k39dis; Q[8, 7] = k89dis; Q[8, 9] = k109dis
        # State 9 (I4)
        Q[9, 3] = k410dis; Q[9, 8] = k910dis; Q[9, 10] = k1110dis
        # State 10 (I5)
        Q[10, 4] = k511dis; Q[10, 9] = k1011dis; Q[10, 11] = k1211dis
        # State 11 (IO)
        Q[11, 5] = k612dis; Q[11, 10] = k1112dis

        # Region 2: Drug-Bound States (12-23) - Intrinsic Gating
        # State 12 (DC1) 
        Q[12, 13] = k21dis; Q[12, 18] = k71dis
        # State 13 (DC2) 
        Q[13, 12] = k12dis; Q[13, 14] = k32dis; Q[13, 19] = k82dis
        # State 14 (DC3) 
        Q[14, 13] = k23dis; Q[14, 15] = k43dis; Q[14, 20] = k93dis
        # State 15 (DC4) 
        Q[15, 14] = k34dis; Q[15, 16] = k54dis; Q[15, 21] = k104dis
        # State 16 (DC5) 
        Q[16, 15] = k45dis; Q[16, 17] = k65dis; Q[16, 22] = k115dis
        # State 17 (DO) 
        Q[17, 16] = k56dis; Q[17, 23] = k126dis
        # State 18 (DI1) 
        Q[18, 12] = k17dis; Q[18, 19] = k87dis
        # State 19 (DI2) 
        Q[19, 13] = k28dis; Q[19, 18] = k78dis; Q[19, 20] = k98dis
        # State 20 (DI3) 
        Q[20, 14] = k39dis; Q[20, 19] = k89dis; Q[20, 21] = k109dis
        # State 21 (DI4) 
        Q[21, 15] = k410dis; Q[21, 20] = k910dis; Q[21, 22] = k1110dis
        # State 22 (DI5) 
        Q[22, 16] = k511dis; Q[22, 21] = k1011dis; Q[22, 23] = k1211dis
        # State 23 (DIO) 
        Q[23, 17] = k612dis; Q[23, 22] = k1112dis

        # Region 3: Drug Binding/Unbinding Transitions
        # Drug-free (0-5, C1-O) <-> Drug-bound (12-17, DC1-DO) 
        Q[12, 0] = k_on_closed; Q[0, 12] = k_off_closed # C1 <-> DC1
        Q[13, 1] = k_on_closed; Q[1, 13] = k_off_closed # C2 <-> DC2
        Q[14, 2] = k_on_closed; Q[2, 14] = k_off_closed # C3 <-> DC3
        Q[15, 3] = k_on_closed; Q[3, 15] = k_off_closed # C4 <-> DC4
        Q[16, 4] = k_on_closed; Q[4, 16] = k_off_closed # C5 <-> DC5
        Q[17, 5] = k_on_closed; Q[5, 17] = k_off_closed # O  <-> DO

        # Drug-free (6-11, I1-IO) <-> Drug-bound (18-23, DI1-DIO) 
        Q[18, 6] = k_on_inact; Q[6, 18] = k_off_inact   # I1 <-> DI1
        Q[19, 7] = k_on_inact; Q[7, 19] = k_off_inact   # I2 <-> DI2
        Q[20, 8] = k_on_inact; Q[8, 20] = k_off_inact   # I3 <-> DI3
        Q[21, 9] = k_on_inact; Q[9, 21] = k_off_inact   # I4 <-> DI4
        Q[22, 10] = k_on_inact; Q[10, 22] = k_off_inact # I5 <-> DI5
        Q[23, 11] = k_on_inact; Q[11, 23] = k_off_inact # IO <-> DIO

        # --- Diagonal entries: Q[i, i] = - (sum of all rates leaving state i) ---
        # Drug-Free States (0-11)
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

        # Drug-Bound States (12-23)
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

        # Full matrix multiplication
        dstdt = np.zeros_like(y)
        for i in range(24):
            for j in range(24):
                dstdt[i] += Q[i, j] * st[j]

        if np.any(np.isnan(dstdt)) or np.any(np.isinf(dstdt)):
            return np.zeros_like(st)

        return dstdt

    def Sweep(self, SwpNo):
        """Run voltage-clamp sweep for 24-state model."""
        if SwpNo >= self.SwpSeq.shape[1] or SwpNo < 0:
            raise ValueError(f"Invalid sweep number {SwpNo}")
        
        SwpSeq = self.SwpSeq
        NumEpchs = int(SwpSeq[0, SwpNo])
        
        if NumEpchs <= 0 or 2*NumEpchs + 1 >= SwpSeq.shape[0]:
            raise ValueError("Invalid number of epochs in protocol")
        
        total_points = int(SwpSeq[2*NumEpchs + 1, SwpNo]) + 1
        sampint = 0.005
        
        # Pre-allocate result arrays
        self.SimSwp = np.zeros(total_points)
        self.SimOp = np.zeros(total_points)
        self.SimIn = np.zeros(total_points)
        self.SimAv = np.zeros(total_points)
        self.SimCom = np.zeros(total_points)
        self.SimDrugBound = np.zeros(total_points)  # Track drug-bound fraction
        
        # Extract epoch parameters
        epoch_voltages = np.zeros(NumEpchs + 1)
        epoch_end_times = np.zeros(NumEpchs + 1)
        
        epoch_voltages[0] = SwpSeq[2, SwpNo]
        epoch_end_times[0] = 0.0
        
        for e in range(1, NumEpchs + 1):
            epoch_voltages[e] = SwpSeq[2 * e, SwpNo]
            epoch_end_times[e] = int(SwpSeq[2 * e + 1, SwpNo]) * sampint
        
        # Set initial voltage and calculate equilibrium
        self.vm = epoch_voltages[0]
        self.CurrVolt()
        self.pop = self.EquilOccup(self.vm)
                
        self._store_results_24(0, 0)
        
        current_time = 0.0
        store_idx = 1
        
        # Process each epoch
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
            
            # Use 24 states for initial conditions
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
        """Store results for single time point (24-state version)."""
        vidx = np.searchsorted(self.vt, self.vm)
        vidx = np.clip(vidx, 0, len(self.vt) - 1)
        
        # Current comes from drug-free open state only
        # Drug-bound open state is non-conducting (blocked)
        open_prob_free = self.pop[5]    # Drug-free open state
        open_prob_drug = self.pop[17]   # Drug-bound open state (blocked)
        
        # Only drug-free open state contributes to current
        conducting_open_prob = open_prob_free
        
        current = conducting_open_prob * self.iscft[vidx] * self.numchan * self.current_scaling
        
        self.SimSwp[idx] = current
        self.SimOp[idx] = open_prob_free + open_prob_drug  # Total open probability
        self.SimIn[idx] = np.sum(self.pop[6:12]) + np.sum(self.pop[18:24])  # All inactivated
        self.SimAv[idx] = np.sum(self.pop[:6]) + np.sum(self.pop[12:18])    # All available
        self.SimCom[idx] = self.vm
        self.SimDrugBound[idx] = np.sum(self.pop[12:24])  # All drug-bound states
    
    def _store_results_vectorized_24(self, indices, batch_states, voltage):
        """Store batch results (24-state version)."""
        if len(indices) == 0:
            return
        
        vidx = np.searchsorted(self.vt, voltage)
        vidx = np.clip(vidx, 0, len(self.vt) - 1)
        current_factor = self.iscft[vidx]
        
        # Only drug-free open state contributes to current (drug-bound open is blocked)
        conducting_open_probs = batch_states[:, 5]  # Drug-free open state only
        total_open_probs = batch_states[:, 5] + batch_states[:, 17]  # Total open for tracking
        
        # Current from conducting (drug-free) open states only
        currents = conducting_open_probs * current_factor * self.numchan * self.current_scaling
        
        # Aggregate probabilities
        inactivated = np.sum(batch_states[:, 6:12], axis=1) + np.sum(batch_states[:, 18:24], axis=1)
        available = np.sum(batch_states[:, :6], axis=1) + np.sum(batch_states[:, 12:18], axis=1)
        drug_bound = np.sum(batch_states[:, 12:24], axis=1)
        
        self.SimSwp[indices] = currents
        self.SimOp[indices] = total_open_probs  # Track total open probability
        self.SimIn[indices] = inactivated
        self.SimAv[indices] = available
        self.SimCom[indices] = voltage
        self.SimDrugBound[indices] = drug_bound
        
    def create_default_protocol(self, target_voltages=None, holding_potential=-80,
                              holding_duration=98, test_duration=200, tail_duration=2):
        """Create default protocol identical to 12-state model."""
        self.BsNm = "AnticonvulsantProtocol"
        
        if target_voltages is None:
            # Use SAME voltages as 12-state model for consistency
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
    
    def set_drug_concentration(self, concentration):
        """Set anticonvulsant concentration and update all dependent rates."""        
        self.drug_concentration = concentration
        self._update_drug_rates()
        
        # Recalculate equilibrium with new concentration
        if hasattr(self, 'vm'):
            self.pop = self.EquilOccup(self.vm)

    def set_drug_type(self, drug_type):
        """Change drug type and reinitialize parameters."""
        self.drug_type = drug_type.upper()
        old_concentration = self.drug_concentration
        self.init_parameters()
        self.set_drug_concentration(old_concentration)