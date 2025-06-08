import numpy as np
from scipy.integrate import solve_ivp

class CTBNMarkovModel:
    """
    Implements a Continuous-Time Bayesian Network (CTBN) based Markov model
    for simulating ion channel currents, specifically sodium channels.

    This model represents channel states and transitions using CTBN principles,
    allowing for dynamic calculation of state probabilities over time in response
    to changing membrane voltages. It includes parameters for activation,
    inactivation, and various transition rates between channel states.

    The model is designed for vectorized operations where possible to improve
    simulation speed, particularly in the `Sweep` method which simulates
    the channel's response to a voltage-clamp protocol. It calculates
    state occupancies and resulting ionic currents.

    Key Attributes:
        NumSwps (int): Number of sweeps in the current voltage protocol.
        vm (float): Current membrane potential in mV.
        state_probs_flat (np.ndarray): A 1D array representing the probabilities
            of the channel being in each of its 12 states (6 activation states
            for I=0 and 6 for I=1).
        SwpSeq (np.ndarray): The current voltage clamp protocol sequence.
        SimSwp (np.ndarray): Stores the simulated current for each time point
            of the last run sweep.
        SimOp (np.ndarray): Stores the probability of the channel being in an
            open state.
        SimIn (np.ndarray): Stores the probability of the channel being in an
            inactivated state.
        SimAv (np.ndarray): Stores the probability of the channel being in an
            available (not inactivated) state.
        SimCom (np.ndarray): Stores the command voltage for each time point.
        time (np.ndarray): Time vector for the simulation.
        vt (np.ndarray): A pre-defined array of voltage points for which rates
            and currents are pre-calculated or looked up.
        iscft (np.ndarray): Current scaling factor for each voltage in `vt`.
        # ... (other important parameters like alcoeff, btslp, etc.)

    The model structure involves:
    - Initialization of biophysical parameters (`init_parameters`).
    - Initialization of data structures for storing simulation results and
      pre-calculated values (`init_waves`).
    - Calculation of voltage-dependent transition rates (`stRatesVolt`).
    - Calculation of current-voltage relationships (`CurrVolt`).
    - Simulation of sweeps using an ODE solver (`Sweep`, `NowDerivs`).
    - Creation of default voltage protocols (`create_default_protocol`).
    """
    def __init__(self):
        """
        Initializes the CTBNMarkovModel instance.

        Sets up default values for sweep counts, membrane potential,
        and then calls helper methods to:
        - Initialize biophysical parameters (`init_parameters`).
        - Initialize data arrays and pre-calculate voltage-dependent values (`init_waves`).
        - Update initial transition rates (`update_rates`).
        - Calculate initial current-voltage relationships (`CurrVolt`).
        - Create a default voltage protocol (`create_default_protocol`).
        """
        self.NumSwps = 0
        
        # CTBN state variables - no change needed, these are scalars
        self.A = 0  # Activation index (0-5)
        self.I = 0  # Inactivation flag (0-1)
        
        # Initialize membrane voltage
        self.vm = -80  # Default holding potential
        
        # Initialize parameters and waves
        self.init_parameters()
        self.init_waves()
        
        # Calculate rates
        self.update_rates()
        self.CurrVolt()
        self.create_default_protocol()
    
    def init_parameters(self):
        """
        Initializes the biophysical parameters of the CTBN Markov model.

        These parameters define the voltage-dependent kinetics of channel
        activation, inactivation, and transitions between different states.
        Parameters include coefficients and slope factors for various rate
        equations (e.g., `alcoeff`, `alslp` for activation alpha rate).
        Also initializes derived parameters like `alfac`, `btfac`, and
        physical constants like `F` (Faraday's constant), `Rgc` (gas constant),
        `Tkel` (temperature in Kelvin), and ion concentrations.
        """

        # Activation parameters
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
        
        # Current scaling factor
        self.current_scaling = 0.0117
        
        self.PNasc = 1e-5

        self.vm = -80

    def init_waves(self):
        """
        Initializes data structures and pre-calculates voltage-dependent values.

        This method sets up:
        - `vt`: A numpy array of voltage points from -200mV to 200mV.
        - `iscft`: An array to store current scaling factors for each voltage in `vt`.
        - `state_probs_flat`: A 1D numpy array (size 12) to store the probability
          of the channel being in each of its 12 states (C1-C6 for I=0, O1-O6 for I=1).
          Initially, the channel is set to be in the C1 state (A=0, I=0).
        - Arrays for storing pre-calculated forward and backward transition rates
          for activation (`fwd_rates_I0`, `fwd_rates_I1`, `bwd_rates_I0`, `bwd_rates_I1`)
          and inactivation/recovery rates (`inact_on_rates`, `inact_off_rates`)
          across the `vt` voltage range.
        
        Calls `update_rates()` at the end to populate initial rate values.
        """

        self.vt = np.arange(-200, 201)
        
        # Initialize current array
        self.iscft = np.zeros_like(self.vt)
        
        # Initialize CTBN state probability as a flattened vector
        # Order: [P(0,0), P(1,0), ..., P(5,0), P(0,1), ..., P(5,1)]
        self.state_probs_flat = np.zeros(12)
        self.state_probs_flat[0] = 1.0  # Start in (A=0, I=0) = C1 state
        
        # Create vectorized rate storage
        # Store all rates as flat arrays indexed by voltage
        num_v = len(self.vt)
        
        # Forward activation rates for I=0 and I=1 (5 rates each)
        self.fwd_rates_I0 = np.zeros((num_v, 5))  # rates for a→a+1 when I=0
        self.fwd_rates_I1 = np.zeros((num_v, 5))  # rates for a→a+1 when I=1
        
        # Backward activation rates for I=0 and I=1 (5 rates each)
        self.bwd_rates_I0 = np.zeros((num_v, 5))  # rates for a→a-1 when I=0
        self.bwd_rates_I1 = np.zeros((num_v, 5))  # rates for a→a-1 when I=1
        
        # Inactivation rates for each activation level (6 levels)
        self.inact_on_rates = np.zeros((num_v, 6))   # 0→1 (inactivation)
        self.inact_off_rates = np.zeros((num_v, 6))  # 1→0 (recovery)
        
        # Calculate initial rates
        self.update_rates()

    def update_rates(self):
        """Recalculates and updates all voltage-dependent state transition rates by calling stRatesVolt."""
        self.stRatesVolt()

    def stRatesVolt(self):
        """
        Calculates and stores voltage-dependent state transition rates.

        This method computes the rates for all transitions in the CTBN model
        across the pre-defined voltage range `self.vt`. These rates include:
        - Activation rates (alpha_m, beta_m equivalents) for transitions
          between activation states (A0 to A5).
        - Inactivation rates (kon, koff equivalents) for transitions between
          non-inactivated (I=0) and inactivated (I=1) states.
        - Other transition rates specific to the CTBN model structure (e.g.,
          gamma, delta, epsilon, zeta).

        The calculated rates are stored in vectorized arrays like `self.fwd_rates_I0`,
        `self.bwd_rates_I0`, `self.inact_on_rates`, etc., for efficient lookup
        during simulations. Temperature scaling is currently fixed (not implemented).
        """

        # Use all voltages at once for vectorized computation
        vt = self.vt
        
        # No temperature scaling - fixed values
        activation_scale = 1.0
        deactivation_scale = 1.0
        
        # Basic rate parameters - vectorized across all voltages
        amt = self.alcoeff * np.exp(vt / self.alslp)
        bmt = self.btcoeff * np.exp(-vt / self.btslp)
        gmt = self.gmcoeff * np.exp(vt / self.gmslp)
        dmt = self.dlcoeff * np.exp(-vt / self.dlslp)
        
        # Inactivation rates - vectorized
        konlo = self.ConCoeff * np.exp(vt / self.ConSlp)
        kofflo = self.CoffCoeff * np.exp(-vt / self.CoffSlp)
        konop = self.OpOnCoeff * np.exp(vt / self.OpOnSlp)
        koffop = self.OpOffCoeff * np.exp(-vt / self.OpOffSlp)
        
        # Update Q_{A|I} rates - fully vectorized
        # For I = 0 (not inactivated)
        for a in range(5):  # Forward rates 0→1, 1→2, 2→3, 3→4, 4→5
            if a < 4:
                self.fwd_rates_I0[:, a] = np.minimum((4-a) * amt, self.ClipRate) # Corrected multiplier to match legacy
            else:  # a = 4, transition to open state
                self.fwd_rates_I0[:, a] = np.minimum(gmt, self.ClipRate)
        
        for a in range(5):  # Backward rates 1→0, 2→1, 3→2, 4→3, 5→4
            if a < 4:
                self.bwd_rates_I0[:, a] = np.minimum((a+1) * bmt, self.ClipRate)
            else:  # a = 4, closing from open state
                self.bwd_rates_I0[:, a] = np.minimum(dmt, self.ClipRate)
        
        # For I = 1 (inactivated) - vectorized with alfac/btfac
        for a in range(5):  # Forward rates with alfac factor
            if a < 4:
                self.fwd_rates_I1[:, a] = np.minimum((4-a) * amt * self.alfac, self.ClipRate) # Corrected multiplier to match legacy
            else:  # a = 4, transition to open state
                self.fwd_rates_I1[:, a] = np.minimum(gmt, self.ClipRate)
        
        for a in range(5):  # Backward rates with btfac factor
            if a < 4:
                self.bwd_rates_I1[:, a] = np.minimum((a+1) * bmt / self.btfac, self.ClipRate)
            else:  # a = 4, closing from open state
                self.bwd_rates_I1[:, a] = np.minimum(dmt, self.ClipRate)
        
        # Update Q_{I|A} rates - fully vectorized
        # Vectorized computation for all activation states
        alfac_powers = np.array([self.alfac**a for a in range(5)])
        btfac_powers = np.array([self.btfac**a for a in range(5)])
        
        # Closed states (a = 0 to 4)
        for a in range(5):
            self.inact_on_rates[:, a] = np.minimum(konlo * alfac_powers[a], self.ClipRate)
            self.inact_off_rates[:, a] = np.minimum(kofflo / btfac_powers[a], self.ClipRate)
        
        # Open state (a = 5)
        self.inact_on_rates[:, 5] = np.minimum(konop, self.ClipRate)
        self.inact_off_rates[:, 5] = np.minimum(koffop, self.ClipRate)
        
        # Drug-bound transitions - vectorized
        emt = self.epcoeff * np.exp(vt / self.epslp) * activation_scale
        zmt = self.ztcoeff * np.exp(-vt / self.ztslp) * deactivation_scale
        self.k613dis_vec = np.minimum(emt, self.ClipRate)
        self.k136dis_vec = np.minimum(zmt, self.ClipRate)

    def CurrVolt(self):
        """
        Calculates the current-voltage (I-V) relationship for the open state.

        This method computes the single-channel current (`iscft`) for each
        voltage in the `self.vt` array. It uses the Goldman-Hodgkin-Katz (GHK)
        current equation, considering sodium ion concentrations (`Nao`, `Nai`)
        and permeability (`PNasc`).

        The results are stored in `self.iscft`, which is used as a scaling
        factor during simulations to calculate the total macroscopic current based
        on the probability of the channel being in the open state.
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

        This method constructs the transition rate matrix (Q matrix) for the
        CTBN model at the specified voltage `vm`. It then solves the system
        dQ/dt = 0, subject to sum(probabilities) = 1, to find the steady-state
        probabilities for each of the 12 channel states.

        This is typically used to determine the initial state probabilities
        before starting a dynamic simulation or to analyze the channel's
        behavior at a constant holding potential.

        Args:
            vm (float): The membrane potential (in mV) at which to calculate
                        equilibrium occupancies.

        Returns:
            np.ndarray: A 1D array of 12 elements representing the equilibrium
                        probabilities for each state [P(A0,I0), P(A1,I0), ...,
                        P(A5,I0), P(A0,I1), ..., P(A5,I1)].
        """
        self.vm = vm
        self.update_rates()  # Update rates for this voltage
        
        # Get voltage index for rate lookup
        vidx = np.argmin(np.abs(self.vt - vm))
        
        # Extract rates at this voltage - all as vectors
        fwd_I0 = self.fwd_rates_I0[vidx]  # Shape (5,)
        bwd_I0 = self.bwd_rates_I0[vidx]  # Shape (5,)
        fwd_I1 = self.fwd_rates_I1[vidx]  # Shape (5,)
        bwd_I1 = self.bwd_rates_I1[vidx]  # Shape (5,)
        
        # Calculate relative equilibrium probabilities using detailed balance
        # For I=0 states - vectorized calculation
        rel_prob_A_I0 = np.ones(6)
        rel_prob_A_I0[1:] = np.cumprod(fwd_I0 / bwd_I0)
        
        # For I=1 states - vectorized calculation
        rel_prob_A_I1 = np.ones(6)
        rel_prob_A_I1[1:] = np.cumprod(fwd_I1 / bwd_I1)
        
        # Normalize within each I condition
        rel_prob_A_I0 /= rel_prob_A_I0.sum()
        rel_prob_A_I1 /= rel_prob_A_I1.sum()
        
        # Calculate relative probabilities between I=0 and I=1
        # Vectorized weighted average of transition rates
        inact_on = self.inact_on_rates[vidx]   # Shape (6,)
        inact_off = self.inact_off_rates[vidx] # Shape (6,)
        
        total_rate_I0_to_I1 = np.dot(rel_prob_A_I0, inact_on)
        total_rate_I1_to_I0 = np.dot(rel_prob_A_I1, inact_off)
        
        # Relative probability of I=1 vs I=0
        if total_rate_I1_to_I0 > 0:
            rel_prob_I1 = total_rate_I0_to_I1 / total_rate_I1_to_I0
        else:
            rel_prob_I1 = 0
        
        # Final normalization - vectorized
        total_prob = 1 + rel_prob_I1
        prob_I0 = 1 / total_prob
        prob_I1 = rel_prob_I1 / total_prob
        
        # Create flattened equilibrium probability vector
        eq_probs_flat = np.zeros(12)
        eq_probs_flat[:6] = rel_prob_A_I0 * prob_I0    # (A,I=0) states
        eq_probs_flat[6:12] = rel_prob_A_I1 * prob_I1  # (A,I=1) states
        
        # Convert to legacy format for compatibility
        pop = np.zeros(20)
        pop[1:7] = eq_probs_flat[:6]    # C1-C5, O (I=0)
        pop[7:13] = eq_probs_flat[6:12] # I1-I6 (I=1)
        
        return pop

    def NowDerivs(self, t, y):
        """
        Calculates the derivatives of state probabilities for the ODE solver.

        This function is used by `scipy.integrate.solve_ivp` during the
        simulation of a voltage sweep. It computes dP/dt for each state P,
        based on the current state probabilities `y` and the transition rates
        at the current membrane potential `self.vm`.

        The Q matrix (transition rate matrix) is constructed dynamically using
        pre-calculated rates fetched by `_get_rates_at_vm(self.vm)`.
        The derivative is then `y_dot = Q.T @ y`.

        Args:
            t (float): The current time point in the simulation (not explicitly
                       used in rate calculation as rates depend on `self.vm`
                       which is updated by the `Sweep` method).
            y (np.ndarray): A 1D array of current state probabilities for all
                            12 states.

        Returns:
            np.ndarray: A 1D array representing the derivatives (dP/dt) for
                        each of the 12 states.
        """
        # Cache number of channels for performance metrics
        N = self.numchan
        
        # ===== STAGE 1: FAST VOLTAGE LOOKUP =====
        # Optimized O(1) voltage lookup using cached values
        if not hasattr(self, '_voltage_lut_cache') or self._voltage_lut_cache[0] != self.vm:
            vidx = np.searchsorted(self.vt, self.vm)
            vidx = min(max(vidx, 0), len(self.vt) - 1)
            # Cache the result to avoid redundant calculations
            self._voltage_lut_cache = (self.vm, vidx)
        else:
            # Use cached index
            vidx = self._voltage_lut_cache[1]
        
        # ===== STAGE 2: CACHED RATE RETRIEVAL =====
        # Get rates at current voltage (with caching for repeated calls)
        if not hasattr(self, '_rate_cache') or self._rate_cache[0] != vidx:
            # Extract rates at current voltage using direct indexing
            fwd_I0 = self.fwd_rates_I0[vidx]
            bwd_I0 = self.bwd_rates_I0[vidx]
            fwd_I1 = self.fwd_rates_I1[vidx]
            bwd_I1 = self.bwd_rates_I1[vidx]
            inact_on = self.inact_on_rates[vidx]
            inact_off = self.inact_off_rates[vidx]
            
            # Cache the rates for future use
            self._rate_cache = (vidx, fwd_I0, bwd_I0, fwd_I1, bwd_I1, inact_on, inact_off)
        else:
            # Use cached rates
            _, fwd_I0, bwd_I0, fwd_I1, bwd_I1, inact_on, inact_off = self._rate_cache
        
        # ===== STAGE 3: DIRECT FLUX CALCULATION WITHOUT MATRIX CONSTRUCTION =====
        # Get views of state probability vectors for efficiency
        probs_I0 = y[:6]  # Activation state probabilities with I=0
        probs_I1 = y[6:12]  # Activation state probabilities with I=1
        
        # Allocate the derivative array once and use views
        dstdt = np.zeros_like(y)
        deriv_I0 = dstdt[:6]  # View into first half of dstdt
        deriv_I1 = dstdt[6:12]  # View into second half of dstdt
        
        # ===== CORE ALGORITHM: DIRECT FLUX CALCULATIONS =====
        # This is mathematically equivalent to matrix multiplication but avoids matrix construction
        # 1. Forward activation transitions (a → a+1) for I=0 states
        for i in range(5):
            flux = fwd_I0[i] * probs_I0[i]
            deriv_I0[i] -= flux     # Outgoing flux
            deriv_I0[i+1] += flux   # Incoming flux
        
        # 2. Backward activation transitions (a → a-1) for I=0 states
        for i in range(5):
            flux = bwd_I0[i] * probs_I0[i+1]
            deriv_I0[i+1] -= flux   # Outgoing flux
            deriv_I0[i] += flux     # Incoming flux
        
        # 3. Forward activation transitions (a → a+1) for I=1 states
        for i in range(5):
            flux = fwd_I1[i] * probs_I1[i]
            deriv_I1[i] -= flux     # Outgoing flux
            deriv_I1[i+1] += flux   # Incoming flux
        
        # 4. Backward activation transitions (a → a-1) for I=1 states
        for i in range(5):
            flux = bwd_I1[i] * probs_I1[i+1]
            deriv_I1[i+1] -= flux   # Outgoing flux
            deriv_I1[i] += flux     # Incoming flux
        
        # 5. Inactivation transitions (I=0 → I=1) - coupling between activation components
        for i in range(6):
            flux = inact_on[i] * probs_I0[i]
            deriv_I0[i] -= flux    # State leaving I=0
            deriv_I1[i] += flux    # State entering I=1
        
        # 6. Recovery transitions (I=1 → I=0) - coupling between activation components
        for i in range(6):
            flux = inact_off[i] * probs_I1[i]
            deriv_I1[i] -= flux    # State leaving I=1
            deriv_I0[i] += flux    # State entering I=0
        
        return dstdt

    def _get_rates_at_vm(self, vm):
        """
        Retrieves pre-calculated transition rates for a given membrane potential.

        This helper method finds the closest voltage in `self.vt` to the given
        `vm` and returns the corresponding pre-calculated forward activation rates
        (I=0 and I=1), backward activation rates (I=0 and I=1), inactivation
        on-rates, and inactivation off-rates.

        This is used to efficiently access rates during ODE solving (`NowDerivs`)
        and equilibrium calculations (`EquilOccup`) without recomputing them.

        Args:
            vm (float): The membrane potential (in mV) for which to retrieve rates.

        Returns:
            tuple: A tuple containing six numpy arrays:
                   (fwd_rates_I0_at_vm, bwd_rates_I0_at_vm,
                    fwd_rates_I1_at_vm, bwd_rates_I1_at_vm,
                    inact_on_rates_at_vm, inact_off_rates_at_vm)
                   Each array contains the rates for the respective transitions
                   at the specified voltage.
        """
        # Find closest voltage index
        vidx = np.argmin(np.abs(self.vt - vm))
        
        # Return all rates as a dictionary of vectors
        return {
            'fwd_I0': self.fwd_rates_I0[vidx],
            'bwd_I0': self.bwd_rates_I0[vidx],
            'fwd_I1': self.fwd_rates_I1[vidx],
            'bwd_I1': self.bwd_rates_I1[vidx],
            'inact_on': self.inact_on_rates[vidx],
            'inact_off': self.inact_off_rates[vidx],
            'k613': self.k613dis_vec[vidx],
            'k136': self.k136dis_vec[vidx]
        }

    def Sweep(self, SwpNo):
        """
        Runs a single voltage-clamp sweep simulation.

        This method simulates the channel's response to a specific sweep (`SwpNo`)
        from the current voltage protocol (`self.SwpSeq`). It involves:
        1. Setting initial state probabilities, typically using `EquilOccup` at the
           holding potential of the first epoch.
        2. Iterating through each epoch (voltage step) defined in the protocol.
        3. For each epoch:
            a. Setting `self.vm` to the epoch's voltage.
            b. Using `scipy.integrate.solve_ivp` with `self.NowDerivs` to solve
               the system of ODEs describing state probability changes over time.
            c. Storing the results (current, open probability, etc.) at sampled
               time points using `_store_ctbn_results_vectorized`.
        4. Populating `self.time` with the time vector for the simulation.

        Args:
            SwpNo (int): The sweep number (0-indexed) from the `self.SwpSeq`
                         protocol to simulate.

        Returns:
            tuple: A tuple containing:
                - t (np.ndarray): The time points at which the ODE solver evaluated
                  the solution (may not match `self.time` exactly).
                - self.SimSwp (np.ndarray): The array of simulated currents for the sweep.
        """
        # Get sweep parameters with bounds checking
        if SwpNo >= self.NumSwps or SwpNo < 0:
            raise ValueError(f"Invalid sweep number {SwpNo}")
        
        # Extract sweep parameters - vectorized indexing
        NumEpchs = int(self.SwpSeq[0, SwpNo])
        
        if NumEpchs <= 0:
            raise ValueError("Invalid number of epochs in protocol")
        
        # Calculate total simulation points
        total_points = int(self.SwpSeq[2*NumEpchs + 1, SwpNo]) + 1
        sampint = 0.005  # 5 μs sampling interval
        
        # Initialize result arrays - all vectorized
        self.SimSwp = np.zeros(total_points)   # Current trace
        self.SimOp = np.zeros(total_points)    # Open probability
        self.SimIn = np.zeros(total_points)    # Inactivated probability
        self.SimAv = np.zeros(total_points)    # Available probability
        self.SimCom = np.zeros(total_points)   # Command voltage
        # Removed SimOB (drug-bound probabilities)
        
        # Initialize CTBN state as flat vector
        self.state_probs_flat = np.zeros(12)
        self.state_probs_flat[0] = 1.0  # Start in (A=0, I=0) = C1
        
        # Pre-extract all epoch voltages and end times for the sweep
        # This avoids repeated indexing operations in the loop
        epoch_voltages = np.zeros(NumEpchs + 1)
        epoch_end_times = np.zeros(NumEpchs + 1)
        
        for e in range(NumEpchs + 1):
            if e == 0:
                epoch_voltages[e] = self.SwpSeq[2, SwpNo]
                epoch_end_times[e] = 0.0
            else:
                epoch_voltages[e] = self.SwpSeq[2 * e, SwpNo]
                epoch_end_times[e] = int(self.SwpSeq[2 * e + 1, SwpNo]) * sampint
        
        # Initial voltage
        self.vm = epoch_voltages[0]
        
        # Calculate equilibrium at holding potential
        self.CurrVolt()
        eq_pop = self.EquilOccup(self.vm)
        
        # Convert legacy equilibrium to CTBN flat format
        self.state_probs_flat[:6] = eq_pop[1:7]    # C1-C5, O (I=0)
        self.state_probs_flat[6:12] = eq_pop[7:13] # I1-I6 (I=1)
        
        # Store initial values
        self._store_ctbn_results(0, 0)
        
        # Pre-allocate reusable y0 array to avoid memory allocations in the loop
        if not hasattr(self, '_reusable_y0') or len(self._reusable_y0) < 12:
            self._reusable_y0 = np.zeros(12)  # 12 states for CTBN model
        
        # Time tracking
        current_time = 0.0
        store_idx = 1
        
        # Process each epoch
        for epoch in range(1, NumEpchs + 1):
            # Get epoch voltage and end time from pre-extracted arrays
            self.vm = epoch_voltages[epoch]
            epoch_end_time = epoch_end_times[epoch]
            
            # Update rates for new voltage
            self.update_rates()
            self.CurrVolt()
            
            # Create time evaluation points
            num_points = max(2, int((epoch_end_time - current_time) / sampint) + 1)
            t_eval = np.linspace(current_time, epoch_end_time, num_points)
            
            if len(t_eval) <= 1:
                current_time = epoch_end_time
                continue
            
            # Prepare initial conditions with reusable array
            self._reusable_y0 = self.state_probs_flat
            
            # Solve ODE system
            sol = solve_ivp(
                self.NowDerivs,
                [current_time, epoch_end_time],
                self._reusable_y0,  # Use all 12 state elements
                method='LSODA',  # Keep the same solver method
                t_eval=t_eval,
                rtol=1e-6,
                atol=1e-8
            )
            
            # Process batch of results instead of point by point
            batch_size = len(sol.t)
            if batch_size > 0:
                # Calculate indices for batch storage
                end_idx = min(store_idx + batch_size, total_points)
                batch_indices = np.arange(store_idx, end_idx)
                actual_batch_size = len(batch_indices)
                
                # Only process the batch if we have indices to store
                if actual_batch_size > 0:
                    # Prepare the batch data for processing
                    states_subset = sol.y[:, :actual_batch_size]
                    batch_states = states_subset.T  # Transpose to get [batch_size, 12]
                    batch_voltages = np.full(actual_batch_size, self.vm)  # Same voltage for all points in epoch
                    
                    # Store results in vectorized fashion
                    self._store_ctbn_results_vectorized(
                        batch_indices, 
                        batch_states, 
                        batch_voltages
                    )
                    
                    # Update state for the next epoch using the last solution point
                    self.state_probs_flat = sol.y[:, -1]
                    
                    # Update the current index after processing batch
                    store_idx = end_idx
            
            current_time = epoch_end_time
        
        # Create time vector once at the end
        self.time = np.arange(0, total_points * sampint, sampint)[:total_points]
        
        # Return peak current (most negative for inward current)
        return sol.t, self.SimSwp

    def _store_ctbn_results(self, idx, t):
        """
        Stores the simulation results for a single time point.

        This is a convenience wrapper around `_store_ctbn_results_vectorized`
        for non-batched (single time point) storage of simulation outputs like
        current, open probability, etc., based on the current `self.state_probs_flat`
        and `self.vm`.

        Args:
            idx (int): The index in the simulation output arrays (e.g., `self.SimSwp`)
                       where results for the current time point should be stored.
            t (float): The current simulation time (not directly used for storage
                       logic but often available when this method is called during
                       non-vectorized result storage).
        """
        # Call the vectorized version with single indices
        self._store_ctbn_results_vectorized(
            [idx], 
            np.array([self.state_probs_flat]), 
            np.array([self.vm])
        )
    
    def _store_ctbn_results_vectorized(self, indices, state_probs_batch, voltages):
        """
        Stores a batch of simulation results in their respective arrays.

        Calculates and stores currents, open probabilities, inactivation probabilities,
        available probabilities, and command voltages for a batch of time points.
        Uses vectorized operations for efficiency.

        Args:
            indices (np.ndarray or list): Array of indices in the output arrays
                                          (e.g., `self.SimSwp`) where results
                                          should be stored.
            state_probs_batch (np.ndarray): A 2D array where each row contains the
                                            12 state probabilities for a time point.
            voltages (np.ndarray): A 1D array of membrane potentials corresponding
                                   to each row in `state_probs_batch`.
        """
        if len(indices) == 0:
            return
        
        # Use searchsorted instead of argmin for better performance with sorted arrays
        voltage_indices = np.searchsorted(self.vt, voltages)
        # Clip to valid range without loops
        voltage_indices = np.clip(voltage_indices, 0, len(self.vt) - 1)
        
        # Extract current scaling factors for each voltage - single vectorized operation
        current_factors = self.iscft[voltage_indices]
        
        # Open state is at index 5 (vectorized extraction)
        open_probs = state_probs_batch[:, 5]
        
        # Calculate currents with optimized single operation
        scale_factor = self.numchan * self.current_scaling
        currents = open_probs * current_factors * scale_factor
        
        # Calculate aggregate probabilities with optimized axis operations
        inactivation = np.sum(state_probs_batch[:, 6:12], axis=1)
        available = np.sum(state_probs_batch[:, :6], axis=1)
        
        # Store all results at once with optimized array assignment
        self.SimSwp[indices] = currents
        self.SimOp[indices] = open_probs
        self.SimIn[indices] = inactivation
        self.SimAv[indices] = available
        self.SimCom[indices] = voltages
        
    def create_default_protocol(self, target_voltages=None, holding_potential=-80,
                               holding_duration=98, test_duration=200, tail_duration=2):
        """
        Creates a default multi-step voltage clamp protocol.

        The protocol consists of a holding period, a test pulse to various
        target voltages, and a tail pulse back to the holding potential.

        Args:
            target_voltages (list, optional): A list of voltages (mV) for the
                test pulse. Defaults to [30, 0, -20, -30, -40, -50, -60].
                The number of sweeps will be equal to the number of target voltages.
            holding_potential (float, optional): Voltage (mV) for the holding
                and tail periods. Defaults to -80 mV.
            holding_duration (float, optional): Duration (ms) of the initial
                holding period. Defaults to 98 ms.
            test_duration (float, optional): Duration (ms) of the test pulse.
                Defaults to 200 ms.
            tail_duration (float, optional): Duration (ms) of the tail pulse.
                Defaults to 2 ms.

        Sets `self.NumSwps` and `self.SwpSeq` with the generated protocol.
        Also stores the protocol under an attribute named `SwpSeq{self.BsNm}`.
        Calls `self.CurrVolt()` to ensure current-voltage relationships are up to date.
        """

        self.BsNm = "MultiStepKeyVoltages"
        
        # Use default target voltages if none provided
        if target_voltages is None:
            target_voltages = [30, 0, -20, -30, -40, -50, -60]
        
        # Convert to numpy array for vectorized operations
        target_voltages = np.array(target_voltages)
        
        # Set number of sweeps
        self.NumSwps = len(target_voltages)
        
        # Initialize protocol array
        self.SwpSeq = np.zeros((8, self.NumSwps))
        
        # Calculate time points in samples
        holding_samples = int(holding_duration / 0.005)
        test_samples = int(test_duration / 0.005)
        tail_samples = int(tail_duration / 0.005)
        total_samples = holding_samples + test_samples + tail_samples
        
        # Vectorized protocol setup
        # Row 0: Number of epochs (3 for all sweeps)
        self.SwpSeq[0, :] = 3
        
        # Epoch 1: Holding period
        self.SwpSeq[2, :] = holding_potential
        self.SwpSeq[3, :] = holding_samples
        
        # Epoch 2: Step to target voltages (vectorized assignment)
        self.SwpSeq[4, :] = target_voltages
        self.SwpSeq[5, :] = holding_samples + test_samples
        
        # Epoch 3: Tail period
        self.SwpSeq[6, :] = holding_potential
        self.SwpSeq[7, :] = total_samples
        
        # Validation using vectorized comparison
        assert self.NumSwps == len(target_voltages), "Voltage count mismatch"
        assert np.allclose(self.SwpSeq[4,:], target_voltages), "Voltage assignment error"
        
        # Store protocol
        setattr(self, f"SwpSeq{self.BsNm}", self.SwpSeq.copy())
        
        # Recalculate voltage-dependent currents
        self.CurrVolt()