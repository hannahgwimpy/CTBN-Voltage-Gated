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

        # --- Parameters for Phantom State Extension Demonstration ---
        self.demonstrate_cooperative_transition = False
        self.k_coop = 100.0  # Rate for S_A -> S_P (e.g., 1/ms)
        self.k_phantom = 1e6 # Rate for S_P -> S_B (e.g., 1/ms, very fast)
        # ----------------------------------------------------------

        
        # CTBN state variables - no change needed, these are scalars
        self.A = 0  # Activation index (0-5)
        self.I = 0  # Inactivation flag (0-1)
        self.num_states = 12 # Total number of states in the model
    
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

        If `self.demonstrate_cooperative_transition` is True, this method
        modifies the forward and backward rates for the specific cooperative
        pathway (A0,I0 <-> A1,I0 <-> A2,I0, corresponding to states 0, 1, 2)
        to use `self.k_coop` and `self.k_phantom` and sets reverse rates to zero
        for this pathway. All other state transitions in the model remain active
        and use their standard voltage-dependent rates.
        Otherwise, it computes dP/dt for all states based on the full model
        using all standard rates.

        Args:
            t (float): Current time.
            y (np.ndarray): Current state probabilities.

        Returns:
            np.ndarray: Derivatives (dP/dt) for each state.
        """
        dstdt = np.zeros_like(y) # Initialize all derivatives to zero

        # ===== STAGE 1: FAST VOLTAGE LOOKUP =====
        if not hasattr(self, '_voltage_lut_cache') or self._voltage_lut_cache[0] != self.vm:
            vidx = np.searchsorted(self.vt, self.vm)
            vidx = min(max(vidx, 0), len(self.vt) - 1)
            self._voltage_lut_cache = (self.vm, vidx)
        else:
            vidx = self._voltage_lut_cache[1]
        
        # ===== STAGE 2: CACHED RATE RETRIEVAL =====
        # Retrieve original, unmodified rates from the model's tables or cache
        if not hasattr(self, '_rate_cache') or self._rate_cache[0] != vidx:
            _fwd_I0_orig = self.fwd_rates_I0[vidx]
            _bwd_I0_orig = self.bwd_rates_I0[vidx]
            _fwd_I1_orig = self.fwd_rates_I1[vidx]
            _bwd_I1_orig = self.bwd_rates_I1[vidx]
            _inact_on_orig = self.inact_on_rates[vidx]
            _inact_off_orig = self.inact_off_rates[vidx]
            self._rate_cache = (vidx, _fwd_I0_orig, _bwd_I0_orig, _fwd_I1_orig, _bwd_I1_orig, _inact_on_orig, _inact_off_orig)
        else:
            # Use cached rates
            _, _fwd_I0_orig, _bwd_I0_orig, _fwd_I1_orig, _bwd_I1_orig, _inact_on_orig, _inact_off_orig = self._rate_cache

        # Make copies of rates that might be modified for the demonstration
        current_fwd_I0 = np.copy(_fwd_I0_orig)
        current_bwd_I0 = np.copy(_bwd_I0_orig)
        
        # Rates that are not modified by the demonstration can be used directly from original retrieved versions
        current_fwd_I1 = _fwd_I1_orig
        current_bwd_I1 = _bwd_I1_orig
        current_inact_on = _inact_on_orig
        current_inact_off = _inact_off_orig

        if self.demonstrate_cooperative_transition:
            # Override specific rates for the cooperative pathway S_A -> S_P -> S_B
            # This pathway is mapped to states (A0,I0) -> (A1,I0) -> (A2,I0)
            # which correspond to indices 0, 1, 2 of the probs_I0 array.
            
            # fwd_I0[0] is rate for (A0,I0) -> (A1,I0)
            current_fwd_I0[0] = self.k_coop
            # fwd_I0[1] is rate for (A1,I0) -> (A2,I0)
            current_fwd_I0[1] = self.k_phantom
            
            # bwd_I0[0] is rate for (A1,I0) -> (A0,I0)
            current_bwd_I0[0] = 0.0  # Make unidirectional for demonstration clarity
            # bwd_I0[1] is rate for (A2,I0) -> (A1,I0)
            current_bwd_I0[1] = 0.0  # Make unidirectional for demonstration clarity

        # ===== STAGE 3: DIRECT FLUX CALCULATION WITHOUT MATRIX CONSTRUCTION =====
        probs_I0 = y[:6]  # Activation state probabilities with I=0
        probs_I1 = y[6:12] # Activation state probabilities with I=1
        
        deriv_I0 = dstdt[:6]  # View into first half of dstdt
        deriv_I1 = dstdt[6:12] # View into second half of dstdt
        
        # 1. Forward activation transitions (a → a+1) for I=0 states
        for i in range(5):
            flux = current_fwd_I0[i] * probs_I0[i] # Uses potentially modified rates
            deriv_I0[i] -= flux
            deriv_I0[i+1] += flux
        
        # 2. Backward activation transitions (a → a-1) for I=0 states
        for i in range(5):
            flux = current_bwd_I0[i] * probs_I0[i+1] # Uses potentially modified rates
            deriv_I0[i+1] -= flux
            deriv_I0[i] += flux
        
        # 3. Forward activation transitions (a → a+1) for I=1 states
        for i in range(5):
            flux = current_fwd_I1[i] * probs_I1[i] # Uses original rates
            deriv_I1[i] -= flux
            deriv_I1[i+1] += flux
        
        # 4. Backward activation transitions (a → a-1) for I=1 states
        for i in range(5):
            flux = current_bwd_I1[i] * probs_I1[i+1] # Uses original rates
            deriv_I1[i+1] -= flux
            deriv_I1[i] += flux
        
        # 5. Inactivation transitions (I=0 → I=1) - coupling between activation components
        for i in range(6):
            flux = current_inact_on[i] * probs_I0[i] # Uses original rates
            deriv_I0[i] -= flux
            deriv_I1[i] += flux
        
        # 6. Recovery transitions (I=1 → I=0) - coupling between activation components
        for i in range(6):
            flux = current_inact_off[i] * probs_I1[i] # Uses original rates
            deriv_I1[i] -= flux
            deriv_I0[i] += flux
                
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

class AnticonvulsantCTBNMarkovModel:
    """
    Anticonvulsant-extended CTBN Markov model for simulating sodium channel currents.
    
    This model extends the CTBN approach to include drug binding based on Kuo 1998.
    Uses a 3D state space (A,I,D) where:
    - A ∈ {0,1,2,3,4,5}: activation index
    - I ∈ {0,1}: inactivation flag  
    - D ∈ {0,1}: drug binding flag
    
    Total states: 6×2×2 = 24 states, equivalent to legacy 24-state model.
    Key findings from Kuo 1998:
    - Common external binding site for DPH, CBZ, LTG
    - Higher affinity for inactivated states (KI ~9-25 μM) vs resting (KR ~mM)
    - Single occupancy: one drug molecule per channel
    - External application only
    """
    
    def __init__(self, drug_concentration=0.0, drug_type='mixed'):
        """
        Initialize the anticonvulsant CTBN model.
        
        Args:
            drug_concentration (float): Anticonvulsant concentration in μM
            drug_type (str): 'CBZ', 'LTG', 'DPH', or 'mixed' for average parameters
        """
        self.NumSwps = 0
        self.num_states = 25  # 25 states for API compatibility (24 actual + padding)
        self.drug_concentration = drug_concentration  # μM
        self.drug_type = drug_type.upper()
        
        # CTBN state variables
        self.A = 0  # Activation index (0-5)
        self.I = 0  # Inactivation flag (0-1) 
        self.D = 0  # Drug binding flag (0-1)
        
        # Initialize membrane voltage
        self.vm = -80  # Default holding potential
        
        # Initialize parameters and data structures
        self.init_parameters()
        self.init_waves()
        
        # Calculate rates and currents
        self.update_rates()
        self.CurrVolt()
        self.create_default_protocol()

    def set_drug_type(self, drug_type):
        """
        Set the type of anticonvulsant drug and update model parameters.

        Args:
            drug_type (str): 'CBZ', 'LTG', 'DPH', or 'mixed'.
        """
        self.drug_type = drug_type.upper()
        self.init_parameters()  # Re-initialize drug-specific parameters
        self.update_rates()     # Re-calculate transition rates with new drug type

    def set_drug_concentration(self, drug_concentration):
        """
        Set the anticonvulsant drug concentration and update model parameters.

        Args:
            drug_concentration (float): Concentration in μM.
        """
        self.drug_concentration = drug_concentration
        self.update_rates()     # Re-calculate transition rates with new concentration
        
    def init_parameters(self):
        """
        Initialize biophysical parameters consistent with Kuo 1998.
        
        Sets up both intrinsic channel parameters (same as CTBN model) and
        drug-specific parameters based on Kuo 1998 experimental data.
        """
        # =============  INTRINSIC CHANNEL PARAMETERS (unchanged from CTBN) =============
        
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
        
        # Calculate alfac and btfac as in original code
        self.alfac = np.sqrt(np.sqrt(self.ConHiCoeff / self.ConCoeff))
        self.btfac = np.sqrt(np.sqrt(self.CoffCoeff / self.CoffHiCoeff))
        
        # ============= DRUG-SPECIFIC PARAMETERS (Kuo 1998 based) =============
        
        # Drug-specific binding affinities from Kuo 1998 Table/Results
        self.drug_params = {
            'CBZ': {
                'KI_inactivated': 25.0,  # μM - from Kuo 1998 steady-state analysis
                'recovery_tau': 189.0,   # ms - average of 180-197 ms from kinetic fits
                'k_off': 1.0 / 189.0     # /ms - calculated from recovery time constant
            },
            'LTG': {
                'KI_inactivated': 9.0,   # μM - from Kuo 1998 steady-state analysis  
                'recovery_tau': 321.0,   # ms - average of 317-325 ms from kinetic fits
                'k_off': 1.0 / 321.0     # /ms - calculated from recovery time constant
            },
            'DPH': {
                'KI_inactivated': 9.0,   # μM - from Kuo 1998 steady-state analysis
                'recovery_tau': 189.0,   # ms - similar to CBZ based on Kuo 1998 data
                'k_off': 1.0 / 189.0     # /ms - calculated from recovery time constant
            },
            'MIXED': {
                'KI_inactivated': 15.0,  # μM - average of the three drugs
                'recovery_tau': 233.0,   # ms - average recovery time
                'k_off': 1.0 / 233.0     # /ms - calculated from average recovery time
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
        
        # Update drug-dependent rates
        self._update_drug_rates()
    
    def _update_drug_rates(self):
        """
        Update drug binding rates based on current concentration.
        
        This function calculates concentration-dependent binding rates using the
        relationship: k_on_effective = k_on_base × concentration
        
        From Kuo 1998: Drug binding follows simple bimolecular kinetics with
        much higher affinity for inactivated states than resting states.
        """
        # Calculate concentration-dependent binding rates
        # k_on_effective = k_on_base × [drug_concentration]
        self.k_on_resting = self.k_on_resting_base * self.drug_concentration
        self.k_on_inactivated = self.k_on_inactivated_base * self.drug_concentration
        
        # k_off is concentration-independent (drug-specific intrinsic unbinding rate)
        self.k_off_resting = self.k_off
        self.k_off_inactivated = self.k_off
    
    def init_waves(self):
        """
        Initialize data structures with FLATTENED memory layout for cache efficiency.
        
        Optimization 1: Use 2D arrays instead of 4D for better cache locality.
        Rate organization: [voltage_idx, rate_flat_idx]
        where rate_flat_idx = (I * 2 + D) * 5 + a for activation rates
        """
        self.vt = np.arange(-200, 201)
        self.iscft = np.zeros_like(self.vt)
        
        # Initialize CTBN state probability
        self.state_probs_flat = np.zeros(25)
        self.state_probs_flat[0] = 1.0  # Start in (A=0, I=0, D=0)
        
        # ============= FLATTENED MEMORY LAYOUT (Optimization 1) =============
        num_v = len(self.vt)
        
        # Flattened activation rates: [voltages, (I,D,a) flattened]
        # Layout: I=0,D=0,a=0-4 | I=0,D=1,a=0-4 | I=1,D=0,a=0-4 | I=1,D=1,a=0-4
        self.fwd_rates_flat = np.zeros((num_v, 20))  # 4 conditions × 5 rates
        self.bwd_rates_flat = np.zeros((num_v, 20))
        
        # Flattened inactivation rates: [voltages, (D,A) flattened]  
        # Layout: D=0,A=0-5 | D=1,A=0-5
        self.inact_on_rates_flat = np.zeros((num_v, 12))  # 2 D states × 6 A states
        self.inact_off_rates_flat = np.zeros((num_v, 12))
        
        # Drug binding rates (unchanged - already efficient)
        self.drug_on_rates_I0 = np.zeros(6)
        self.drug_off_rates_I0 = np.zeros(6)
        self.drug_on_rates_I1 = np.zeros(6)
        self.drug_off_rates_I1 = np.zeros(6)
        
        # Pre-allocate contiguous buffers for rate caching (Optimization 3)
        self._rate_cache_buffer = {
            'fwd': np.zeros(20),
            'bwd': np.zeros(20),
            'inact_on': np.zeros(12),
            'inact_off': np.zeros(12)
        }
        self._last_vidx = -1
        
        # Pre-allocate work arrays for vectorized NowDerivs
        self._state_work_array = np.zeros((4, 6))  # [I×D combinations, A states]
        self._deriv_work_array = np.zeros((4, 6))
        
        # Calculate initial rates
        self.update_rates()

    def update_rates(self):
        """Recalculates and updates all voltage-dependent state transition rates by calling stRatesVolt."""
        self.stRatesVolt()

    def stRatesVolt(self):
        """
        Calculate rates using FLATTENED memory layout for better cache performance.
        
        Optimization 1: Populate flattened arrays with better memory access patterns.
        """
        vt = self.vt
        
        # Basic voltage-dependent rates (unchanged)
        amt = self.alcoeff * np.exp(vt / self.alslp)
        bmt = self.btcoeff * np.exp(-vt / self.btslp)
        gmt = self.gmcoeff * np.exp(vt / self.gmslp)
        dmt = self.dlcoeff * np.exp(-vt / self.dlslp)
        
        konlo = self.ConCoeff * np.exp(vt / self.ConSlp)
        kofflo = self.CoffCoeff * np.exp(-vt / self.CoffSlp)
        konop = self.OpOnCoeff * np.exp(vt / self.OpOnSlp)
        koffop = self.OpOffCoeff * np.exp(-vt / self.OpOffSlp)
        
        # ============= POPULATE FLATTENED ARRAYS =============
        # Helper function to calculate flattened index
        def act_idx(i, d, a):
            return (i * 2 + d) * 5 + a
        
        # Forward activation rates - vectorized population
        for a in range(4):
            # I=0, D=0
            self.fwd_rates_flat[:, act_idx(0, 0, a)] = np.minimum((4-a) * amt, self.ClipRate)
            # I=0, D=1 (same as drug-free)
            self.fwd_rates_flat[:, act_idx(0, 1, a)] = np.minimum((4-a) * amt, self.ClipRate)
            # I=1, D=0
            self.fwd_rates_flat[:, act_idx(1, 0, a)] = np.minimum((4-a) * amt * self.alfac, self.ClipRate)
            # I=1, D=1
            self.fwd_rates_flat[:, act_idx(1, 1, a)] = np.minimum((4-a) * amt * self.alfac, self.ClipRate)
        
        # Open state transitions (a=4) - all conditions
        for i in range(2):
            for d in range(2):
                self.fwd_rates_flat[:, act_idx(i, d, 4)] = np.minimum(gmt, self.ClipRate)
        
        # Backward activation rates
        for a in range(4):
            rate_I0 = np.minimum((a+1) * bmt, self.ClipRate)
            rate_I1 = np.minimum((a+1) * bmt / self.btfac, self.ClipRate)
            
            self.bwd_rates_flat[:, act_idx(0, 0, a)] = rate_I0
            self.bwd_rates_flat[:, act_idx(0, 1, a)] = rate_I0
            self.bwd_rates_flat[:, act_idx(1, 0, a)] = rate_I1
            self.bwd_rates_flat[:, act_idx(1, 1, a)] = rate_I1
        
        # Closing from open state (a=4)
        for i in range(2):
            for d in range(2):
                self.bwd_rates_flat[:, act_idx(i, d, 4)] = np.minimum(dmt, self.ClipRate)
        
        # ============= INACTIVATION RATES (FLATTENED) =============
        def inact_idx(d, a):
            return d * 6 + a
        
        # Vectorized calculation
        alfac_powers = self.alfac ** np.arange(5)
        btfac_powers = self.btfac ** np.arange(5)
        
        # Populate for both D=0 and D=1
        for d in range(2):
            # Closed states - vectorized across voltages
            for a in range(5):
                self.inact_on_rates_flat[:, inact_idx(d, a)] = np.minimum(
                    konlo * alfac_powers[a], self.ClipRate
                )
                self.inact_off_rates_flat[:, inact_idx(d, a)] = np.minimum(
                    kofflo / btfac_powers[a], self.ClipRate
                )
            
            # Open state
            self.inact_on_rates_flat[:, inact_idx(d, 5)] = np.minimum(konop, self.ClipRate)
            self.inact_off_rates_flat[:, inact_idx(d, 5)] = np.minimum(koffop, self.ClipRate)
        
        # Update drug-specific rates
        self._update_drug_rates()
        
        # Drug binding rates (unchanged)
        self.drug_on_rates_I0[:] = self.k_on_resting
        self.drug_off_rates_I0[:] = self.k_off_resting
        self.drug_on_rates_I1[:] = self.k_on_inactivated
        self.drug_off_rates_I1[:] = self.k_off_inactivated

    def CurrVolt(self):
        """
        Calculate the current-voltage (I-V) relationship for conducting states.
        
        Uses Goldman-Hodgkin-Katz (GHK) current equation same as parent models.
        
        Key difference from legacy models:
        - Only drug-free open states conduct current (index 5 in drug-free states)
        - Drug-bound open states are blocked/non-conducting (Kuo 1998)
        
        The results are stored in self.iscft for use during simulations.
        """
        # Set temperature in Kelvin to fixed reference value (22°C)
        # Same as parent models for consistency
        self.Tkel = 273.15 + 22.0
        
        # No temperature scaling for permeability
        scaled_PNasc = self.PNasc
        
        # Vectorized voltage processing (same as parent models)
        v_volts = self.vt * 1e-3  # Convert all voltages from mV to V
        
        # Create masks for near-zero voltages to handle division by zero
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
        Calculate equilibrium state occupancies using detailed balance.
        
        Updated to work with FLATTENED array memory layout.
        
        Args:
            vm (float): Membrane potential in mV
            
        Returns:
            np.ndarray: 25-element array (24 states + 1 padding) for API compatibility
        """
        self.vm = vm
        self.update_rates()  # Update rates for this voltage
        
        # Get voltage index for rate lookup
        vidx = np.argmin(np.abs(self.vt - vm))
        
        # Safe division helper function
        def safe_div(a, b, default=0.0):
            if np.isscalar(b):
                return a / b if abs(b) > 1e-10 else default
            else:
                result = np.full_like(a, default, dtype=float)
                mask = np.abs(b) > 1e-10
                if np.any(mask):
                    result[mask] = a[mask] / b[mask]
                return result
        
        # ============= EXTRACT RATES FROM FLATTENED ARRAYS =============
        # Helper functions for index calculation (must match stRatesVolt)
        def act_idx(i, d, a):
            return (i * 2 + d) * 5 + a
        
        def inact_idx(d, a):
            return d * 6 + a
        
        # Extract activation rates for each condition
        fwd_I0D0 = np.array([self.fwd_rates_flat[vidx, act_idx(0, 0, a)] for a in range(5)])
        bwd_I0D0 = np.array([self.bwd_rates_flat[vidx, act_idx(0, 0, a)] for a in range(5)])
        
        fwd_I1D0 = np.array([self.fwd_rates_flat[vidx, act_idx(1, 0, a)] for a in range(5)])
        bwd_I1D0 = np.array([self.bwd_rates_flat[vidx, act_idx(1, 0, a)] for a in range(5)])
        
        fwd_I0D1 = np.array([self.fwd_rates_flat[vidx, act_idx(0, 1, a)] for a in range(5)])
        bwd_I0D1 = np.array([self.bwd_rates_flat[vidx, act_idx(0, 1, a)] for a in range(5)])
        
        fwd_I1D1 = np.array([self.fwd_rates_flat[vidx, act_idx(1, 1, a)] for a in range(5)])
        bwd_I1D1 = np.array([self.bwd_rates_flat[vidx, act_idx(1, 1, a)] for a in range(5)])
        
        # Extract inactivation rates
        inact_on_D0 = np.array([self.inact_on_rates_flat[vidx, inact_idx(0, a)] for a in range(6)])
        inact_off_D0 = np.array([self.inact_off_rates_flat[vidx, inact_idx(0, a)] for a in range(6)])
        
        inact_on_D1 = np.array([self.inact_on_rates_flat[vidx, inact_idx(1, a)] for a in range(6)])
        inact_off_D1 = np.array([self.inact_off_rates_flat[vidx, inact_idx(1, a)] for a in range(6)])
        
        # ============= DETAILED BALANCE FOR ACTIVATION STATES =============
        # Calculate relative equilibrium probabilities using detailed balance
        
        # For I=0,D=0 states (drug-free resting)
        rel_prob_A_I0D0 = np.ones(6)
        rel_prob_A_I0D0[1:] = np.cumprod(safe_div(fwd_I0D0, bwd_I0D0, 1.0))
        rel_prob_A_I0D0 /= rel_prob_A_I0D0.sum()
        
        # For I=1,D=0 states (drug-free inactivated)
        rel_prob_A_I1D0 = np.ones(6)
        rel_prob_A_I1D0[1:] = np.cumprod(safe_div(fwd_I1D0, bwd_I1D0, 1.0))
        rel_prob_A_I1D0 /= rel_prob_A_I1D0.sum()
        
        # For I=0,D=1 states (drug-bound resting)
        rel_prob_A_I0D1 = np.ones(6)
        rel_prob_A_I0D1[1:] = np.cumprod(safe_div(fwd_I0D1, bwd_I0D1, 1.0))
        rel_prob_A_I0D1 /= rel_prob_A_I0D1.sum()
        
        # For I=1,D=1 states (drug-bound inactivated)
        rel_prob_A_I1D1 = np.ones(6)
        rel_prob_A_I1D1[1:] = np.cumprod(safe_div(fwd_I1D1, bwd_I1D1, 1.0))
        rel_prob_A_I1D1 /= rel_prob_A_I1D1.sum()
        
        # ============= INACTIVATION EQUILIBRIUM =============
        # Calculate relative probabilities between I=0 and I=1 states
        
        # For drug-free states (D=0)
        total_rate_I0_to_I1_D0 = np.dot(rel_prob_A_I0D0, inact_on_D0)
        total_rate_I1_to_I0_D0 = np.dot(rel_prob_A_I1D0, inact_off_D0)
        
        # Relative probability of I=1 vs I=0 for drug-free states
        rel_prob_I1_D0 = safe_div(total_rate_I0_to_I1_D0, total_rate_I1_to_I0_D0, 0.0)
        
        # For drug-bound states (D=1)
        total_rate_I0_to_I1_D1 = np.dot(rel_prob_A_I0D1, inact_on_D1)
        total_rate_I1_to_I0_D1 = np.dot(rel_prob_A_I1D1, inact_off_D1)
        
        # Relative probability of I=1 vs I=0 for drug-bound states
        rel_prob_I1_D1 = safe_div(total_rate_I0_to_I1_D1, total_rate_I1_to_I0_D1, 0.0)
        
        # ============= DRUG BINDING EQUILIBRIUM =============
        # P(D=1)/P(D=0) = k_on/k_off for each (I) condition
        
        # Drug binding equilibrium for resting states (I=0)
        drug_factor_I0 = safe_div(self.k_on_resting, self.k_off_resting, 0.0)
        
        # Drug binding equilibrium for inactivated states (I=1)
        drug_factor_I1 = safe_div(self.k_on_inactivated, self.k_off_inactivated, 0.0)
        
        # ============= COMBINE ALL EQUILIBRIA =============
        # Calculate unnormalized probabilities for all states
        
        # Using drug-free resting as reference (P(I=0,D=0) = 1)
        prob_I0D0_unnorm = 1.0
        prob_I1D0_unnorm = rel_prob_I1_D0
        prob_I0D1_unnorm = drug_factor_I0
        prob_I1D1_unnorm = drug_factor_I1 * rel_prob_I1_D1
        
        # Total unnormalized probability
        total_unnorm = prob_I0D0_unnorm + prob_I1D0_unnorm + prob_I0D1_unnorm + prob_I1D1_unnorm
        
        # Normalize
        if total_unnorm > 1e-10:
            prob_I0D0 = prob_I0D0_unnorm / total_unnorm
            prob_I1D0 = prob_I1D0_unnorm / total_unnorm
            prob_I0D1 = prob_I0D1_unnorm / total_unnorm
            prob_I1D1 = prob_I1D1_unnorm / total_unnorm
        else:
            # Fallback: all probability on drug-free resting states
            prob_I0D0 = 1.0
            prob_I1D0 = prob_I0D1 = prob_I1D1 = 0.0
        
        # ============= CONSTRUCT FINAL STATE VECTOR =============
        # Create 25-element array for API compatibility 
        eq_probs = np.zeros(25)
        
        # Drug-free resting states: (A,I=0,D=0) -> indices 0-5
        eq_probs[0:6] = rel_prob_A_I0D0 * prob_I0D0
        
        # Drug-free inactivated states: (A,I=1,D=0) -> indices 6-11
        eq_probs[6:12] = rel_prob_A_I1D0 * prob_I1D0
        
        # Drug-bound resting states: (A,I=0,D=1) -> indices 12-17
        eq_probs[12:18] = rel_prob_A_I0D1 * prob_I0D1
        
        # Drug-bound inactivated states: (A,I=1,D=1) -> indices 18-23
        eq_probs[18:24] = rel_prob_A_I1D1 * prob_I1D1
        
        # Index 24 remains 0 (padding for API compatibility)
        
        # Final safety check and normalization
        total_prob = np.sum(eq_probs[:24])
        if total_prob > 1e-10:
            eq_probs[:24] /= total_prob
        else:
            # Ultimate fallback: start in drug-free C1 state
            eq_probs[:] = 0.0
            eq_probs[0] = 1.0
        
        # Ensure no NaN values
        eq_probs = np.nan_to_num(eq_probs, nan=0.0)
        
        # Update internal state for consistency
        self.state_probs_flat[:] = eq_probs[:]
        
        # Create and update pop array for API compatibility
        self.pop = np.zeros(25)
        self.pop[:] = eq_probs[:]
        
        # Return pop for API compatibility
        return self.pop

    def NowDerivs(self, t, y):
        """
        Calculate derivatives using VECTORIZED operations without nested loops.
        
        Optimizations:
        2. Eliminate nested loops - process all (I,D) combinations vectorized
        3. Reduce array slicing - cache rates and use direct operations
        """
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return np.zeros_like(y)
        
        # ===== FAST VOLTAGE LOOKUP (unchanged) =====
        if not hasattr(self, '_voltage_lut_cache') or self._voltage_lut_cache[0] != self.vm:
            vidx = np.searchsorted(self.vt, self.vm)
            vidx = min(max(vidx, 0), len(self.vt) - 1)
            self._voltage_lut_cache = (self.vm, vidx)
        else:
            vidx = self._voltage_lut_cache[1]
        
        # ===== OPTIMIZED RATE CACHING (Optimization 3) =====
        # Only extract rates if voltage changed
        if vidx != self._last_vidx:
            # Copy contiguous slices into pre-allocated buffers
            self._rate_cache_buffer['fwd'][:] = self.fwd_rates_flat[vidx, :]
            self._rate_cache_buffer['bwd'][:] = self.bwd_rates_flat[vidx, :]
            self._rate_cache_buffer['inact_on'][:] = self.inact_on_rates_flat[vidx, :]
            self._rate_cache_buffer['inact_off'][:] = self.inact_off_rates_flat[vidx, :]
            self._last_vidx = vidx
        
        # Direct references to cached rates
        fwd_rates = self._rate_cache_buffer['fwd']
        bwd_rates = self._rate_cache_buffer['bwd']
        inact_on = self._rate_cache_buffer['inact_on']
        inact_off = self._rate_cache_buffer['inact_off']
        
        # ===== VECTORIZED STATE PROCESSING (Optimization 2) =====
        # Reshape states into work array without allocation
        # Order: [I=0,D=0], [I=0,D=1], [I=1,D=0], [I=1,D=1]
        self._state_work_array[0, :] = y[0:6]      # I=0, D=0
        self._state_work_array[1, :] = y[12:18]    # I=0, D=1
        self._state_work_array[2, :] = y[6:12]     # I=1, D=0
        self._state_work_array[3, :] = y[18:24]    # I=1, D=1
        
        # Zero out derivative array
        self._deriv_work_array[:] = 0.0
        
        # ===== VECTORIZED ACTIVATION TRANSITIONS =====
        # Process all 4 (I,D) combinations simultaneously
        for combo_idx in range(4):
            # Calculate rate indices for this combination
            rate_start = combo_idx * 5
            
            # Get state probabilities and derivatives for this combination
            probs = self._state_work_array[combo_idx, :]
            deriv = self._deriv_work_array[combo_idx, :]
            
            # Forward transitions (a → a+1) - vectorized
            fwd_flux = fwd_rates[rate_start:rate_start+5] * probs[:5]
            deriv[:5] -= fwd_flux
            deriv[1:6] += fwd_flux
            
            # Backward transitions (a+1 → a) - vectorized  
            bwd_flux = bwd_rates[rate_start:rate_start+5] * probs[1:6]
            deriv[1:6] -= bwd_flux
            deriv[:5] += bwd_flux
        
        # ===== VECTORIZED INACTIVATION TRANSITIONS =====
        # Process D=0 and D=1 simultaneously
        for d in range(2):
            # Indices into state array
            i0_idx = 0 if d == 0 else 1  # I=0 states
            i1_idx = 2 if d == 0 else 3  # I=1 states
            
            # Rate indices
            rate_idx = slice(d*6, (d+1)*6)
            
            # I=0 → I=1 transitions (vectorized across all A states)
            inact_flux = inact_on[rate_idx] * self._state_work_array[i0_idx, :]
            self._deriv_work_array[i0_idx, :] -= inact_flux
            self._deriv_work_array[i1_idx, :] += inact_flux
            
            # I=1 → I=0 transitions (vectorized across all A states)
            recov_flux = inact_off[rate_idx] * self._state_work_array[i1_idx, :]
            self._deriv_work_array[i1_idx, :] -= recov_flux
            self._deriv_work_array[i0_idx, :] += recov_flux
        
        # ===== VECTORIZED DRUG BINDING =====
        # Drug binding for I=0 (vectorized across all A states)
        drug_flux_I0 = (self.drug_on_rates_I0 * self._state_work_array[0, :] - 
                        self.drug_off_rates_I0 * self._state_work_array[1, :])
        self._deriv_work_array[0, :] -= drug_flux_I0
        self._deriv_work_array[1, :] += drug_flux_I0
        
        # Drug binding for I=1 (vectorized across all A states)
        drug_flux_I1 = (self.drug_on_rates_I1 * self._state_work_array[2, :] - 
                        self.drug_off_rates_I1 * self._state_work_array[3, :])
        self._deriv_work_array[2, :] -= drug_flux_I1
        self._deriv_work_array[3, :] += drug_flux_I1
        
        # ===== RESHAPE BACK TO FLAT ARRAY =====
        # Direct copy without creating intermediate arrays
        dstdt = np.zeros_like(y)
        dstdt[0:6] = self._deriv_work_array[0, :]    # I=0, D=0
        dstdt[6:12] = self._deriv_work_array[2, :]   # I=1, D=0  
        dstdt[12:18] = self._deriv_work_array[1, :]  # I=0, D=1
        dstdt[18:24] = self._deriv_work_array[3, :]  # I=1, D=1
        
        return dstdt

    def _get_rates_at_vm(self, vm):
        """
        Retrieve rates with improved cache efficiency using flattened arrays.
        
        Optimization 3: Return direct slices without creating views of 4D arrays.
        """
        vidx = np.searchsorted(self.vt, vm)
        vidx = np.clip(vidx, 0, len(self.vt) - 1)
        
        # Return direct slices from flattened arrays
        return {
            'fwd_flat': self.fwd_rates_flat[vidx, :],
            'bwd_flat': self.bwd_rates_flat[vidx, :],
            'inact_on_flat': self.inact_on_rates_flat[vidx, :],
            'inact_off_flat': self.inact_off_rates_flat[vidx, :],
            'drug_on_I0': self.drug_on_rates_I0,
            'drug_off_I0': self.drug_off_rates_I0,
            'drug_on_I1': self.drug_on_rates_I1,
            'drug_off_I1': self.drug_off_rates_I1
        }

    def Sweep(self, SwpNo):
        """
        Run a single voltage-clamp sweep simulation for CTBN anticonvulsant model.
        
        Follows same structure as both parent models but uses CTBN state representation
        and includes drug-bound state tracking. Maintains API compatibility.
        
        Args:
            SwpNo (int): The sweep number (0-indexed) from self.SwpSeq protocol to simulate.
            
        Returns:
            tuple: (t, self.SimSwp) where:
                - t (np.ndarray): Time points from ODE solver
                - self.SimSwp (np.ndarray): Simulated current trace
        """
        # ============= INPUT VALIDATION =============
        if SwpNo >= self.NumSwps or SwpNo < 0:
            raise ValueError(f"Invalid sweep number {SwpNo}")
        
        # Extract sweep parameters with bounds checking
        NumEpchs = int(self.SwpSeq[0, SwpNo])
        
        if NumEpchs <= 0:
            raise ValueError("Invalid number of epochs in protocol")
        
        # Calculate total simulation points
        total_points = int(self.SwpSeq[2*NumEpchs + 1, SwpNo]) + 1
        sampint = 0.005  # 5 μs sampling interval
        
        # ============= INITIALIZE RESULT ARRAYS =============
        # Pre-allocate all result arrays (API compatible with both parent models)
        self.SimSwp = np.zeros(total_points)       # Current trace
        self.SimOp = np.zeros(total_points)        # Open probability
        self.SimIn = np.zeros(total_points)        # Inactivated probability
        self.SimAv = np.zeros(total_points)        # Available probability
        self.SimCom = np.zeros(total_points)       # Command voltage
        self.SimDrugBound = np.zeros(total_points) # Drug-bound probability (anticonvulsant-specific)
        
        # ============= INITIALIZE CTBN STATE =============
        # Initialize CTBN state as flat vector (24 active states)
        self.state_probs_flat = np.zeros(25)  # 25 for API compatibility
        self.state_probs_flat[0] = 1.0  # Start in (A=0, I=0, D=0) = drug-free C1
        
        # Pre-extract all epoch voltages and end times for vectorized access
        epoch_voltages = np.zeros(NumEpchs + 1)
        epoch_end_times = np.zeros(NumEpchs + 1)
        
        for e in range(NumEpchs + 1):
            if e == 0:
                epoch_voltages[e] = self.SwpSeq[2, SwpNo]
                epoch_end_times[e] = 0.0
            else:
                epoch_voltages[e] = self.SwpSeq[2 * e, SwpNo]
                epoch_end_times[e] = int(self.SwpSeq[2 * e + 1, SwpNo]) * sampint
        
        # ============= INITIAL EQUILIBRIUM =============
        # Set initial voltage and calculate equilibrium
        self.vm = epoch_voltages[0]
        
        # Calculate equilibrium at holding potential
        self.CurrVolt()
        eq_pop = self.EquilOccup(self.vm)
        
        # Set initial state probabilities
        self.state_probs_flat[:] = eq_pop[:]
        
        # Sync with legacy pop array for API compatibility
        self.pop = np.zeros(25)
        self.pop[:] = eq_pop[:]
        
        # Store initial values
        self._store_ctbn_results(0, 0)
        
        # Pre-allocate reusable y0 array to avoid memory allocations
        if not hasattr(self, '_reusable_y0') or len(self._reusable_y0) < 24:
            self._reusable_y0 = np.zeros(24)  # 24 states for ODE solver
        
        # ============= SWEEP SIMULATION =============
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
            
            # Prepare initial conditions with reusable array (only 24 active states)
            self._reusable_y0[:] = self.state_probs_flat[:24]
            
            # Solve ODE system using CTBN NowDerivs
            sol = solve_ivp(
                self.NowDerivs,
                [current_time, epoch_end_time],
                self._reusable_y0,  # Use 24 state elements
                method='LSODA',     # Same solver as parent models
                t_eval=t_eval,
                rtol=1e-6,
                atol=1e-8
            )
            
            # ============= PROCESS RESULTS =============
            # Store solution for debugging if needed
            if hasattr(self, 'full_sol_t'):
                self.full_sol_t = sol.t
                self.full_sol_y = sol.y
            
            # Process batch of results
            batch_size = len(sol.t)
            if batch_size > 0:
                # Calculate indices for batch storage
                end_idx = min(store_idx + batch_size, total_points)
                batch_indices = np.arange(store_idx, end_idx)
                actual_batch_size = len(batch_indices)
                
                # Only process if we have indices to store
                if actual_batch_size > 0:
                    # Prepare batch data for processing
                    states_subset = sol.y[:, :actual_batch_size]
                    batch_states = states_subset.T  # Transpose to get [batch_size, 24]
                    batch_voltages = np.full(actual_batch_size, self.vm)  # Same voltage for epoch
                    
                    # Store results in vectorized fashion
                    self._store_ctbn_results_vectorized(
                        batch_indices, 
                        batch_states, 
                        batch_voltages
                    )
                    
                    # Update state for next epoch using last solution point
                    self.state_probs_flat[:24] = sol.y[:, -1]
                    self.state_probs_flat[24] = 0.0  # Keep padding at 0
                    
                    # Sync with legacy pop array
                    self.pop[:24] = sol.y[:, -1]
                    self.pop[24] = 0.0
                    
                    # Update the current index after processing batch
                    store_idx = end_idx
            
            current_time = epoch_end_time
        
        # ============= FINALIZE RESULTS =============
        # Create time vector once at the end
        self.time = np.arange(0, total_points * sampint, sampint)[:total_points]
        
        # Return results for API compatibility
        return sol.t, self.SimSwp
    
    def _store_ctbn_results(self, idx, t):
        """
        Store simulation results for a single time point (CTBN anticonvulsant version).
        
        Wrapper around vectorized version for single time point storage.
        Calculates current, probabilities, and drug-bound fraction based on current
        state_probs_flat and vm.
        
        Args:
            idx (int): Index in output arrays where results should be stored
            t (float): Current simulation time (for reference, not used in calculation)
        """
        # Call vectorized version with single indices
        self._store_ctbn_results_vectorized(
            [idx], 
            np.array([self.state_probs_flat[:24]]), 
            np.array([self.vm])
        )
    
    def _store_ctbn_results_vectorized(self, indices, state_probs_batch, voltages):
        """
        Store a batch of simulation results for anticonvulsant CTBN model.
        
        Calculates and stores currents, probabilities, and drug effects for batch of time points.
        Uses vectorized operations for efficiency, following same patterns as parent models.
        
        Key difference from parent models: Only drug-free open states conduct current.
        Drug-bound open states are blocked/non-conducting per Kuo 1998.
        
        State organization (24 states):
        - Indices 0-5:   Drug-free resting states (A=0-5, I=0, D=0)
        - Indices 6-11:  Drug-free inactivated states (A=0-5, I=1, D=0)  
        - Indices 12-17: Drug-bound resting states (A=0-5, I=0, D=1)
        - Indices 18-23: Drug-bound inactivated states (A=0-5, I=1, D=1)
        
        Args:
            indices (np.ndarray or list): Array of indices where results should be stored
            state_probs_batch (np.ndarray): 2D array [batch_size, 24] of state probabilities
            voltages (np.ndarray or float): Membrane potentials for each time point
        """
        if len(indices) == 0:
            return
        
        # ============= VOLTAGE INDEX LOOKUP =============
        # Use searchsorted for better performance with sorted arrays (same as CTBNMarkovModel)
        if np.isscalar(voltages):
            voltage_indices = np.searchsorted(self.vt, voltages)
            voltage_indices = np.clip(voltage_indices, 0, len(self.vt) - 1)
            current_factors = self.iscft[voltage_indices]
        else:
            voltage_indices = np.searchsorted(self.vt, voltages)
            voltage_indices = np.clip(voltage_indices, 0, len(self.vt) - 1)
            current_factors = self.iscft[voltage_indices]
        
        # ============= CURRENT CALCULATION =============
        # Key insight from Kuo 1998: Drug-bound channels are non-conducting when open
        # Only drug-free open state contributes to macroscopic current
        
        # Drug-free open probability: (A=5, I=0, D=0) -> index 5
        conducting_open_probs = state_probs_batch[:, 5]
        
        # Total open probability for tracking: includes drug-bound open states
        # Drug-bound open state: (A=5, I=0, D=1) -> index 17
        total_open_probs = state_probs_batch[:, 5] + state_probs_batch[:, 17]
        
        # Calculate macroscopic current from conducting states only
        scale_factor = self.numchan * self.current_scaling
        currents = conducting_open_probs * current_factors * scale_factor
        
        # ============= PROBABILITY CALCULATIONS =============
        # Calculate aggregate probabilities using vectorized operations
        
        # Inactivated probability: All states with I=1
        # Drug-free inactivated: indices 6-11 (A=0-5, I=1, D=0)
        # Drug-bound inactivated: indices 18-23 (A=0-5, I=1, D=1)  
        inactivated = (np.sum(state_probs_batch[:, 6:12], axis=1) + 
                      np.sum(state_probs_batch[:, 18:24], axis=1))
        
        # Available probability: All states with I=0 (not inactivated)
        # Drug-free available: indices 0-5 (A=0-5, I=0, D=0)
        # Drug-bound available: indices 12-17 (A=0-5, I=0, D=1)
        available = (np.sum(state_probs_batch[:, 0:6], axis=1) + 
                    np.sum(state_probs_batch[:, 12:18], axis=1))
        
        # Drug-bound probability: All states with D=1 (anticonvulsant-specific)
        # Drug-bound available: indices 12-17 (A=0-5, I=0, D=1)
        # Drug-bound inactivated: indices 18-23 (A=0-5, I=1, D=1)
        drug_bound = (np.sum(state_probs_batch[:, 12:18], axis=1) + 
                     np.sum(state_probs_batch[:, 18:24], axis=1))
        
        # ============= STORE ALL RESULTS =============
        # Store results using vectorized array assignment (optimized single operation)
        self.SimSwp[indices] = currents
        self.SimOp[indices] = total_open_probs          # Total open probability for tracking
        self.SimIn[indices] = inactivated               # Total inactivated probability
        self.SimAv[indices] = available                 # Total available probability
        self.SimCom[indices] = voltages                 # Command voltage
        self.SimDrugBound[indices] = drug_bound         # Drug-bound probability (anticonvulsant-specific)
     
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