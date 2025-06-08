import numpy as np
from scipy.integrate import solve_ivp


class MarkovModel:
    def __init__(self):
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
        """Initialize all model parameters - no matrices needed"""
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
        
        # Current scaling factor to match HH model
        self.current_scaling = 0.0117

        # Initialize membrane voltage
        self.vm = -80  # Default holding potential
        
        self.PNasc = 1e-5 
        
        # Pre-allocate reusable arrays
        self._reusable_y0 = np.zeros(12)  # For ODE initial conditions - 12 states as used in NowDerivs
    
    def init_waves(self):
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
        """Create vectorized rate storage arrays"""
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
        """Calculate voltage-dependent state transition rates - fully vectorized"""
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
        """Update scalar rate constants for current vm from vectorized arrays"""
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
        """Calculate voltage-dependent sodium currents using Goldman-Hodgkin-Katz equation - fully vectorized"""
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
        """Calculate equilibrium occupancy for a given voltage - vectorized"""
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
            """Safe division that handles zero denominators without evaluation errors"""
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
        """Simulate voltage-clamp protocol - vectorized version"""
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
        """Store results for a single time point"""
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
        """Store results for multiple time points - vectorized"""
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
        """Create protocol with specified target voltages - vectorized"""
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