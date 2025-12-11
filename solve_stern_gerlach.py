import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_u, hbar, k
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf
from fe_laplace_solver import solve_stern_gerlach_potential

# ==========================================
# 2. THE MAGNET CLASS (Logic Fixed)
# ==========================================
class SternGerlachMagnet:
    def __init__(self, B0=1.5):
        self.target_B0 = B0
        
        # Placeholders - We do NOT run the solver here yet.
        # We wait until the first time the simulation actually needs data.
        self.interp_Bx = None
        self.interp_Bz = None
        self.interp_Bmag = None
        self.interp_gradX = None
        self.interp_gradZ = None

    def _initialize_field_data(self):
        """
        Runs the solver ONCE and builds the Scipy Interpolators.
        """
        # 1. Run Solver (The Heavy Part)
        x_mm, z_mm, raw_Bx, raw_Bz, raw_Bmag = solve_stern_gerlach_potential()
        
        # 2. Convert Grid Coordinates to METERS (Simulation Units)
        # The atoms fly in meters, so the grid axes must be in meters.
        x_m = x_mm * 1e-3
        z_m = z_mm * 1e-3
        
        # 3. Scale Field Strength to Target B0
        mid = len(x_m) // 2
        center_field = raw_Bmag[mid, mid]
        scale = self.target_B0 / center_field if center_field != 0 else 1.0
        
        print(f"  [System] Scaling field by {scale:.2f} to reach {self.target_B0}T")
        
        Bx = raw_Bx * scale
        Bz = raw_Bz * scale
        Bmag = raw_Bmag * scale
        
        # 4. Calculate Gradients (In Meters!)
        # d/dx = delta_B / delta_x_meters
        dx_m = x_m[1] - x_m[0]
        dz_m = z_m[1] - z_m[0]
        
        grad_B_x, grad_B_z = np.gradient(Bmag, dx_m, dz_m)

        # 5. Build Interpolators (The Cache)
        # bounds_error=False allows atoms to fly slightly out of range without crashing
        # fill_value=0 means field is zero outside the box
        kw = {'bounds_error': False, 'fill_value': 0}
        
        self.interp_Bx = RegularGridInterpolator((x_m, z_m), Bx, fill_value=0)
        self.interp_Bz = RegularGridInterpolator((x_m, z_m), Bz,          fill_value=0)
        self.interp_Bmag = RegularGridInterpolator((x_m, z_m), Bmag,      fill_value=0)
        self.interp_gradX = RegularGridInterpolator((x_m, z_m), grad_B_x, fill_value=0)
        self.interp_gradZ = RegularGridInterpolator((x_m, z_m), grad_B_z, fill_value=0)
        
        print("  [System] Magnet Interpolators Built & Ready.")

    def get_field_vector_and_gradient(self, x, z):
        """
        Fast lookup.
        """
        # LAZY LOAD CHECK:
        # This ensures the solver runs exactly once.
        if self.interp_Bx is None:
            self._initialize_field_data()
            
        # Stack coordinates: (N, 2)
        pts = np.column_stack((x, z))
        
        # Direct Query (Fast)
        Bx = self.interp_Bx(pts)
        Bz = self.interp_Bz(pts)
        Bmag = self.interp_Bmag(pts)
        dBdx = self.interp_gradX(pts)
        dBdz = self.interp_gradZ(pts)
        
        return Bx, Bz, Bmag, dBdx, dBdz

# ==========================================
# 3. EXPERIMENT SETUP
# ==========================================
class ExperimentSetup:
    def __init__(self):
        self.mass = 107.868 * m_u # Silver
        self.mu_b = 9.274e-24     
        self.g_s = 2.0023         
        
        # Geometry (Meters)
        self.y_oven = -0.17 
        self.y_slit2 = -0.04 
        self.y_mag_start = 0.0
        self.y_mag_end = 0.035 
        self.y_detector = self.y_mag_end+0.10 # 10cm past magnet
        
        self.check_slits_called = 0
        self.atom_slits_checked = 0

    def check_slits(self, x, z, y_pos):
        # S2, Rectangular Slit Geometry
        if np.abs(y_pos - self.y_slit2) < 0.005:
            w = 0.8e-3   # 0.8 mm
            h = 0.035e-3 # 35 microns
            
            passed = (np.abs(x) < w/2) & (np.abs(z) < h/2)
            return passed
        return np.ones_like(x, dtype=bool)

# ==========================================
# 4. SIMULATION LOOP
# ==========================================
def run_simulation_fixed():
    setup = ExperimentSetup()
    
    # Initialize Magnet Object (Solver does NOT run yet)
    magnet = SternGerlachMagnet(B0=1.4) 
    
    # 1. Generate Atoms
    N = 5000000
    print(f"Initializing {N} Silver atoms...")
    
    # Initial cluster
    x = np.random.normal(0, 1e-4, N)
    z = np.random.normal(0, 1e-4, N) 
    y = np.full(N, setup.y_oven)
    
    # Velocity (Thermal)
    T_oven = 1300 
    v_th = np.sqrt(2 * k * T_oven / setup.mass)
    
    raw_v = np.random.standard_gamma(4, N) 
    vy = raw_v * (v_th / np.sqrt(2))
    
    # Geometric Cooling (Aim at slit)
    dist = np.abs(setup.y_slit2 - setup.y_oven)
    theta_x = (0.8e-3 / (2*dist)) - np.abs(x)
    theta_z = (0.035e-3 / (2*dist)) -np.abs(z)
    
    vx = np.random.uniform(-theta_x, theta_x, N) * vy
    vz = np.random.uniform(-theta_z, theta_z, N) * vy
    
    spins = np.random.choice([1, -1], N)
    dt = 1e-7

    final_x, final_z, final_s = [], [], []
    v_target = 675.0  # target velocity in m/s
    v_window = 75.0   # pass atoms within +/- 10 m/s
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    velocity_keep = (v_mag > (v_target - v_window)) & (v_mag < (v_target + v_window-50))

    # Filter all initial arrays
    x, y, z = x[velocity_keep], y[velocity_keep], z[velocity_keep]
    vx, vy, vz = vx[velocity_keep], vy[velocity_keep], vz[velocity_keep]
    spins = spins[velocity_keep]
    # vy = vy[velocity_keep] # Make sure the specific forward component is updated too
    N = len(x) # Update atom count for printing

    print(f"Velocity selected: {N} atoms passed the filter.")
    # Loop
    step = 0
    while len(x) > 0:
        step += 1
        x += vx * dt
        y += vy * dt
        z += vz * dt
        
        # --- 1. Slit Check ---
        y_prev = y - vy * dt
        crossing_mask = (y >= setup.y_slit2) & (y_prev < setup.y_slit2)
        
        if np.any(crossing_mask):
            passed = setup.check_slits(x[crossing_mask], z[crossing_mask], setup.y_slit2)
            
            # Logic: We keep everyone EXCEPT the crossers who failed
            keep_mask = np.ones(len(x), dtype=bool)
            
            # Get global indices of crossers
            global_indices = np.where(crossing_mask)[0]
            # Get global indices of failures
            failed_indices = global_indices[~passed]
            
            keep_mask[failed_indices] = False
            
            # Filter
            x, y, z = x[keep_mask], y[keep_mask], z[keep_mask]
            vx, vy, vz = vx[keep_mask], vy[keep_mask], vz[keep_mask]
            spins = spins[keep_mask]

        # --- 2. Detector Check ---
        finished_mask = y > setup.y_detector
        if np.any(finished_mask):
            final_x.extend(x[finished_mask])
            final_z.extend(z[finished_mask])
            final_s.extend(spins[finished_mask])
            
            keep = ~finished_mask
            x, y, z = x[keep], y[keep], z[keep]
            vx, vy, vz = vx[keep], vy[keep], vz[keep]
            spins = spins[keep]

        # --- 3. Magnetic Interaction ---
        if len(x) > 0:
            # Simple box check for magnet region
            in_mag = (y > setup.y_mag_start) & (y < setup.y_mag_end)
            
            if np.any(in_mag):
                # >>> THIS IS WHERE IT REQUESTS THE FIELD <<<
                # The first time it hits this line, it will run the solver.
                # Every subsequent time, it uses the cached interpolator.
                Bx, Bz, B_mag, dBdx, dBdz = magnet.get_field_vector_and_gradient(x[in_mag], z[in_mag])
                
                mu_mag = 0.5 * setup.g_s * setup.mu_b
                
                # F = mu * grad(B)
                force_z = spins[in_mag] * mu_mag * dBdz*-1
                force_x = spins[in_mag] * mu_mag * dBdx
                
                vz[in_mag] += (force_z / setup.mass) * dt
                vx[in_mag] += (force_x / setup.mass) * dt
        
        if step % 100 == 0:
            print(f"Flying: {len(x)} | Collected: {len(final_x)}", end='\r')

    return np.array(final_x), np.array(final_z), np.array(final_s)

if __name__ == "__main__":
    x_final, z_final, s_final = run_simulation_fixed()

    plt.figure(figsize=(10, 5), dpi=150)
    plt.scatter(x_final*1000, z_final*1000, c=s_final, cmap='coolwarm', s=0.5, alpha=0.6)
    plt.title(f"Stern-Gerlach Result ({len(x_final)} Atoms)")
    plt.xlabel("X (mm)")
    plt.ylabel("Z (mm)")
    plt.ylim(-1.5, 1.5)
    plt.xlim(-2.5, 2.5)
    plt.grid(True, alpha=0.3)
    plt.show()