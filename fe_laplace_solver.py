import numpy as np
import matplotlib.pyplot as plt

def solve_stern_gerlach_potential(grid_size=400, max_iter=5000):
    """
    Solves Laplace's Equation to generate the 'Eye' Magnet field.
    Returns: x_mm, z_mm, Bx, Bz, B_mag (in Tesla-ish units before scaling)
    """
    print(f"  [System] Running Finite Difference Solver ({grid_size}x{grid_size})...")
    
    # Grid in Millimeters
    lim = 6.0 
    x_range = np.linspace(-lim, lim, grid_size) 
    z_range = np.linspace(-lim, lim, grid_size)
    X, Z = np.meshgrid(x_range, z_range, indexing='ij') 
    
    Phi = np.zeros((grid_size, grid_size))
    
    # --- Geometry (Two-Wire / Wedge-Groove Approx) ---
    # Top Wedge (Sharper)
    slope = 1.0 / np.tan(np.radians(32))
    z_tip = 2.5 
    mask_top = Z > (slope * np.abs(X) + z_tip)
    
    # Bottom Block + Groove
    z_base = -0.3
    block_width = 3.0
    groove_width = 1.2
    
    mask_block = (Z < z_base) & (np.abs(X) < block_width)
    mask_groove = (np.abs(X) < groove_width) & (Z > -8.0)
    
    # Combine Bottom
    mask_bottom = mask_block & (~mask_groove)
    
    # Voltages
    V_top = 1000 
    V_bot = -1000
    Phi[mask_top] = V_top
    Phi[mask_bottom] = V_bot
    mask_fixed = mask_top | mask_bottom
    
    # Iterative Solution
    for i in range(max_iter):
        Phi[1:-1, 1:-1] = 0.25 * (Phi[1:-1, 0:-2] + Phi[1:-1, 2:] + 
                                  Phi[0:-2, 1:-1] + Phi[2:, 1:-1])
        Phi[mask_fixed] = (V_top * mask_top[mask_fixed]) + (V_bot * mask_bottom[mask_fixed])

    # Gradients to get B-field (Unscaled)
    # Gradient of Potential = Field
    dx = x_range[1] - x_range[0]
    dz = z_range[1] - z_range[0]
    
    dPhidx, dPhidz = np.gradient(Phi, dx, dz)
    Bx = -dPhidx
    Bz = -dPhidz
    B_mag = np.sqrt(Bx**2 + Bz**2)
    
    return x_range, z_range, Bx, Bz, B_mag

def plot_simulation_results():
    X, Z, Bx, Bz, B_mag, mask_fixed = solve_stern_gerlach_potential()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot Magnitude Heatmap
    # We mask the magnets themselves so they appear solid color (optional)
    B_mag_plot = np.ma.masked_where(mask_fixed, B_mag)
    
    # Use 'inferno' or 'viridis' to match the paper style
    mesh = ax.pcolormesh(X, Z, B_mag_plot, cmap='inferno', shading='auto', vmax=np.percentile(B_mag, 99))
    cbar = plt.colorbar(mesh, ax=ax, label='Relative Magnetic Field Strength |B|')
    
    # Plot Field Lines (Streamlines)
    # We start points near the top magnet to trace them down
    st = ax.streamplot(X, Z, Bx, Bz, color='cyan', linewidth=0.6, 
                       density=1.5, arrowstyle='->', arrowsize=1.0)
    
    # Overlay the Magnet Shapes (Grey areas)
    # We use contourf to draw the solid magnets on top
    ax.contourf(X, Z, mask_fixed, levels=[0.5, 1.5], colors=['#A0A0A0'])
    
    ax.set_title("Stern-Gerlach Field Simulation (Finite Difference Laplace)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_simulation_results()