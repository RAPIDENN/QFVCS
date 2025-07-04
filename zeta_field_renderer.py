import pygame
import numpy as np
from scipy.integrate import cumtrapz
import time
from numba import njit
import warnings

pygame.init()

WIDTH, HEIGHT = 400, 400 # Adjust as needed for performance vs detail
GRID_SIZE = 400 # Calculation resolution
BACKGROUND_COLOR = (0, 0, 0)

# A simplified color map for performance, assuming normalized magnitude [0, 1]
# Could add more steps or different colors based on preference
COLOR_MAP_ARRAY = np.array([
    [0, 0, 0],      # Black
    [0, 0, 255],    # Blue
    [0, 255, 0],    # Green
    [255, 255, 0],  # Yellow
    [255, 0, 0],    # Red
    [255, 255, 255] # White
], dtype=np.uint8)


# Numba-optimized core calculation function
@njit
def calculate_field_fast(x_flat, y_flat, t, params_scalar, params_array, r_array):
    """
    Calculates the field value at each point (x, y) for a given time t.
    Optimized for performance with Numba.
    Assumes parameters are scalars or numpy arrays passed directly.
    """
    # Unpack scalar parameters
    A0, D, eps, beta, alpha, omega0, chi3, t0, phi_val, rho_val, E_field_val, x0_val, y0_val, sigma_val = params_scalar

    # Derived scalar parameters (calculated at current time t)
    tau_t = t0 * np.exp(-(t / params_array[0])**2) # Assuming params_array[0] is t0
    I_t = np.abs(E_field_val)**2

    # Pre-calculate 2pi and pi * r outside loops
    two_pi = 6.283185307179586 # 2 * pi
    pi = 3.141592653589793

    num_points = x_flat.shape[0]
    field_values_flat = np.zeros(num_points, dtype=np.complex128)

    # This loop runs for each pixel/point
    for i in range(num_points):
        x = x_flat[i]
        y = y_flat[i]
        # Note: Original sqrt(x^2+y^2) might be 0; using np.sqrt requires float inputs, ensure x, y are floats.
        dist_xy_sq = x**2 + y**2
        if dist_xy_sq < 1e-9: # Avoid division by near zero
             # Handle center point - assign a high default magnitude to prevent NaNs.
             # This is a simplification, actual behavior at center needs proper model.
             r_term_A = tau_t + eps # Effectively treats center dist as 0 but adds tau_t and eps
        else:
            r_term_A = np.sqrt(dist_xy_sq) + tau_t + eps

        # Guard against very small r_term_A after adding offsets
        if r_term_A < 1e-9:
             fractal_amp = 1.0 # Assign max amp to avoid issues
        else:
             fractal_amp = A0 * (1.0 / r_term_A)**D * (1.0 + beta * I_t)**alpha

        phase_structure_prod = 1.0
        for rl in r_array:
             
             phase_arg = pi * rl * x
             phase_structure_prod *= np.cos(phase_arg)**2
        phase_structure = phase_structure_prod

        # Foc Subcicle (G)
        # Assuming x0_val, y0_val, sigma_val are scalar values for current time t
        Gauss = np.exp(-((x - x0_val)**2 + (y - y0_val)**2) / (sigma_val**2 + 1e-9)) # Add eps for stability


        # Onda Chirp de Atts (W)
        # Assuming phi_val is scalar value for current time t
        k_t = omega0 / 299792458.0 # Use speed of light in m/s for k(t) relation. Adjust units. Omega0 is freq in? assuming Hz here.
        # If omega0 is in eV (energy), convert to angular freq: omega0_rad = omega0_eV * 1.602e-19 / 6.582e-16
        omega0_rad = omega0 * 4.135667697e16 # Convert eV (approx 30eV -> THz/PHz range) to rad/s (adjust scale as needed)
        # Assuming phi_val is chirp parameter in rad/s^2 (adjust units!)
        wave_arg = (k_t * x - omega0_rad * t - 0.5 * phi_val * t**2)
        # Consider only the real part or split into cos/sin if complex not needed by consumer
        wave = np.exp(1j * wave_arg) # Return complex for fidelity

    
        # Using simplified E_field(t) and rho(t) as scalar values at time t.
        # np.gradient is temporal; calculating here inside spatial loop needs refactoring.
        # Let's use a simplified, non-derivative version for @njit, just dependent on scalar E and rho at time t.
        P_t_simplified = chi3 * E_field_val**3 # Simplified
        
        # Original formula: Z_n+1 = Z_n + A * F * G * W * P
        # Interpreting as: the complex field at (x,y,t) is proportional to A*F*G*W*P
        # Omitting Z_n for single frame rendering
        field_value = fractal_amp * phase_structure * Gauss * wave * P_t_simplified

        field_values_flat[i] = field_value

    # For visualization, return magnitude or phase, or real part.
    # Returning magnitude seems standard for visualizing field strength.
    return np.abs(field_values_flat)


# Numba-optimized color mapping function
@njit
def map_to_color_fast(normalized_intensity_flat, color_map_array):
    """Maps normalized intensity values [0, 1] to colors."""
    num_points = normalized_intensity_flat.shape[0]
    num_colors = color_map_array.shape[0]
    rgb_flat = np.zeros((num_points, 3), dtype=np.uint8)

    max_color_idx = num_colors - 1
    # Linear interpolation between color map steps
    float_indices = normalized_intensity_flat * max_color_idx
    left_indices = np.floor(float_indices).astype(np.int32)
    right_indices = left_indices + 1
    ratios = float_indices - left_indices

    # Clamp indices to valid range [0, max_color_idx]
    left_indices = np.maximum(0, np.minimum(max_color_idx, left_indices))
    right_indices = np.maximum(0, np.minimum(max_color_idx, right_indices))


    for i in range(num_points):
         # Interpolate color components (R, G, B)
         for c in range(3): # 3 color channels
            rgb_flat[i, c] = (color_map_array[left_indices[i], c] * (1.0 - ratios[i]) +
                              color_map_array[right_indices[i], c] * ratios[i])


    return rgb_flat


class ZetaRenderer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Zeta Phase Attosecond Field")
        self.clock = pygame.time.Clock()
        self.running = False

        # Physical and simulation parameters
        self.params = {
            'A0': 1.0,          # Amplitude scale factor
            'D': 0.5,           # Fractal dimension parameter (0.5-3.0 suggested)
            'eps': 0.01,        # Epsilon for smoothing denominator near zero
            'beta': 0.1,        # Nonlinearity coefficient (intensity coupling)
            'alpha': 0.8,       # Nonlinearity exponent
            'r': [0.1, 0.3, 0.5], # Parameters for structured phase (converted to np array)
            'omega0_eV': 30.0,  # Central frequency in eV
            'chi3': 1e-9,       # Third-order nonlinear susceptibility (adjust units!)
            'tau0': 100e-18,    # Base pulse duration (seconds, e.g., 100 attoseconds)
            't0': 50e-18,       # Pulse envelope width (seconds, e.g., 50 attoseconds)
            # Functions of time (implemented as lambda returning value at time t)
            'x0_func': lambda t: 0.5 + 0.1 * np.sin(t * 1e16), # Streaking/motion center in X [0,1]
            'y0_func': lambda t: 0.5 + 0.1 * np.cos(t * 1e16), # Streaking/motion center in Y [0,1]
            'sigma_func': lambda t: 0.1 + 0.05 * np.abs(np.sin(t * 1e16)), # Dynamic width [0,1]
            'phi_func': lambda t: 1e32 * np.sin(t * 1e16), # Nonlinear chirp (rad/s^2), scale adjusted for t in seconds
            'rho_func': lambda t: np.cos(t * 1e16)**2 # Electron density (simplified time dependence)
            # E_field function is defined below
        }

        # Convert parameters to types suitable for Numba ahead of time
        self.params['r_array'] = np.array(self.params['r'], dtype=np.float64) # Convert r list
        self.params['omega0_rad'] = self.params['omega0_eV'] * 1.60218e-19 / 6.58212e-16 # Convert eV to rad/s
        self.params['t0_s'] = self.params['t0'] # Alias t0 in seconds


        # Precompute coordinates grid flattened
        x_vals = np.linspace(0.0, 1.0, GRID_SIZE, dtype=np.float64)
        y_vals = np.linspace(0.0, 1.0, GRID_SIZE, dtype=np.float64)
        X, Y = np.meshgrid(x_vals, y_vals)
        self.x_flat = X.flatten()
        self.y_flat = Y.flatten()

        # Prepare scalar parameters for Numba (need to update these based on time in the loop)
        self._current_numba_params_scalar = np.zeros(14, dtype=np.float64) # A0, D, eps, beta, alpha, omega0, chi3, t0, phi_val, rho_val, E_field_val, x0_val, y0_val, sigma_val
        self._params_numba_array_static = np.array([self.params['t0_s']], dtype=np.float64) # Store parameters constant with time needed inside Numba func

        self.time_s = 0.0 # Simulation time in seconds
        self.time_scale = 1e-17 # How much simulation time advances per real-time tick (e.g., 1e-17 s/tick)

        # Store colormap as numpy array
        self.colormap_array = COLOR_MAP_ARRAY

    def E_field(self, t):
        """Simplified pulse electric field shape"""
        amplitude = 1e13 # Example amplitude V/m (adjust scale!)
        t0_s = self.params['t0_s']
        return amplitude * np.exp(-t**2 / (2 * t0_s**2)) * np.cos(self.params['omega0_rad'] * t) # Simple oscillating Gaussian envelope

    def update_numba_scalars(self, t):
         # Calculate time-dependent parameters
        phi_val = self.params['phi_func'](t)
        rho_val = self.params['rho_func'](t)
        E_field_val = self.E_field(t) # This needs to be passed *into* the numba function if calculated here

        # x0_val, y0_val, sigma_val for current time
        x0_val = self.params['x0_func'](t)
        y0_val = self.params['y0_func'](t)
        sigma_val = self.params['sigma_func'](t)


        # Update the numpy array of scalars for Numba
        self._current_numba_params_scalar[:] = (
            self.params['A0'], self.params['D'], self.params['eps'],
            self.params['beta'], self.params['alpha'], self.params['omega0_rad'],
            self.params['chi3'], self.params['tau0'],
            phi_val, rho_val, E_field_val, # Note: these are scalar values *at time t*
            x0_val, y0_val, sigma_val
        )


    def render_frame(self):
        """Generates and draws a single frame."""
        try:
            # Update time-dependent parameters needed by numba
            self.update_numba_scalars(self.time_s)

            # Calculate the complex field using the fast numba function
            magnitude_flat = calculate_field_fast(
                self.x_flat,
                self.y_flat,
                self.time_s, # Pass scalar time
                self._current_numba_params_scalar, # Pass array of scalar params
                self._params_numba_array_static, # Pass array of static/less frequent params
                self.params['r_array'] # Pass numpy array for r
            )

            # Reshape magnitude to 2D grid
            magnitude_2d = magnitude_flat.reshape(GRID_SIZE, GRID_SIZE)

            # Normalize magnitude to [0, 1] for color mapping
            min_mag = np.min(magnitude_2d)
            max_mag = np.max(magnitude_2d)

            if max_mag - min_mag > 1e-9: # Avoid division by near zero
                normalized_intensity_2d = (magnitude_2d - min_mag) / (max_mag - min_mag)
            else:
                 normalized_intensity_2d = np.zeros_like(magnitude_2d, dtype=np.float64)


            # Map normalized intensity to RGB colors using fast numba function
            normalized_intensity_flat = normalized_intensity_2d.flatten()
            rgb_flat = map_to_color_fast(normalized_intensity_flat, self.colormap_array)

            # Reshape RGB back to 2D grid (HEIGHT, WIDTH, 3)
            rgb_array = rgb_flat.reshape(GRID_SIZE, GRID_SIZE, 3)

            # Convert NumPy array to Pygame surface
            # Pygame surface is usually (width, height) for surfarray, numpy is (row, col) i.e. (height, width)
            # May need transpose: (GRID_SIZE, GRID_SIZE, 3) -> (GRID_SIZE, GRID_SIZE, 3) with x, y swapped
            surface = pygame.surfarray.make_surface(rgb_array) # Pygame surface expects (width, height, color)

            # Blit the surface to the screen, scaling it if GRID_SIZE != WIDTH/HEIGHT
            if GRID_SIZE != WIDTH or GRID_SIZE != HEIGHT:
                 scaled_surface = pygame.transform.scale(surface, (WIDTH, HEIGHT))
                 self.screen.blit(scaled_surface, (0, 0))
            else:
                 self.screen.blit(surface, (0, 0))

            # Update the full display surface
            pygame.display.flip()

        except Exception as e:
            warnings.warn(f"Rendering error: {e}")
            # Optional: Render error message on screen
            font = pygame.font.Font(None, 36)
            text = font.render("Rendering Error!", True, (255, 0, 0))
            self.screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
            pygame.display.flip()


    def run(self):
        """Runs the main Pygame loop."""
        self.running = True
        start_time_real = time.time() # Real time simulation starts

        while self.running:
            dt_real = self.clock.tick(60) / 1000.0 # Delta time in seconds (real time, e.g., 1/60 s)

            # Update simulation time (advance field visualization in attoseconds/femtoseconds scale)
            # Use the actual delta time for smoother time evolution independent of render lag
            self.time_s += dt_real * self.time_scale

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break

            # Render the current frame based on updated simulation time
            self.render_frame()

        # Cleanup
        pygame.quit()


if __name__ == '__main__':
    renderer = ZetaRenderer()
    renderer.run()
