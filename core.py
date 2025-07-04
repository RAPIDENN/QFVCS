# Copyright (C) 2024 Nathaniel (RAPIDENN)
# Part of QFVCS - Licensed under GPL-3.0
# See LICENSE and core.py for details

"""
QFVCS
-------------------------------
"""

import sys
import numpy as np
import threading
import time
import colorsys
import psutil
import os
from datetime import datetime

# Conditional imports for GPU support - completely optional
HAS_CUPY = False
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None
    print("CuPy not available - using CPU only mode")

# PyQt5 for UI - ensure all needed widgets are imported
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QGridLayout,
                            QVBoxLayout, QSlider, QPushButton, QLabel, QTabWidget, QComboBox,
                            QGroupBox, QCheckBox, QSplitter, QRadioButton, QButtonGroup,
                            QMessageBox, QFileDialog, QOpenGLWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QMetaObject, Q_ARG
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPolygonF

# OpenGL for 3D visualization - make imports robust to handle missing dependencies
HAS_OPENGL = False
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.error import GLError
    
    # Test if OpenGL actually works
    if callable(glGenBuffers) and callable(glDrawArrays):
        HAS_OPENGL = True
    else:
        print("Warning: OpenGL functions not properly initialized.")
except ImportError:
    print("Warning: OpenGL libraries not found. Running in fallback mode.")

# Define dummy OpenGL functions if OpenGL not available
if not HAS_OPENGL:
    # Define dummy OpenGL functions
    def glEnable(cap): pass
    def glDisable(cap): pass
    def glClearColor(r, g, b, a): pass
    def glClear(mask): pass
    def glLoadIdentity(): pass
    def glTranslatef(x, y, z): pass
    def glRotatef(angle, x, y, z): pass
    def glViewport(x, y, width, height): pass
    def glMatrixMode(mode): pass
    def glGenBuffers(n): return [0] * n
    def glBindBuffer(target, buffer): pass
    def glBufferData(target, size, data, usage): pass
    def glEnableClientState(array): pass
    def glDisableClientState(array): pass
    def glVertexPointer(size, type, stride, pointer): pass
    def glColorPointer(size, type, stride, pointer): pass
    def glDrawArrays(mode, first, count): pass
    def glDeleteBuffers(n, buffers): pass
    def glBlendFunc(sfactor, dfactor): pass
    def glPointSize(size): pass
    def glLineWidth(width): pass
    def gluPerspective(fovy, aspect, zNear, zFar): pass
    def glIsBuffer(buffer): return False
    def glDrawElements(mode, count, type, indices): pass
    def glNormalPointer(type, stride, pointer): pass
    def glMaterialfv(face, pname, params): pass
    def glMaterialf(face, pname, param): pass
    def glLightfv(light, pname, params): pass
    def glBegin(mode): pass
    def glEnd(): pass
    def glVertex3f(x, y, z): pass
    def glColor3f(r, g, b): pass
    
    # Define dummy OpenGL constants
    GL_TRIANGLES = 4
    GL_LINES = 1
    GL_LINE_STRIP = 3
    GL_POINTS = 0
    GL_UNSIGNED_INT = 5125
    GL_FLOAT = 5126
    GL_COLOR_BUFFER_BIT = 16384
    GL_DEPTH_BUFFER_BIT = 256
    GL_LIGHTING = 2896
    GL_LIGHT0 = 16384
    GL_POSITION = 4611
    GL_COLOR_MATERIAL = 2903
    GL_DEPTH_TEST = 2929
    GL_POINT_SMOOTH = 2832
    GL_BLEND = 3042
    GL_SRC_ALPHA = 770
    GL_ONE_MINUS_SRC_ALPHA = 771
    GL_ONE = 1
    GL_ARRAY_BUFFER = 34962
    GL_ELEMENT_ARRAY_BUFFER = 34963
    GL_STATIC_DRAW = 35044
    GL_STREAM_DRAW = 35040
    GL_VERTEX_ARRAY = 32884
    GL_COLOR_ARRAY = 32886
    GL_NORMAL_ARRAY = 32885
    GL_FRONT_AND_BACK = 1032
    GL_AMBIENT = 4608
    GL_DIFFUSE = 4609
    GL_SPECULAR = 4610
    GL_SHININESS = 5633
    GL_PROJECTION = 5889
    GL_MODELVIEW = 5888
    GL_FRONT = 1028
    GL_BACK = 1029
    GL_FILL = 6914
    GL_POLYGON_MODE = 2880
    
    class GLError(Exception):
        pass

# SciPy for FFT operations when not using GPU
from scipy.fft import fftn, ifftn

# Matplotlib for 2D plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Optional: Import skimage for marching cubes algorithm if available
try:
    from skimage.measure import marching_cubes
    HAS_MARCHING_CUBES = True
except ImportError:
    HAS_MARCHING_CUBES = False
    print("Warning: scikit-image not found, isosurface visualization will be limited.")
    # Create dummy marching_cubes function to prevent crashes
    def marching_cubes(volume, level):
        # Return empty arrays with correct shapes
        n = max(1, min(len(volume), 10))  # Small number of vertices
        return (
            np.zeros((n, 3)),  # vertices
            np.zeros((n-2, 3), dtype=np.int32),  # faces
            np.zeros((n, 3)),  # normals
            np.zeros(n)  # values
        )


#############################################################################
# PART 1: Quantum Wave System with Zeta Formula Support
#############################################################################

class QuantumZetaSystem:
    """
    Core simulation system supporting both quantum physics (Schrödinger equation)
    and Zeta fractal formula visualization.
    """
    def __init__(self, use_gpu=False, config=None):
        # Configure GPU usage - force to False if CuPy not available
        self.use_gpu = use_gpu and HAS_CUPY
        
        # If use_gpu is True but CuPy is not available, print a warning
        if use_gpu and not HAS_CUPY:
            print("Warning: GPU requested but CuPy not available. Using CPU instead.")
        
        # Set appropriate calculation module and FFT functions
        if self.use_gpu:
            self.xp = cp
            self.fft = cp.fft.fftn
            self.ifft = cp.fft.ifftn
        else:
            self.xp = np
            # Import SciPy FFT here to avoid unnecessary import if GPU is used
            from scipy.fft import fftn, ifftn
            self.fft = fftn
            self.ifft = ifftn
        
        # Thread safety
        self.lock = threading.Lock()
        
        # System mode
        self.mode = 'schrodinger'  # 'schrodinger' or 'zeta'
        
        # Initialize parameters with defaults
        self.params = {
            # Simulation parameters
            'N': 32,               # Grid resolution (default 32³)
            'dx': 0.1e-9,          # Spatial step (0.1 nm)
            'dt': 1e-18,           # Time step (1 attosecond)
            'time': 0.0,           # Current simulation time
            
            # Physical constants
            'hbar': 1.0545718e-34, # Reduced Planck constant (J·s)
            'm': 9.1093837e-31,    # Electron mass (kg)
            
            # Quantum wave parameters
            'potential_type': 'harmonic',    # Type of potential
            'potential_height': 1.602e-19,   # 1 eV in J
            'sigma': 1e-9,                   # Wave packet width (1 nm)
            'k0': [1e9, 0, 0, 0],            # Initial momentum
            
            # Zeta fractal parameters
            's': 1.25,             # s parameter (fractal dimension D)
            'freq_scale': 1.0,     # Frequency scaling (ω')
            'amp_mod': 1.0,        # Amplitude modulation (A_0)
            'N_modes': 50,         # Number of modes (N)
            'r': 0.5,              # Fractal scaling factor (r)
            'epsilon': 0.01,       # Small value to avoid division by zero (ε)
            
            # Visualization parameters
            'dimensions': 3,       # Number of dimensions (3 or 4)
        }
        
        # Override with provided config if available
        if config:
            if isinstance(config, dict):
                self.params.update(config)
            elif isinstance(config, str):
                # Assume config is a file path
                try:
                    import json
                    with open(config, 'r') as f:
                        self.params.update(json.load(f))
                except Exception as e:
                    print(f"Error loading config file: {e}")
        
        # Internal state
        self.psi = None            # Complex wave function
        self.psi_k = None          # Wave function in momentum space
        self.V = None              # Potential energy function
        self.grid = None           # Spatial grid
        self.k_grid = None         # Momentum grid
        
        # Performance monitoring
        self.performance_stats = {
            'last_update_time': time.time(),
            'update_count': 0,
            'avg_update_time': 0,
            'max_update_time': 0,
            'memory_usage': 0
        }
        
        # Initialize simulation
        self.initialize()
    
    def initialize(self):
        """Initialize the quantum system with grid and wave function"""
        with self.lock:
            # Set dimensions and grid size
            N = self.params['N']
            dx = self.params['dx']
            dimensions = self.params['dimensions']
            
            # Create spatial and momentum grids based on dimensions
            if dimensions == 3:
                # 3D grid initialization
                x = self.xp.linspace(-N//2 * dx, N//2 * dx, N)
                self.grid = self.xp.meshgrid(x, x, x, indexing='ij')
                
                # Initialize wave function based on mode
                if self.mode == 'zeta':
                    self.initialize_zeta_quantum()
                else:
                    self.initialize_gaussian_3d()
                
                # Create potential function
                self.V = self.create_potential_3d()
                
                # Initialize momentum grid
                k = 2 * self.xp.pi * self.xp.fft.fftfreq(N, dx)
                KX, KY, KZ = self.xp.meshgrid(k, k, k, indexing='ij')
                self.k_grid = KX**2 + KY**2 + KZ**2
                
            elif dimensions == 4:
                # 4D grid initialization (limited resolution for performance)
                n4d = min(N, 16)  # Limit 4D resolution
                x = self.xp.linspace(-n4d//2 * dx, n4d//2 * dx, n4d)
                self.grid = self.xp.meshgrid(x, x, x, x, indexing='ij')
                
                # Initialize based on mode
                if self.mode == 'zeta':
                    self.initialize_zeta_quantum_4d()
                else:
                    self.initialize_gaussian_4d()
                
                # Create 4D potential
                self.V = self.create_potential_4d()
                
                # Initialize 4D momentum grid
                k = 2 * self.xp.pi * self.xp.fft.fftfreq(n4d, dx)
                KX, KY, KZ, KW = self.xp.meshgrid(k, k, k, k, indexing='ij')
                self.k_grid = KX**2 + KY**2 + KZ**2 + KW**2
            
            # Calculate memory usage
            memory_bytes = 0
            if self.psi is not None:
                memory_bytes += self.psi.nbytes
            if self.V is not None:
                memory_bytes += self.V.nbytes
            if self.k_grid is not None:
                memory_bytes += self.k_grid.nbytes
            
            self.performance_stats['memory_usage'] = memory_bytes / 1024 / 1024  # MB
    
    def initialize_gaussian_3d(self):
        """Initialize a 3D Gaussian wave packet with specified parameters"""
        N = self.params['N']
        sigma = self.params['sigma']
        k0 = self.params['k0'][:3]  # Use first 3 components for 3D
        X, Y, Z = self.grid
        
        # Calculate Gaussian wave packet with normalization
        norm_const = (1/(2*self.xp.pi*sigma**2))**0.75
        r_squared = X**2 + Y**2 + Z**2
        envelope = norm_const * self.xp.exp(-r_squared/(4*sigma**2))
        phase = self.xp.exp(1j * (k0[0]*X + k0[1]*Y + k0[2]*Z))
        
        self.psi = envelope * phase
        self.normalize()
    
    def initialize_gaussian_4d(self):
        """Initialize a 4D Gaussian wave packet"""
        sigma = self.params['sigma']
        k0 = self.params['k0']  # Use all 4 components
        X, Y, Z, W = self.grid
        
        # 4D Gaussian with normalization
        norm_const = (1/(2*self.xp.pi*sigma**2))**1  # 4D normalization
        r_squared = X**2 + Y**2 + Z**2 + W**2
        envelope = norm_const * self.xp.exp(-r_squared/(4*sigma**2))
        phase = self.xp.exp(1j * (k0[0]*X + k0[1]*Y + k0[2]*Z + k0[3]*W))
        
        self.psi = envelope * phase
        self.normalize()
    
    def initialize_zeta_quantum(self):
        """Initialize a quantum state based on Zeta Fractal Formula:
        Z_{n+1} = Z_n + [A_0 ⋅ (∏cos²(π⋅r^l⋅x)) ⋅ (1/√(x²+y²)+t+ε)^D ⋅ exp(-(r-r_0)²/σ²) ⋅ exp(i(k'x-ω't))]
        """
        # Get parameters
        A_0 = self.params['amp_mod']
        D = self.params['s']
        N_modes = int(self.params['N_modes'])
        r = self.params['r']
        epsilon = self.params['epsilon']
        freq_scale = self.params['freq_scale']
        sigma = self.params['sigma']
        time = self.params['time']
        k0 = self.params['k0'][0]  # Use first component for wave number
        
        X, Y, Z = self.grid
        
        # Center coordinates
        x_0, y_0, z_0 = 0, 0, 0
        
        # Compute fractal product term for phi(x) - the core fractal pattern
        phi_x = self.xp.ones_like(X)
        for l in range(N_modes):
            phi_x *= self.xp.cos(self.xp.pi * r**l * X)**2
            
        # Calculate R with epsilon to avoid division by zero
        R = self.xp.sqrt(X**2 + Y**2 + epsilon)
        
        # Fractal amplitude term
        A = A_0 * (1.0 / (R + time + epsilon))**D
        
        # Focusing term (Gaussian envelope)
        focus = self.xp.exp(-((X - x_0)**2 + (Y - y_0)**2 + (Z - z_0)**2) / sigma**2)
        
        # Dynamic phase with frequency scaling
        omega = freq_scale * 2.0 * self.xp.pi
        phase = phi_x * omega * time
        
        # Calculate zeta wave function: psi = A * focus * exp(i * phase)
        self.psi = A * focus * self.xp.exp(1j * (k0 * X - omega * time + phase))
        
        # Normalize for quantum dynamics
        self.normalize()
    
    def initialize_zeta_quantum_4d(self):
        """Initialize a 4D quantum state based on Zeta Fractal Formula with 4D extension"""
        # Get parameters
        A_0 = self.params['amp_mod']
        D = self.params['s']
        N_modes = int(self.params['N_modes'])
        r = self.params['r']
        epsilon = self.params['epsilon']
        freq_scale = self.params['freq_scale']
        sigma = self.params['sigma']
        time = self.params['time']
        k0 = self.params['k0'][0]  # Wave number
        
        X, Y, Z, W = self.grid
        
        # Center coordinates in 4D
        x_0, y_0, z_0, w_0 = 0, 0, 0, 0
        
        # Compute fractal product terms for each dimension
        phi_x = self.xp.ones_like(X)
        phi_y = self.xp.ones_like(Y)
        phi_z = self.xp.ones_like(Z)
        phi_w = self.xp.ones_like(W)
        
        for l in range(min(N_modes, 8)):  # Limit iterations for 4D performance
            phi_x *= self.xp.cos(self.xp.pi * r**l * X)**2
            phi_y *= self.xp.cos(self.xp.pi * r**l * Y)**2
            phi_z *= self.xp.cos(self.xp.pi * r**l * Z)**2
            phi_w *= self.xp.cos(self.xp.pi * r**l * W)**2
        
        # 4D radius with epsilon
        R = self.xp.sqrt(X**2 + Y**2 + Z**2 + W**2 + epsilon)
        
        # 4D fractal amplitude
        A = A_0 * (1.0 / (R + time + epsilon))**D
        
        # 4D focusing term
        focus = self.xp.exp(-((X - x_0)**2 + (Y - y_0)**2 + (Z - z_0)**2 + (W - w_0)**2) / sigma**2)
        
        # Dynamic phase
        omega = freq_scale * 2.0 * self.xp.pi
        phase = phi_x * phi_y * phi_z * phi_w
        
        # Calculate 4D zeta wave function
        self.psi = A * focus * self.xp.exp(1j * (k0 * (X + Y + Z + W) - omega * time + phase))
        
        # Normalize
        self.normalize()
    
    def create_potential_3d(self):
        """Create the potential energy function in 3D space"""
        potential_type = self.params['potential_type']
        X, Y, Z = self.grid
        V0 = self.params['potential_height']  # Potential height/depth
        
        if potential_type == 'free':
            # Free particle (zero potential)
            return self.xp.zeros_like(X)
        
        elif potential_type == 'harmonic':
            # 3D harmonic oscillator: V = 0.5 * k * r²
            k = 1e3  # Spring constant
            return 0.5 * k * (X**2 + Y**2 + Z**2)
        
        elif potential_type == 'box':
            # Infinite potential well (box)
            box_size = 5e-9  # 5 nm box
            wall_height = 1e20  # Very high energy
            
            return self.xp.where(
                (self.xp.abs(X) > box_size) | 
                (self.xp.abs(Y) > box_size) | 
                (self.xp.abs(Z) > box_size), 
                wall_height, 0)
        
        elif potential_type == 'barrier':
            # Potential barrier for tunneling demonstration
            barrier_width = 1e-9  # 1 nm barrier width
            return self.xp.where(self.xp.abs(X) < barrier_width/2, V0, 0)
        
        elif potential_type == 'coulomb':
            # Coulomb potential (e.g., hydrogen atom): V = -k/r
            r = self.xp.sqrt(X**2 + Y**2 + Z**2 + 1e-30)  # Avoid division by zero
            return -V0 / r
        
        else:
            # Default to zero potential
            return self.xp.zeros_like(X)
    
    def create_potential_4d(self):
        """Create the potential energy function in 4D space"""
        potential_type = self.params['potential_type']
        X, Y, Z, W = self.grid
        V0 = self.params['potential_height']
        
        if potential_type == 'harmonic':
            # 4D harmonic oscillator
            k = 1e3  # Spring constant
            return 0.5 * k * (X**2 + Y**2 + Z**2 + W**2)
        
        elif potential_type == 'box':
            # 4D box potential
            box_size = 5e-9
            wall_height = 1e20
            
            return self.xp.where(
                (self.xp.abs(X) > box_size) | 
                (self.xp.abs(Y) > box_size) | 
                (self.xp.abs(Z) > box_size) |
                (self.xp.abs(W) > box_size), 
                wall_height, 0)
        
        else:
            # Default to zero potential in 4D
            return self.xp.zeros_like(X)
    
    def normalize(self):
        """Normalize the wave function to ensure probability conservation"""
        if self.psi is None:
            return
        
        # Calculate the norm (sqrt of probability integral)
        probability = self.xp.abs(self.psi)**2
        
        # Use appropriate volume element based on dimensions
        if self.params['dimensions'] == 3:
            dx = self.params['dx']
            norm = self.xp.sqrt(self.xp.sum(probability) * dx**3)
        else:  # 4D
            dx = self.params['dx']
            norm = self.xp.sqrt(self.xp.sum(probability) * dx**4)
        
        if norm > 0:
            # Normalize by dividing by the norm
            self.psi /= norm
    
    def update(self, dt=None):
        """Advance the quantum system one time step"""
        start_time = time.time()
        
        with self.lock:
            # Use default dt if not specified
            if dt is None:
                dt = self.params['dt']
            
            # Update time
            self.params['time'] += dt
            
            try:
                if self.mode == 'zeta':
                    # For zeta mode, we regenerate the zeta function at each step
                    if self.params['dimensions'] == 3:
                        self.initialize_zeta_quantum()
                    else:  # 4D
                        self.initialize_zeta_quantum_4d()
                else:
                    # Apply split-operator method for Schrödinger equation
                    if self.params['dimensions'] == 3:
                        self.split_operator_step_3d(dt)
                    else:  # 4D
                        self.split_operator_step_4d(dt)
                
                # Track performance
                update_time = time.time() - start_time
                self.performance_stats['update_count'] += 1
                self.performance_stats['max_update_time'] = max(
                    self.performance_stats['max_update_time'], 
                    update_time)
                
                # Update average time with exponential moving average
                if self.performance_stats['avg_update_time'] == 0:
                    self.performance_stats['avg_update_time'] = update_time
                else:
                    self.performance_stats['avg_update_time'] = (
                        0.9 * self.performance_stats['avg_update_time'] + 
                        0.1 * update_time)
                
                # Return CPU version of wave function if using GPU
                if self.use_gpu:
                    return cp.asnumpy(self.psi)
                else:
                    return self.psi
                
            except Exception as e:
                print(f"Error in quantum simulation update: {e}")
                # Ensure the wave function remains normalized even after error
                try:
                    self.normalize()
                except:
                    pass
                # Handle failures gracefully
                return None
    
    def split_operator_step_3d(self, dt):
        """Perform one time step using the split-operator method in 3D"""
        # Get parameters
        hbar = self.params['hbar']
        m = self.params['m']
        
        # First half-step in position space (potential term)
        self.psi *= self.xp.exp(-1j * self.V * dt / (2 * hbar))
        
        # Transform to momentum space
        psi_k = self.fft(self.psi)
        
        # Full step in momentum space (kinetic term)
        psi_k *= self.xp.exp(-1j * hbar * self.k_grid * dt / (2 * m))
        
        # Transform back to position space
        self.psi = self.ifft(psi_k)
        
        # Second half-step in position space (potential term)
        self.psi *= self.xp.exp(-1j * self.V * dt / (2 * hbar))
        
        # Normalize to ensure conservation of probability
        self.normalize()
    
    def split_operator_step_4d(self, dt):
        """Perform one time step using the split-operator method in 4D"""
        # Similar to 3D, but with 4D wave function and operators
        hbar = self.params['hbar']
        m = self.params['m']
        
        # First half-step in position space
        self.psi *= self.xp.exp(-1j * self.V * dt / (2 * hbar))
        
        # Transform to momentum space
        psi_k = self.fft(self.psi)
        
        # Full step in momentum space
        psi_k *= self.xp.exp(-1j * hbar * self.k_grid * dt / (2 * m))
        
        # Transform back to position space
        self.psi = self.ifft(psi_k)
        
        # Second half-step in position space
        self.psi *= self.xp.exp(-1j * self.V * dt / (2 * hbar))
        
        # Normalize
        self.normalize()
    
    def probability_density(self):
        """Calculate the probability density |ψ|²"""
        if self.psi is None:
            return None
        
        # Return CPU version if using GPU
        if self.use_gpu:
            return cp.asnumpy(self.xp.abs(self.psi)**2)
        else:
            return self.xp.abs(self.psi)**2
    
    def get_phase(self):
        """Get the phase of the wave function"""
        if self.psi is None:
            return None
            
        # Return CPU version if using GPU
        if self.use_gpu:
            return cp.asnumpy(self.xp.angle(self.psi))
        else:
            return self.xp.angle(self.psi)
    
    def export_state(self, file_path):
        """Export the current quantum state to a file"""
        if self.psi is None:
            return False
        
        try:
            # Convert to CPU arrays if necessary
            if self.use_gpu:
                psi_real = cp.asnumpy(self.psi.real)
                psi_imag = cp.asnumpy(self.psi.imag)
                V = cp.asnumpy(self.V)
            else:
                psi_real = self.psi.real
                psi_imag = self.psi.imag
                V = self.V
            
            # Save using numpy
            np.savez(file_path,
                    psi_real=psi_real,
                    psi_imag=psi_imag,
                    V=V,
                    mode=self.mode,
                    time=self.params['time'],
                    params=self.params)
            return True
        except Exception as e:
            print(f"Error exporting quantum state: {e}")
            return False
    
    def import_state(self, file_path):
        """Import a quantum state from a file"""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Move data to GPU if necessary
            if self.use_gpu:
                self.psi = cp.array(data['psi_real'] + 1j * data['psi_imag'])
                self.V = cp.array(data['V'])
            else:
                self.psi = data['psi_real'] + 1j * data['psi_imag']
                self.V = data['V']
            
            # Set mode
            if 'mode' in data:
                self.mode = str(data['mode'])
            
            # Set time
            if 'time' in data:
                self.params['time'] = float(data['time'])
            
            # Update parameters with loaded ones
            if 'params' in data:
                loaded_params = data['params'].item()
                if isinstance(loaded_params, dict):
                    self.params.update(loaded_params)
            
            # Reinitialize grids based on the loaded wave function
            self.initialize()
            
            return True
        except Exception as e:
            print(f"Error importing quantum state: {e}")
            return False


#############################################################################
# PART 2: Enhanced 3D Quantum Visualizer
#############################################################################

class QuantumVisualizerWidget(QOpenGLWidget):
    """
    OpenGL Widget for visualizing 3D quantum wave functions.
    Supports both isosurface and particle visualization methods.
    """
    def __init__(self, quantum_system, parent=None):
        try:
            super(QuantumVisualizerWidget, self).__init__(parent)
        except Exception as e:
            print(f"Error initializing QuantumVisualizerWidget: {e}")
            # This might happen if QOpenGLWidget is not properly initialized
        self.quantum_system = quantum_system
        
        # Visualization parameters
        self.visualization_type = "particles"  # "particles" or "isosurface"
        self.num_particles = 1000
        self.point_size = 5.0
        self.glow_effect = True
        self.color_by_phase = True
        self.trail_effect = True
        self.max_trail_length = 5
        self.isosurface_threshold = 0.3
        
        # Rendering state
        self.particles = []
        self.particle_trails = []
        self.vertices = None
        self.colors = None
        self.vbo = None
        self.color_vbo = None
        self.isosurface_vertices = None
        self.isosurface_faces = None
        self.isosurface_normals = None
        self.isosurface_vbo = None
        self.isosurface_cached = False
        self.last_iso_threshold = 0.0
        
        # View parameters
        self.x_rot = 0
        self.y_rot = 0
        self.z_rot = 0
        self.z_trans = -15.0
        self.last_pos = None
        
        # Performance tracking
        self.performance_stats = {
            'render_time': 0.0,
            'update_time': 0.0,
            'avg_render_time': 0.0,
            'avg_update_time': 0.0
        }
    
    def __del__(self):
        self.cleanup()
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if not HAS_OPENGL:
            return

        try:
            if hasattr(self, 'vbo') and self.vbo is not None and callable(glIsBuffer) and glIsBuffer(self.vbo):
                if callable(glDeleteBuffers):
                    glDeleteBuffers(1, [self.vbo])
                self.vbo = None
                
            if hasattr(self, 'color_vbo') and self.color_vbo is not None and callable(glIsBuffer) and glIsBuffer(self.color_vbo):
                if callable(glDeleteBuffers):
                    glDeleteBuffers(1, [self.color_vbo])
                self.color_vbo = None
                
            if hasattr(self, 'isosurface_vbo') and self.isosurface_vbo is not None and callable(glIsBuffer) and glIsBuffer(self.isosurface_vbo):
                if callable(glDeleteBuffers):
                    glDeleteBuffers(1, [self.isosurface_vbo])
                self.isosurface_vbo = None
        except GLError as e:
            print(f"OpenGL error during cleanup: {e}")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def initialize_particles(self):
        """Initialize particles based on quantum wave function probability density"""
        start_time = time.time()
        
        try:
            # Get probability density from quantum system
            prob = self.quantum_system.probability_density()
            if prob is None:
                return
            
            # Get phase if needed
            phase = self.quantum_system.get_phase() if self.color_by_phase else None
            
            # Flatten probability and normalize for sampling
            prob_flat = prob.flatten()
            if np.sum(prob_flat) == 0:
                return
                
            prob_flat = prob_flat / np.sum(prob_flat)
            
            # Sample particle positions from probability distribution
            N = prob.shape[0]  # Assuming cubic grid for simplicity
            dx = self.quantum_system.params['dx']
            
            # Sample indices based on probability
            indices = np.random.choice(N**3, size=self.num_particles, p=prob_flat)
            coords = np.unravel_index(indices, (N, N, N))
            
            # Convert indices to physical positions
            x = (coords[0] - N//2) * dx * 1e9  # Convert to nm for visualization
            y = (coords[1] - N//2) * dx * 1e9
            z = (coords[2] - N//2) * dx * 1e9
            
            # Get phase information for coloring (optional)
            phases = None
            if self.color_by_phase and phase is not None:
                phases_flat = phase.flatten()
                phases = [phases_flat[i] for i in indices]
            
            # Create or update particle list
            if not self.particles:
                self.particles = []
                self.particle_trails = []
                
                for i in range(self.num_particles):
                    intensity = prob_flat[indices[i]] / np.max(prob_flat)
                    particle_phase = phases[i] if phases is not None else 0
                    
                    color = self.probability_to_color(intensity, particle_phase)
                    
                    self.particles.append({
                        'pos': [x[i], y[i], z[i]],
                        'color': color,
                        'intensity': intensity,
                        'size': self.point_size * (0.5 + 1.5 * intensity)
                    })
                    
                    self.particle_trails.append([])
            else:
                # Update existing particles
                for i in range(min(self.num_particles, len(self.particles))):
                    # Add current position to trail
                    if self.trail_effect and len(self.particle_trails[i]) < self.max_trail_length:
                        self.particle_trails[i].append(self.particles[i]['pos'].copy())
                    elif self.trail_effect:
                        self.particle_trails[i].pop(0)
                        self.particle_trails[i].append(self.particles[i]['pos'].copy())
                    
                    # Update position and color
                    intensity = prob_flat[indices[i]] / np.max(prob_flat)
                    particle_phase = phases[i] if phases is not None else 0
                    
                    color = self.probability_to_color(intensity, particle_phase)
                    
                    self.particles[i]['pos'] = [x[i], y[i], z[i]]
                    self.particles[i]['color'] = color
                    self.particles[i]['intensity'] = intensity
                    self.particles[i]['size'] = self.point_size * (0.5 + 1.5 * intensity)
            
            # Update vertex arrays for rendering
            self.update_vertex_arrays()
            
            update_time = time.time() - start_time
            self.performance_stats['update_time'] = update_time
            self.performance_stats['avg_update_time'] = (
                0.9 * self.performance_stats['avg_update_time'] + 
                0.1 * update_time)
                
        except Exception as e:
            print(f"Error initializing particles: {e}")
    
    def compute_isosurface(self):
        """Compute isosurface using marching cubes algorithm"""
        start_time = time.time()
        
        # Skip recalculation if threshold hasn't changed significantly
        if (self.isosurface_cached and 
            abs(self.last_iso_threshold - self.isosurface_threshold) < 0.01):
            return
            
        if not HAS_MARCHING_CUBES:
            print("Warning: scikit-image not available for marching cubes.")
            return
            
        try:
            # Get probability density
            prob = self.quantum_system.probability_density()
            if prob is None or not np.any(prob):
                print("Warning: No valid probability density data for isosurface")
                return
                
            # Check for valid data range
            prob_min, prob_max = np.min(prob), np.max(prob)
            if prob_min == prob_max:
                print("Warning: Flat probability distribution, skipping isosurface")
                return
                
            # Calculate threshold based on probability max
            threshold = self.isosurface_threshold * np.max(prob)
            if threshold <= prob_min or threshold >= prob_max:
                threshold = prob_min + 0.5 * (prob_max - prob_min)
                print(f"Adjusting threshold to valid range: {threshold}")
            
            # Apply marching cubes algorithm
            try:
                verts, faces, normals, _ = marching_cubes(prob, threshold)
                
                # Validate output
                if len(verts) == 0 or len(faces) == 0:
                    print("Warning: Marching cubes produced empty mesh")
                    return
                    
                # Scale vertices to match the coordinate system
                N = prob.shape[0]
                dx = self.quantum_system.params['dx']
                verts = (verts - N/2) * dx * 1e9  # Convert to nm
                
                self.isosurface_vertices = verts
                self.isosurface_faces = faces
                self.isosurface_normals = normals
                self.isosurface_cached = True
                self.last_iso_threshold = self.isosurface_threshold
                
            except ValueError as ve:
                print(f"Error in marching_cubes: {ve}")
                return
                
            update_time = time.time() - start_time
            self.performance_stats['update_time'] = update_time
            
        except Exception as e:
            print(f"Error computing isosurface: {e}")
            import traceback
            traceback.print_exc()
    
    def probability_to_color(self, intensity, phase=0):
        """Convert probability and phase to a color"""
        if self.color_by_phase:
            # Use phase for hue, intensity for value
            hue = (phase + np.pi) / (2 * np.pi)  # Map [-pi, pi] to [0, 1]
            saturation = 0.8
            value = 0.4 + 0.6 * intensity
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            return [r, g, b, 0.2 + 0.8 * intensity]
        else:
            # Use colormap based on intensity
            rgba = cm.viridis(intensity)
            return [rgba[0], rgba[1], rgba[2], 0.2 + 0.8 * intensity]
    
    def update_vertex_arrays(self):
        """Update vertex and color arrays for rendering"""
        try:
            # Build arrays for particles
            vertices = []
            colors = []
            
            # Add trail vertices first
            if self.trail_effect:
                for i, trail in enumerate(self.particle_trails):
                    for pos in trail:
                        vertices.append(pos)
                        
                        # Trail color - fade based on position in trail
                        base_color = self.particles[i]['color'].copy()
                        alpha = base_color[3] * 0.4  # Reduced alpha for trails
                        colors.append([base_color[0], base_color[1], base_color[2], alpha])
            
            # Add particle vertices
            for p in self.particles:
                vertices.append(p['pos'])
                colors.append(p['color'])
            
            # Convert to numpy arrays
            self.vertices = np.array(vertices, dtype=np.float32)
            self.colors = np.array(colors, dtype=np.float32)
        except Exception as e:
            print(f"Error updating vertex arrays: {e}")
            self.vertices = np.zeros((1, 3), dtype=np.float32)
            self.colors = np.zeros((1, 4), dtype=np.float32)
    
    def update_vbo(self):
        """Update OpenGL vertex buffer objects"""
        if not HAS_OPENGL:
            return
            
        try:
            # Check if OpenGL functions are available
            if not callable(glGenBuffers) or not callable(glBindBuffer) or not callable(glBufferData):
                print("Warning: OpenGL functions not available")
                return
                
            # Update particle VBOs
            if not hasattr(self, 'vbo') or self.vbo is None:
                self.vbo = glGenBuffers(1)
                if not self.vbo:
                    print("Warning: Failed to generate VBO")
                    return
                
            if not hasattr(self, 'color_vbo') or self.color_vbo is None:
                self.color_vbo = glGenBuffers(1)
                if not self.color_vbo:
                    print("Warning: Failed to generate color VBO")
                    return
            
            if self.vertices is not None and len(self.vertices) > 0:
                # Update position VBO
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
                # Check vertices data type and convert if needed
                if not isinstance(self.vertices, np.ndarray) or self.vertices.dtype != np.float32:
                    self.vertices = np.array(self.vertices, dtype=np.float32)
                glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STREAM_DRAW)
                
                # Update color VBO
                glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
                # Check colors data type and convert if needed
                if not isinstance(self.colors, np.ndarray) or self.colors.dtype != np.float32:
                    self.colors = np.array(self.colors, dtype=np.float32)
                glBufferData(GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, GL_STREAM_DRAW)
            
            # Update isosurface VBO if needed
            if self.visualization_type == "isosurface" and self.isosurface_vertices is not None:
                if not hasattr(self, 'isosurface_vbo') or self.isosurface_vbo is None:
                    self.isosurface_vbo = glGenBuffers(1)
                    if not self.isosurface_vbo:
                        print("Warning: Failed to generate isosurface VBO")
                        return
                
                glBindBuffer(GL_ARRAY_BUFFER, self.isosurface_vbo)
                # Check isosurface_vertices data type and convert if needed
                if not isinstance(self.isosurface_vertices, np.ndarray) or self.isosurface_vertices.dtype != np.float32:
                    self.isosurface_vertices = np.array(self.isosurface_vertices, dtype=np.float32)
                glBufferData(GL_ARRAY_BUFFER, self.isosurface_vertices.nbytes, 
                            self.isosurface_vertices, GL_STATIC_DRAW)
                
        except Exception as e:
            print(f"Error in update_vbo: {e}")
    
    def initializeGL(self):
        """Initialize OpenGL context"""
        # Skip if OpenGL not available
        if not HAS_OPENGL:
            return
            
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1, 0])
        glEnable(GL_COLOR_MATERIAL)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable point smoothing for particles
        glEnable(GL_POINT_SMOOTH)
        
        glClearColor(0.05, 0.05, 0.1, 1.0)
    
    def resizeGL(self, width, height):
        """Handle widget resize event"""
        # Skip if OpenGL not available
        if not HAS_OPENGL:
            return
            
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / height if height != 0 else 1
        gluPerspective(45, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """Render the scene"""
        # Skip if OpenGL not available
        if not HAS_OPENGL:
            return
            
        start_time = time.time()
        
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            # Set camera position
            glTranslatef(0.0, 0.0, self.z_trans)
            glRotatef(self.x_rot / 16.0, 1.0, 0.0, 0.0)
            glRotatef(self.y_rot / 16.0, 0.0, 1.0, 0.0)
            glRotatef(self.z_rot / 16.0, 0.0, 0.0, 1.0)
            
            try:
                # Draw based on visualization type
                if self.visualization_type == "particles":
                    self.render_particles()
                elif self.visualization_type == "isosurface":
                    self.render_isosurface()
                
                # Draw axes
                self.draw_axes()
                
                # Update performance tracking
                render_time = time.time() - start_time
                self.performance_stats['render_time'] = render_time
                self.performance_stats['avg_render_time'] = (
                    0.9 * self.performance_stats['avg_render_time'] + 
                    0.1 * render_time)
                    
            except Exception as e:
                print(f"Error in paintGL rendering: {e}")
                
        except Exception as e:
            print(f"Error in paintGL: {e}")
    
    def render_particles(self):
        """Render quantum system as particles"""
        # Skip if OpenGL not available
        if not HAS_OPENGL:
            return
            
        try:
            # Initialize particles if needed
            if not self.particles:
                self.initialize_particles()
                
            # Ensure we have vertices to render
            if self.vertices is None or len(self.vertices) == 0:
                self.update_vertex_arrays()
                    
            if self.vertices is None or len(self.vertices) == 0:
                return
                
            # Update VBO if needed
            if not hasattr(self, 'vbo') or self.vbo is None:
                self.update_vbo()
                if not hasattr(self, 'vbo') or self.vbo is None:
                    return
            
            # Set up rendering
            glDisable(GL_LIGHTING)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            
            # Render trails as lines
            if self.trail_effect:
                trail_count = 0
                for trail in self.particle_trails:
                    trail_count += len(trail)
                
                if trail_count > 0:
                    glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
                    glVertexPointer(3, GL_FLOAT, 0, None)
                    
                    glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
                    glColorPointer(4, GL_FLOAT, 0, None)
                    
                    glLineWidth(1.5)
                    for i, trail in enumerate(self.particle_trails):
                        if len(trail) >= 2:
                            offset = sum(len(t) for t in self.particle_trails[:i])
                            glDrawArrays(GL_LINE_STRIP, offset, len(trail))
            
            # Calculate offset for particle vertices
            particle_offset = 0
            if self.trail_effect:
                particle_offset = sum(len(trail) for trail in self.particle_trails)
            
            # Render particles with glow effect
            if self.glow_effect:
                # First pass: larger, dimmer points for glow
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
                glVertexPointer(3, GL_FLOAT, 0, None)
                
                glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
                glColorPointer(4, GL_FLOAT, 0, None)
                
                glPointSize(self.point_size * 2.0)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                glDrawArrays(GL_POINTS, particle_offset, len(self.particles))
                
                # Second pass: smaller, brighter points
                glPointSize(self.point_size)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glDrawArrays(GL_POINTS, particle_offset, len(self.particles))
            else:
                # Standard rendering without glow
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
                glVertexPointer(3, GL_FLOAT, 0, None)
                
                glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
                glColorPointer(4, GL_FLOAT, 0, None)
                
                glPointSize(self.point_size)
                glDrawArrays(GL_POINTS, particle_offset, len(self.particles))
            
            # Clean up state
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glEnable(GL_LIGHTING)
        except Exception as e:
            print(f"Error rendering particles: {e}")
    
    def render_isosurface(self):
        """Render quantum system as an isosurface"""
        # Skip if OpenGL not available
        if not HAS_OPENGL:
            return
            
        # Compute isosurface if needed
        if self.isosurface_vertices is None or not self.isosurface_cached:
            self.compute_isosurface()
            if self.isosurface_vertices is None:
                return
            self.update_vbo()
            
        if self.isosurface_vertices is None or self.isosurface_faces is None:
            return
            
        try:
            # Set up lighting
            glEnable(GL_LIGHTING)
            ambient = [0.2, 0.2, 0.3, 1.0]
            diffuse = [0.4, 0.6, 1.0, 0.8]
            specular = [1.0, 1.0, 1.0, 1.0]
            
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient)
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular)
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)
            
            # Draw the isosurface
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            
            # Bind vertices
            glBindBuffer(GL_ARRAY_BUFFER, self.isosurface_vbo)
            glVertexPointer(3, GL_FLOAT, 0, None)
            
            # Use computed normals
            if isinstance(self.isosurface_normals, np.ndarray):
                if self.isosurface_normals.dtype != np.float32:
                    self.isosurface_normals = self.isosurface_normals.astype(np.float32)
                glNormalPointer(GL_FLOAT, 0, self.isosurface_normals)
            
                # Draw triangles - only if faces is valid
                if self.isosurface_faces is not None and isinstance(self.isosurface_faces, np.ndarray):
                    # Convert faces to proper type if needed
                    if self.isosurface_faces.dtype != np.uint32:
                        self.isosurface_faces = self.isosurface_faces.astype(np.uint32)
                    if self.isosurface_faces.size > 0:
                        glDrawElements(GL_TRIANGLES, int(self.isosurface_faces.size), 
                                      GL_UNSIGNED_INT, self.isosurface_faces)
            
            # Clean up
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        except Exception as e:
            print(f"Error rendering isosurface: {e}")
    
    def draw_axes(self):
        """Draw coordinate axes"""
        if not HAS_OPENGL:
            return
            
        try:
            glDisable(GL_LIGHTING)
            
            axis_length = 10.0
            
            glLineWidth(2.0)
            
            # X axis (red)
            glBegin(GL_LINES)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(0, 0, 0)
            glVertex3f(axis_length, 0, 0)
            glEnd()
            
            # Y axis (green)
            glBegin(GL_LINES)
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, axis_length, 0)
            glEnd()
            
            # Z axis (blue)
            glBegin(GL_LINES)
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, axis_length)
            glEnd()
            
            glLineWidth(1.0)
            glEnable(GL_LIGHTING)
        except Exception as e:
            print(f"Error drawing axes: {e}")
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for rotation"""
        if self.last_pos:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            
            self.x_rot += dy * 8
            self.y_rot += dx * 8
            
            self.last_pos = event.pos()
            self.update()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zoom"""
        self.z_trans += event.angleDelta().y() / 120.0 * 0.5
        self.update()
    
    def set_visualization_type(self, viz_type):
        """Set visualization type (particles or isosurface)"""
        if viz_type in ["particles", "isosurface"]:
            self.visualization_type = viz_type
            # Reset isosurface cache when switching to isosurface mode
            if viz_type == "isosurface":
                self.isosurface_cached = False
            self.update()
    
    def toggle_glow_effect(self, enabled):
        """Toggle particle glow effect"""
        self.glow_effect = enabled
        self.update()
    
    def toggle_trail_effect(self, enabled):
        """Toggle particle trail effect"""
        self.trail_effect = enabled
        if not enabled:
            self.particle_trails = [[] for _ in range(len(self.particles))]
        self.update()
    
    def toggle_color_by_phase(self, enabled):
        """Toggle coloring particles by phase"""
        self.color_by_phase = enabled
        self.initialize_particles()
        self.update()
    
    def set_isosurface_threshold(self, value):
        """Set isosurface threshold (0-1)"""
        self.isosurface_threshold = max(0.01, min(1.0, value))
        # Mark isosurface as needing recalculation
        if abs(self.isosurface_threshold - self.last_iso_threshold) > 0.01:
            self.isosurface_cached = False
        
        if self.visualization_type == "isosurface":
            self.compute_isosurface()
            self.update_vbo()
            self.update()
    
    def set_num_particles(self, num):
        """Set the number of particles to render"""
        self.num_particles = max(100, min(10000, num))
        self.initialize_particles()
        self.update()
    
    def set_point_size(self, size):
        """Set the particle point size"""
        self.point_size = max(1.0, min(20.0, size))
        self.update()


#############################################################################
# PART 3: 2D Wave Function Slice Visualizer
#############################################################################

class WaveFunctionVisualizer2D(QWidget):
    """
    2D visualizer for quantum wave function slices,
    showing probability density, real/imaginary parts, and phase.
    """
    def __init__(self, quantum_system, parent=None):
        super(WaveFunctionVisualizer2D, self).__init__(parent)
        self.quantum_system = quantum_system
        
        # Visualization parameters
        self.display_mode = "probability"  # "probability", "real", "imag", "phase"
        self.slice_axis = 2  # 0=yz, 1=xz, 2=xy
        self.slice_position = 0.5  # Normalized position (0-1)
        self.colormap = "viridis"  # Default colormap
        
        # Set minimum size
        self.setMinimumSize(200, 200)
    
    def set_display_mode(self, mode):
        """Set display mode (probability, real, imag, phase)"""
        self.display_mode = mode
        self.update()
    
    def set_slice_axis(self, axis):
        """Set the axis to slice along (0, 1, or 2)"""
        self.slice_axis = axis % 3
        self.update()
    
    def set_slice_position(self, position):
        """Set the normalized slice position (0-1)"""
        self.slice_position = max(0.0, min(1.0, position))
        self.update()
    
    def set_colormap(self, cmap_name):
        """Set the colormap for display"""
        self.colormap = cmap_name
        self.update()
    
    def get_slice_data(self):
        """Get 2D slice of wave function data"""
        if not hasattr(self.quantum_system, 'psi') or self.quantum_system.psi is None:
            return None
            
        # Get wave function from system (ensuring we have a CPU array)
        if hasattr(self.quantum_system, 'use_gpu') and self.quantum_system.use_gpu:
            import cupy as cp
            psi = cp.asnumpy(self.quantum_system.psi)
        else:
            psi = self.quantum_system.psi
        
        # Get the 3D data based on display mode
        if self.display_mode == "probability":
            data_3d = np.abs(psi)**2
        elif self.display_mode == "real":
            data_3d = psi.real
        elif self.display_mode == "imag":
            data_3d = psi.imag
        elif self.display_mode == "phase":
            data_3d = np.angle(psi)
        else:
            data_3d = np.abs(psi)**2
        
        # Extract the slice
        N = data_3d.shape[0]
        idx = min(int(self.slice_position * (N-1)), N-1)
        
        if self.slice_axis == 0:  # yz plane (x fixed)
            return data_3d[idx, :, :]
        elif self.slice_axis == 1:  # xz plane (y fixed)
            return data_3d[:, idx, :]
        else:  # xy plane (z fixed)
            return data_3d[:, :, idx]
    
    def paintEvent(self, event):
        """Paint the 2D visualization"""
        painter = QPainter(self)
        
        # Clear background
        painter.fillRect(self.rect(), QColor(30, 30, 40))
        
        # Get the slice data
        data = self.get_slice_data()
        if data is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "No wave function data available")
            return
        
        # Normalize data for display
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data_norm = (data - data_min) / (data_max - data_min)
        else:
            data_norm = np.zeros_like(data)
        
        # Get painter dimensions
        width = self.width()
        height = self.height()
        
        # Calculate cell size
        N = data.shape[0]
        cell_width = width / N
        cell_height = height / N
        
        # Draw cells
        for i in range(N):
            for j in range(N):
                value = data_norm[i, j]
                
                # Get color from colormap
                if self.colormap == "viridis":
                    color = self.viridis_colormap(value)
                elif self.colormap == "plasma":
                    color = self.plasma_colormap(value)
                elif self.colormap == "phase":
                    if self.display_mode == "phase":
                        # Map phase (-pi to pi) to hue (0 to 1)
                        phase = (data[i, j] + np.pi) / (2 * np.pi)
                        r, g, b = colorsys.hsv_to_rgb(phase, 0.8, 0.8)
                        color = QColor(int(r*255), int(g*255), int(b*255))
                    else:
                        color = self.viridis_colormap(value)
                else:
                    color = self.viridis_colormap(value)
                
                # Draw the cell
                x = j * cell_width
                y = i * cell_height
                painter.fillRect(int(x), int(y), max(1, int(cell_width)), max(1, int(cell_height)), color)
        
        # Draw a border
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(0, 0, width-1, height-1)
        
        # Draw axis labels
        painter.setPen(Qt.white)
        axes = ['X', 'Y', 'Z']
        if self.slice_axis == 0:  # yz plane
            h_axis = axes[1]
            v_axis = axes[2]
        elif self.slice_axis == 1:  # xz plane
            h_axis = axes[0]
            v_axis = axes[2]
        else:  # xy plane
            h_axis = axes[0]
            v_axis = axes[1]
        
        # Horizontal axis label - convert float to int for drawText
        painter.drawText(int(width/2 - 10), int(height - 5), h_axis)
        
        # Vertical axis label
        painter.drawText(5, int(height/2), v_axis)
        
        # Display mode and slice info
        info_text = f"{self.display_mode.capitalize()}, {axes[self.slice_axis]}={self.slice_position:.2f}"
        painter.drawText(10, 20, info_text)
        
        # Value range
        range_text = f"Range: [{data_min:.3f}, {data_max:.3f}]"
        painter.drawText(width - 150, 20, range_text)
    
    def viridis_colormap(self, value):
        """Convert a value (0-1) to viridis colormap"""
        # Simplified version of viridis
        if value < 0.25:
            r = 0.2
            g = 0.3 + value
            b = 0.5 + value
        elif value < 0.5:
            r = 0.2 + (value - 0.25) * 2
            g = 0.55
            b = 0.75 - (value - 0.25)
        elif value < 0.75:
            r = 0.7
            g = 0.55 + (value - 0.5) * 1.2
            b = 0.5 - (value - 0.5)
        else:
            r = 0.7 + (value - 0.75) * 1.2
            g = 0.85 + (value - 0.75) * 0.6
            b = 0.25 - (value - 0.75) * 0.25
        
        return QColor(int(r*255), int(g*255), int(b*255))
    
    def plasma_colormap(self, value):
        """Convert a value (0-1) to plasma colormap"""
        # Simplified version of plasma
        if value < 0.25:
            r = 0.2 + value * 2
            g = 0.0
            b = 0.5 + value
        elif value < 0.5:
            r = 0.7
            g = 0.0 + (value - 0.25) * 2
            b = 0.75 - (value - 0.25)
        elif value < 0.75:
            r = 0.7 + (value - 0.5) * 0.5
            g = 0.5 + (value - 0.5)
            b = 0.5 - (value - 0.5) * 2
        else:
            r = 0.85 + (value - 0.75) * 0.6
            g = 0.75 + (value - 0.75)
            b = 0.0
        
        return QColor(int(r*255), int(g*255), int(b*255))
    
    def mousePressEvent(self, event):
        """Handle mouse press to change slice position"""
        # Change slice position based on click position
        self.slice_position = event.y() / self.height()
        self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse drag to change slice position"""
        if event.buttons() & Qt.LeftButton:
            self.slice_position = event.y() / self.height()
            self.update()

# PART 4: Matrix and ND Visualizers
#############################################################################

class MatrixVisualizerWidget(QWidget):
    """
    Widget for visualizing 2D slices of higher-dimensional matrices
    with configurable dimension selection and projection methods.
    """
    def __init__(self, parent=None):
        super(MatrixVisualizerWidget, self).__init__(parent)
        self.setMinimumSize(300, 300)
        self.x = 0
        self.y = 0
        self.size = 300
        self.selected_dims = [0, 1]
        self.fixed_indices = {}
        self.mode = "magnitude"
        self.last_update_time = 0
        self.color_palette = "viridis"
        self.projection_mode = "slice"
        self.error_message = None
        self.error_time = 0
        self.scale_factor = 1.0
        self.dimensions = 4
        self.matrix = None
        self.view_matrix = None
        self.time_value = 0
        self.performance_stats = {
            'update_time': 0.0,
            'render_time': 0.0,
            'avg_update_time': 0.0,
            'avg_render_time': 0.0
        }
        self.agents = []  # List to store agent data
        
        # UI setup
        self.layout = QVBoxLayout(self)
        self.toolbar = QWidget()
        self.toolbar_layout = QHBoxLayout(self.toolbar)
        self.layout.addWidget(self.toolbar)
        
        self.mode_button = QPushButton("Mode: Magnitude")
        self.mode_button.clicked.connect(self.toggle_mode)
        self.toolbar_layout.addWidget(self.mode_button)
        
        self.proj_button = QPushButton("Proj: Slice")
        self.proj_button.clicked.connect(self.toggle_projection)
        self.toolbar_layout.addWidget(self.proj_button)
        
        self.palette_button = QPushButton("Palette: Viridis")
        self.palette_button.clicked.connect(self.toggle_palette)
        self.toolbar_layout.addWidget(self.palette_button)
        
        self.view_area = QWidget()
        self.view_area.setStyleSheet("background-color: #101020;")
        self.view_area.setMinimumSize(280, 280)
        self.layout.addWidget(self.view_area)

    def toggle_mode(self):
        """Cycle through display modes"""
        modes = ["magnitude", "phase", "real", "imag"]
        current_index = modes.index(self.mode)
        self.mode = modes[(current_index + 1) % len(modes)]
        self.mode_button.setText(f"Mode: {self.mode.capitalize()}")
        self.update()
        
    def toggle_projection(self):
        """Cycle through projection modes"""
        modes = ["slice", "max", "mean"]
        current_index = modes.index(self.projection_mode)
        self.projection_mode = modes[(current_index + 1) % len(modes)]
        self.proj_button.setText(f"Proj: {self.projection_mode.capitalize()}")
        self.update()
        
    def toggle_palette(self):
        """Cycle through color palettes"""
        palettes = ["viridis", "plasma", "inferno", "magma"]
        current_index = palettes.index(self.color_palette)
        self.color_palette = palettes[(current_index + 1) % len(palettes)]
        self.palette_button.setText(f"Palette: {self.color_palette.capitalize()}")
        self.update()
        
    def set_dimensions(self, dim1, dim2):
        """Set which dimensions to display"""
        if dim1 == dim2:
            dim2 = (dim1 + 1) % max(2, self.dimensions)
        self.selected_dims = [dim1, dim2]
        self.update()

    def update_matrix(self, matrix, dimensions, time_value):
        """Update the matrix data for visualization"""
        update_start = time.time()
        
        self.matrix = matrix
        self.dimensions = dimensions
        self.time_value = time_value
        
        # Ensure matrix has appropriate dimensions
        if self.matrix is not None and self.matrix.ndim < dimensions:
            padded_shape = list(self.matrix.shape) + [1] * (dimensions - self.matrix.ndim)
            self.matrix = np.reshape(self.matrix, padded_shape)
        
        # Generate view matrix
        self.view_matrix = self.get_slice(matrix, dimensions)
        self.update()
        
        update_time = time.time() - update_start
        self.performance_stats['update_time'] = update_time
        self.performance_stats['avg_update_time'] = (
            0.9 * self.performance_stats['avg_update_time'] + 
            0.1 * update_time)

    def get_slice(self, matrix, dimensions):
        """Get 2D slice or projection of the matrix"""
        update_start = time.time()
        
        if matrix is None:
            return np.zeros((2, 2))
        
        shape = matrix.shape
        dims = min(dimensions, len(shape))
        
        # Validate selected dimensions
        for dim in self.selected_dims:
            if dim >= dims:
                self.error_message = f"Error: Selected dimension {dim+1} exceeds available dimensions {dims}"
                self.error_time = time.time()
                return np.zeros((2, 2))
        
        if self.projection_mode == "slice":
            slice_obj = []
            for d in range(dims):
                if d in self.selected_dims:
                    slice_obj.append(slice(None))
                else:
                    # Use fixed index with fallback and bounds checking
                    idx = self.fixed_indices.get(d, 0)  # Default to 0 instead of shape[d]//2
                    idx = min(max(0, idx), shape[d]-1)  # Ensure index is valid
                    slice_obj.append(idx)
            
            try:
                view = matrix[tuple(slice_obj)]
                
                # Handle transposition consistently
                if self.selected_dims[0] > self.selected_dims[1]:
                    view = view.T
            except Exception as e:
                print(f"Slice error: {e}")
                return np.zeros((2, 2))
                
        elif self.projection_mode == "max":
            try:
                reduced_matrix = matrix
                
                # Dynamically reduce dimensions not in selected_dims
                axes_to_reduce = [d for d in range(dims) if d not in self.selected_dims]
                for axis in sorted(axes_to_reduce, reverse=True):
                    if axis < len(reduced_matrix.shape):
                        reduced_matrix = np.max(reduced_matrix, axis=axis)
                
                # Handle transposition consistently
                if len(reduced_matrix.shape) == 2 and self.selected_dims[0] > self.selected_dims[1]:
                    view = reduced_matrix.T
                else:
                    view = reduced_matrix
            except Exception as e:
                print(f"Max projection error: {e}")
                return np.zeros((2, 2))
                
        elif self.projection_mode == "mean":
            try:
                reduced_matrix = matrix
                
                # Dynamically reduce dimensions not in selected_dims
                axes_to_reduce = [d for d in range(dims) if d not in self.selected_dims]
                for axis in sorted(axes_to_reduce, reverse=True):
                    if axis < len(reduced_matrix.shape):
                        reduced_matrix = np.mean(reduced_matrix, axis=axis)
                
                # Handle transposition consistently
                if len(reduced_matrix.shape) == 2 and self.selected_dims[0] > self.selected_dims[1]:
                    view = reduced_matrix.T
                else:
                    view = reduced_matrix
            except Exception as e:
                print(f"Mean projection error: {e}")
                return np.zeros((2, 2))
        else:
            view = np.zeros((2, 2))
        
        # Handle NaN values
        view = np.nan_to_num(view)
        
        update_time = time.time() - update_start
        self.performance_stats['update_time'] = update_time
        self.performance_stats['avg_update_time'] = 0.9 * self.performance_stats['avg_update_time'] + 0.1 * update_time
        
        return view

    def paintEvent(self, event):
        """Paint the matrix visualization"""
        render_start = time.time()
        super().paintEvent(event)
        
        if self.view_matrix is None or self.view_matrix.size == 0:
            return
            
        painter = QPainter(self.view_area)
        if not painter.isActive():
            return
            
        painter.setRenderHint(QPainter.Antialiasing)
        
        if np.max(self.view_matrix) > np.min(self.view_matrix):
            normalized = (self.view_matrix - np.min(self.view_matrix)) / (np.max(self.view_matrix) - np.min(self.view_matrix))
        else:
            normalized = np.zeros_like(self.view_matrix)
            
        matrix_size = min(self.view_matrix.shape)
        view_width = self.view_area.width()
        view_height = self.view_area.height()
        
        cell_size = min(view_width / self.view_matrix.shape[1], 
                       view_height / self.view_matrix.shape[0])
        
        offset_x = (view_width - cell_size * self.view_matrix.shape[1]) / 2
        offset_y = (view_height - cell_size * self.view_matrix.shape[0]) / 2
        
        phase_offset = self.time_value * 0.2
        
        for i in range(self.view_matrix.shape[0]):
            for j in range(self.view_matrix.shape[1]):
                value = normalized[i, j]
                
                if self.mode == "magnitude":
                    # Get color based on value and selected palette
                    if self.color_palette == "viridis":
                        hue = 0.6 - 0.4 * value
                        sat = 0.8
                        val = 0.4 + 0.6 * value
                    elif self.color_palette == "plasma":
                        hue = 0.8 - 0.6 * value
                        sat = 0.9
                        val = 0.4 + 0.6 * value
                    elif self.color_palette == "inferno":
                        hue = 0.1 - 0.1 * value
                        sat = 0.8 + 0.2 * value
                        val = 0.2 + 0.8 * value
                    else:  # magma
                        hue = 0.85 - 0.05 * value
                        sat = 0.7 + 0.3 * value
                        val = 0.2 + 0.8 * value
                        
                    rgb = colorsys.hsv_to_rgb(hue, sat, val)
                    color = QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                    
                elif self.mode == "phase":
                    phase = (i + j) / (matrix_size * 2) + phase_offset
                    hue = phase % 1.0
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
                    color = QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                    
                elif self.mode == "real":
                    real_val = np.sin(2 * np.pi * ((i + j) / matrix_size + phase_offset))
                    if real_val > 0:
                        intensity = real_val**0.8
                        color = QColor(int(intensity * 255), int(intensity * 200), 255)
                    else:
                        intensity = (-real_val)**0.8
                        color = QColor(255, int((1-intensity) * 255), int((1-intensity) * 255))
                    
                elif self.mode == "imag":
                    imag_val = np.cos(2 * np.pi * ((i + j) / matrix_size + phase_offset))
                    if imag_val > 0:
                        intensity = imag_val**0.8
                        color = QColor(int(intensity * 180), 255, int(intensity * 180))
                    else:
                        intensity = (-imag_val)**0.8
                        color = QColor(255, int((1-intensity) * 200), 255)
                
                x_pos = int(offset_x + j * cell_size)
                y_pos = int(offset_y + i * cell_size)
                cell_width = max(1, int(cell_size))
                cell_height = max(1, int(cell_size))
                
                painter.fillRect(x_pos, y_pos, cell_width, cell_height, color)
                
                if cell_size > 4:
                    painter.setPen(QColor(50, 50, 70))
                    painter.drawRect(x_pos, y_pos, cell_width, cell_height)
        
        # Draw information overlay
        painter.setPen(QColor(200, 200, 255))
        painter.drawText(10, 20, f"Dims: {self.selected_dims[0]+1},{self.selected_dims[1]+1} | Mode: {self.mode} | Proj: {self.projection_mode}")
        
        # Show fixed indices if any
        fixed_indices_text = "Fixed: "
        for dim, idx in self.fixed_indices.items():
            if dim not in self.selected_dims:
                fixed_indices_text += f"dim{dim+1}={idx+1}, "
        
        if len(fixed_indices_text) > 7:  # More than just "Fixed: "
            painter.drawText(10, view_height - 30, fixed_indices_text.rstrip(", "))
        
        # Show statistics
        if self.view_matrix.size > 0:
            stats = f"Min: {np.min(self.view_matrix):.3f}, Max: {np.max(self.view_matrix):.3f}, Mean: {np.mean(self.view_matrix):.3f}"
            painter.drawText(10, view_height - 10, stats)
        
        painter.end()
        
        render_time = time.time() - render_start
        self.performance_stats['render_time'] = render_time
        self.performance_stats['avg_render_time'] = 0.9 * self.performance_stats['avg_render_time'] + 0.1 * render_time

class NDVisualizer(QWidget):
    """
    Widget for visualizing N-dimensional quantum states with 
    configurable dimension selection and projections.
    """
    def __init__(self, quantum_system, parent=None):
        super(NDVisualizer, self).__init__(parent)
        self.quantum_system = quantum_system
        
        # Dimensions and visualization settings
        self.selected_dims = [0, 1, 2, 3]  # Default dimensions to visualize
        self.selection_radius = 5
        self.zoom = 40
        self.drag_start = None
        self.rotation = [0, 0, 0]
        self.projection_matrix = np.eye(4)
        self.display_mode = "points"  # "points", "wireframe", "surfaces"
        self.point_scale = 1.5
        
        # Threshold for probability visualization
        self.threshold = 0.3
        self.last_update_time = 0
        
        # Matrix data for visualization
        self.matrix = None  # Will hold probability density
        self.projected_points = []
        self.point_colors = []
        
        # Performance tracking
        self.performance_stats = {
            'update_time': 0.0,
            'render_time': 0.0,
            'avg_update_time': 0.0,
            'avg_render_time': 0.0
        }
        
        # Layout setup
        self.layout = QVBoxLayout(self)
        
        # Toolbar for controls
        self.toolbar = QWidget()
        self.toolbar_layout = QHBoxLayout(self.toolbar)
        self.layout.addWidget(self.toolbar)
        
        # Display mode button
        self.mode_button = QPushButton("Mode: Points")
        self.mode_button.clicked.connect(self.toggle_display_mode)
        self.toolbar_layout.addWidget(self.mode_button)
        
        # Reset view button
        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        self.toolbar_layout.addWidget(self.reset_button)
        
        # Dimension selector
        self.dim_selector = QWidget()
        self.dim_layout = QGridLayout(self.dim_selector)
        
        # Add dimension selection buttons
        max_dims = 6  # Maximum supported dimensions
        for i in range(max_dims):
            btn = QPushButton(f"Dim {i+1}")
            btn.setCheckable(True)
            btn.setChecked(i in self.selected_dims[:4])
            btn.clicked.connect(lambda checked, dim=i: self.toggle_dimension(dim, checked))
            
            row = i // 3
            col = i % 3
            self.dim_layout.addWidget(btn, row, col)
        
        self.layout.addWidget(self.dim_selector)
        
        # View area for rendering
        self.view_area = QWidget()
        self.view_area.setStyleSheet("background-color: #101020;")
        self.view_area.setMinimumSize(280, 280)
        self.layout.addWidget(self.view_area, stretch=1)
        
        # Initialize projection matrix
        self.update_projection_matrix()
        
    def toggle_display_mode(self):
        """Cycle through display modes"""
        modes = ["points", "wireframe", "surfaces"]
        current_index = modes.index(self.display_mode)
        self.display_mode = modes[(current_index + 1) % len(modes)]
        self.mode_button.setText(f"Mode: {self.display_mode.capitalize()}")
        self.update()
        
    def reset_view(self):
        """Reset view to default position"""
        self.rotation = [0, 0, 0]
        self.zoom = 40
        self.update_projection_matrix()
        self.update()
        
    def update_projection_matrix(self):
        """Update the 4D to 3D projection matrix based on rotation"""
        cx, sx = np.cos(self.rotation[0]), np.sin(self.rotation[0])
        cy, sy = np.cos(self.rotation[1]), np.sin(self.rotation[1])
        cz, sz = np.cos(self.rotation[2]), np.sin(self.rotation[2])
        
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        
        rotation_3d = rz @ ry @ rx
        
        self.projection_matrix = np.eye(4)
        self.projection_matrix[:3, :3] = rotation_3d
        
    def toggle_dimension(self, dim, checked):
        """Enable/disable a dimension for visualization"""
        if checked and dim not in self.selected_dims:
            if len(self.selected_dims) < 4:
                self.selected_dims.append(dim)
                self.selected_dims.sort()
            else:
                # Max 4 dimensions allowed for visualization
                sender = self.sender()
                if sender:
                    sender.setChecked(False)
                QMessageBox.warning(self, "Warning", "Maximum 4 dimensions can be selected")
        elif not checked and dim in self.selected_dims:
            if len(self.selected_dims) > 2:
                self.selected_dims.remove(dim)
            else:
                # Require at least 2 dimensions
                sender = self.sender()
                if sender:
                    sender.setChecked(True)
                QMessageBox.warning(self, "Warning", "Minimum 2 dimensions must be selected")

        # Validate dimensions against available dimensions in quantum system
        valid_dims = [d for d in self.selected_dims if d < self.quantum_system.params['dimensions']]
        if len(valid_dims) < len(self.selected_dims):
            self.selected_dims = valid_dims
            if len(valid_dims) < 2:
                self.selected_dims = list(range(min(2, self.quantum_system.params['dimensions'])))
        
        self.update()
    
    def update_matrix(self):
        """Update matrix data from quantum system for visualization"""
        update_start = time.time()
        
        try:
            # Get probability density (already CPU array)
            prob = self.quantum_system.probability_density()
            if prob is None:
                return
                
            self.matrix = prob
            
            # Ensure matrix has at least 4 dimensions for processing
            if self.matrix.ndim < 4:
                padded_shape = list(self.matrix.shape) + [1] * (4 - self.matrix.ndim)
                self.matrix = np.reshape(self.matrix, padded_shape)
            
            # Create projected visualization
            self.generate_projected_points()
            
            update_time = time.time() - update_start
            self.performance_stats['update_time'] = update_time
            self.performance_stats['avg_update_time'] = (
                0.9 * self.performance_stats['avg_update_time'] + 
                0.1 * update_time)
                
        except Exception as e:
            print(f"Error in ND visualization update: {e}")
            
    def generate_projected_points(self):
        """Generate points for visualization based on probability density"""
        if self.matrix is None:
            return
            
        try:
            # Clear existing points
            self.projected_points = []
            self.point_colors = []
            
            # Get dimensions of the matrix
            shape = self.matrix.shape
            dims = min(len(shape), 6)  # Limit to 6 dimensions maximum
            
            # Ensure we have enough dimensions selected
            valid_dims = [d for d in self.selected_dims if d < dims]
            if len(valid_dims) < 4:
                for d in range(dims):
                    if d not in valid_dims and len(valid_dims) < 4:
                        valid_dims.append(d)
                self.selected_dims = valid_dims[:4]
            
            # Get threshold for significant points
            max_val = np.max(self.matrix)
            if max_val <= 0:  # Prevent division by zero
                return
                
            threshold = max_val * self.threshold
            
            # Sample points from the matrix
            stride = max(1, shape[0] // 32)  # Skip points for performance
            points = []
            colors = []
            
            # Use valid indices for each dimension
            indices_range = [range(0, min(s, shape[d]), stride) for d, s in enumerate([shape[i] if i < len(shape) else 1 for i in range(dims)])]
            
            try:
                for idx in np.ndindex(*[len(r) for r in indices_range]):
                    real_idx = tuple(indices_range[d][idx[d]] for d in range(dims) if d < len(indices_range))
                    
                    # Skip dimensions beyond matrix shape
                    if any(i >= shape[d] for d, i in enumerate(real_idx) if d < len(shape)):
                        continue
                    
                    # Create a properly sized indexing tuple
                    index_tuple = real_idx[:len(shape)]
                    
                    # Calculate value at this point
                    try:
                        value = self.matrix[index_tuple]
                    except IndexError:
                        continue
                    
                    # Skip points below threshold
                    if value < threshold:
                        continue
                    
                    # Create 4D point with normalized coordinates (-1 to 1)
                    point = np.zeros(4)
                    for i, dim in enumerate(self.selected_dims[:4]):
                        if dim < dims and dim < len(real_idx):
                            # Avoid division by zero
                            denom = max(1, shape[dim] - 1)
                            coord = real_idx[dim] / denom * 2 - 1
                            point[i] = coord
                    
                    points.append(point)
                    
                    # Calculate color based on value
                    norm_value = (value - threshold) / (max_val - threshold)
                    hue = 0.6 - 0.4 * norm_value  # Blue to cyan
                    sat = 0.8
                    val = 0.4 + 0.6 * norm_value
                    rgb = colorsys.hsv_to_rgb(hue, sat, val)
                    colors.append((int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
            except Exception as e:
                print(f"Error in point generation: {e}")
                import traceback
                traceback.print_exc()
            
            # Project points from 4D to 3D
            self.projected_points = self.project_points(points)
            self.point_colors = colors
            
        except Exception as e:
            print(f"Error generating projected points: {e}")
            import traceback
            traceback.print_exc()
    
    def project_points(self, points):
        """Project 4D points to 3D space with perspective"""
        if not points:
            return []
            
        result = []
        view_width = self.view_area.width()
        view_height = self.view_area.height()
        center_x = view_width / 2
        center_y = view_height / 2
        
        for p in points:
            try:
                # Protect against division by zero by adding a small value
                # Using 3 + p[3] + 0.01 to ensure denominator is never exactly zero
                w_factor = 1.0 / (3.0 + p[3] + 0.01)
                p_3d = p[:3] * w_factor
                
                # Apply rotation matrix
                p_rot = self.projection_matrix[:3, :3] @ p_3d
                
                # Calculate screen coordinates
                screen_x = center_x + p_rot[0] * self.zoom
                screen_y = center_y + p_rot[1] * self.zoom
                z_depth = p_rot[2]  # For depth sorting
                
                result.append((screen_x, screen_y, z_depth))
            except Exception as e:
                print(f"Error projecting point {p}: {e}")
                # Add a default point as fallback
                result.append((center_x, center_y, 0.0))
            
        return result
    
    def mousePressEvent(self, event):
        """Handle mouse press for rotation"""
        self.drag_start = event.pos()
        
    def mouseMoveEvent(self, event):
        """Handle mouse drag for rotation"""
        if self.drag_start:
            delta_x = event.x() - self.drag_start.x()
            delta_y = event.y() - self.drag_start.y()
            
            self.rotation[1] += delta_x * 0.01
            self.rotation[0] += delta_y * 0.01
            
            self.update_projection_matrix()
            self.update()
            self.drag_start = event.pos()
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        self.drag_start = None
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y() / 120.0 * 2.0
        self.zoom = max(10, min(200, self.zoom + delta))
        self.update()
        
    def paintEvent(self, event):
        """Paint the ND visualization"""
        render_start = time.time()
        super().paintEvent(event)
        
        # Update matrix if needed
        if self.matrix is None:
            self.update_matrix()
            
        if not self.projected_points:
            return
            
        painter = QPainter(self)
        if not painter.isActive():
            return
            
        painter.setRenderHint(QPainter.Antialiasing)
        
        view_rect = self.view_area.geometry()
        
        # Sort points by depth for correct rendering
        point_data = list(zip(self.projected_points, self.point_colors))
        point_data.sort(key=lambda x: x[0][2], reverse=True)
        
        # Draw based on display mode
        if self.display_mode == "points":
            self.paint_points(painter, view_rect, point_data)
        elif self.display_mode == "wireframe":
            self.paint_wireframe(painter, view_rect, point_data)
        elif self.display_mode == "surfaces":
            self.paint_surfaces(painter, view_rect, point_data)
        
        # Draw axes and labels
        self.draw_axes(painter, view_rect)
        
        # Draw dimension info
        painter.setPen(QColor(200, 200, 255))
        dims_text = ', '.join([f"{d+1}" for d in self.selected_dims[:4]])
        painter.drawText(view_rect.left() + 10, view_rect.top() + 20, 
                        f"4D: {dims_text} | Mode: {self.display_mode.capitalize()}")
        
        painter.end()
        
        render_time = time.time() - render_start
        self.performance_stats['render_time'] = render_time
        self.performance_stats['avg_render_time'] = (
            0.9 * self.performance_stats['avg_render_time'] + 
            0.1 * render_time)
            
    def paint_points(self, painter, view_rect, point_data):
        """Paint points visualization mode"""
        for (x, y, z), color in point_data:
            x = view_rect.left() + x
            y = view_rect.top() + y
            
            if view_rect.contains(int(x), int(y)):
                size = max(2, int((1 + z) * 5 * self.point_scale))
                
                painter.setBrush(QColor(*color))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(int(x - size/2), int(y - size/2), size, size)
                
    def paint_wireframe(self, painter, view_rect, point_data):
        """Paint wireframe visualization mode"""
        # First draw points
        for (x, y, z), color in point_data:
            x = view_rect.left() + x
            y = view_rect.top() + y
            
            if view_rect.contains(int(x), int(y)):
                size = max(2, int((1 + z) * 3 * self.point_scale))
                painter.setBrush(QColor(*color))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(int(x - size/2), int(y - size/2), size, size)
        
        # Then draw lines between nearby points
        painter.setPen(QPen(QColor(100, 100, 150, 100), 1))
        
        # Use fewer connections for better performance
        for i, ((x1, y1, z1), color1) in enumerate(point_data):
            if i % 5 != 0:  # Only connect every 5th point
                continue
                
            x1 = view_rect.left() + x1
            y1 = view_rect.top() + y1
            
            # Connect to a few nearby points
            for j, ((x2, y2, z2), color2) in enumerate(point_data[i+1:i+6]):
                x2 = view_rect.left() + x2
                y2 = view_rect.top() + y2
                
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < 50:  # Only connect if they're close
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                    
    def paint_surfaces(self, painter, view_rect, point_data):
        """Paint surfaces visualization mode"""
        # Group points by z-bin (depth levels)
        groups = {}
        for i, ((x, y, z), color) in enumerate(point_data):
            x = view_rect.left() + x
            y = view_rect.top() + y
            
            z_bin = int(z * 10)  # Group into discrete z-levels
            if z_bin not in groups:
                groups[z_bin] = []
            groups[z_bin].append((x, y, color))
        
        # Draw each z-level as a surface (back to front)
        for z_bin in sorted(groups.keys(), reverse=True):
            points = groups[z_bin]
            if len(points) < 3:  # Need at least 3 points for a polygon
                continue
            
            # Calculate average color for this surface
            avg_color = [0, 0, 0]
            for _, _, color in points:
                avg_color[0] += color[0]
                avg_color[1] += color[1]
                avg_color[2] += color[2]
            
            n = len(points)
            avg_color = [c // n for c in avg_color]
            
            # Create polygon from points
            polygon = QPolygonF()
            for x, y, _ in points:
                polygon.append(QPointF(x, y))
            
            # Draw polygon with transparency
            painter.setBrush(QColor(*avg_color, 100))
            painter.setPen(QColor(*avg_color, 200))
            painter.drawPolygon(polygon)
            
    def draw_axes(self, painter, view_rect):
        """Draw coordinate axes with labels"""
        center_x = view_rect.left() + view_rect.width() / 2
        center_y = view_rect.top() + view_rect.height() / 2
        
        # Define 4D axes vectors
        axes = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # X axis
            np.array([0.0, 1.0, 0.0, 0.0]),  # Y axis
            np.array([0.0, 0.0, 1.0, 0.0]),  # Z axis
            np.array([0.0, 0.0, 0.0, 1.0])   # W axis
        ]
        
        # Define colors and labels for axes
        axis_colors = [QColor(255, 0, 0), QColor(0, 255, 0), 
                      QColor(0, 0, 255), QColor(255, 255, 0)]
        axis_labels = ['X', 'Y', 'Z', 'W']
        
        # Project origin and axes
        origin = np.array([0.0, 0.0, 0.0, 0.0])
        origin_proj = self.project_points([origin])[0]
        projected_axes = self.project_points(axes)
        
        # Draw each axis line and label
        for i, (proj, color, label) in enumerate(zip(projected_axes, axis_colors, axis_labels)):
            x, y, _ = proj
            x = center_x + x
            y = center_y + y
            
            painter.setPen(QPen(color, 2))
            painter.drawLine(
                int(center_x + origin_proj[0]), 
                int(center_y + origin_proj[1]), 
                int(x), int(y)
            )
            
            # Calculate label position
            label_x = int(center_x + origin_proj[0] + 1.2 * (x - center_x - origin_proj[0]))
            label_y = int(center_y + origin_proj[1] + 1.2 * (y - center_y - origin_proj[1]))
            
            painter.setPen(color)
            if i < len(self.selected_dims):
                painter.drawText(label_x, label_y, f"{label} ({self.selected_dims[i]+1})")
            else:
                painter.drawText(label_x, label_y, f"{label}")

class EnhancedQuantumControlsWidget(QWidget):
    """
    Widget providing controls for the quantum simulation and visualization settings.
    Allows adjusting quantum parameters, visualization options, and mode selection.
    """
    def __init__(self, quantum_system, visualizer, nd_visualizer=None, parent=None):
        super(EnhancedQuantumControlsWidget, self).__init__(parent)
        self.quantum_system = quantum_system
        self.visualizer = visualizer
        self.nd_visualizer = nd_visualizer
        self._setup_ui()
        self.update_controls_from_system()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        tabs = QTabWidget()
        
        quantum_tab = self._create_quantum_tab()
        tabs.addTab(quantum_tab, "Quantum")
        
        zeta_tab = self._create_zeta_tab()
        tabs.addTab(zeta_tab, "Zeta")
        
        viz_tab = self._create_viz_tab()
        tabs.addTab(viz_tab, "Visualization")
        
        if self.nd_visualizer:
            nd_tab = self._create_nd_tab()
            tabs.addTab(nd_tab, "ND Controls")
        
        main_layout.addWidget(tabs)
        
        io_group = QGroupBox("Import/Export")
        io_layout = QHBoxLayout(io_group)
        export_btn = QPushButton("Export State")
        export_btn.clicked.connect(self.export_quantum_state)
        import_btn = QPushButton("Import State")
        import_btn.clicked.connect(self.import_quantum_state)
        io_layout.addWidget(export_btn)
        io_layout.addWidget(import_btn)
        main_layout.addWidget(io_group)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignRight)
        main_layout.addWidget(self.status_label)
    
    def _create_quantum_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        wave_group = QGroupBox("Wave Packet")
        wave_layout = QGridLayout(wave_group)
        
        wave_layout.addWidget(QLabel("Width (nm):"), 0, 0)
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(1, 50)
        self.width_slider.setValue(10)
        self.width_slider.valueChanged.connect(lambda v: self.update_wave_param('sigma', v/10.0 * 1e-9))
        wave_layout.addWidget(self.width_slider, 0, 1)
        self.width_label = QLabel("1.0 nm")
        wave_layout.addWidget(self.width_label, 0, 2)
        
        wave_layout.addWidget(QLabel("Momentum (nm⁻¹):"), 1, 0)
        self.momentum_slider = QSlider(Qt.Horizontal)
        self.momentum_slider.setRange(-100, 100)
        self.momentum_slider.setValue(10)
        self.momentum_slider.valueChanged.connect(lambda v: self.update_momentum(v/10.0 * 1e9))
        wave_layout.addWidget(self.momentum_slider, 1, 1)
        self.momentum_label = QLabel("1.0 nm⁻¹")
        wave_layout.addWidget(self.momentum_label, 1, 2)
        
        layout.addWidget(wave_group)
        
        pot_group = QGroupBox("Potential")
        pot_layout = QGridLayout(pot_group)
        
        pot_layout.addWidget(QLabel("Type:"), 0, 0)
        self.potential_combo = QComboBox()
        self.potential_combo.addItems(["Free", "Harmonic", "Box", "Barrier", "Coulomb"])
        self.potential_combo.currentTextChanged.connect(self.update_potential_type)
        pot_layout.addWidget(self.potential_combo, 0, 1)
        
        pot_layout.addWidget(QLabel("Height (eV):"), 1, 0)
        self.potential_slider = QSlider(Qt.Horizontal)
        self.potential_slider.setRange(1, 100)
        self.potential_slider.setValue(10)
        self.potential_slider.valueChanged.connect(lambda v: self.update_potential_height(v/10.0 * 1.602e-19))
        pot_layout.addWidget(self.potential_slider, 1, 1)
        self.potential_label = QLabel("1.0 eV")
        pot_layout.addWidget(self.potential_label, 1, 2)
        
        layout.addWidget(pot_group)
        
        sim_group = QGroupBox("Simulation")
        sim_layout = QGridLayout(sim_group)
        
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_simulation)
        sim_layout.addWidget(reset_btn, 0, 0)
        
        sim_layout.addWidget(QLabel("Time Step:"), 1, 0)
        self.dt_slider = QSlider(Qt.Horizontal)
        self.dt_slider.setRange(1, 100)
        self.dt_slider.setValue(10)
        self.dt_slider.valueChanged.connect(lambda v: self.update_time_step(v/10.0))
        sim_layout.addWidget(self.dt_slider, 1, 1)
        self.dt_label = QLabel("1.0x")
        sim_layout.addWidget(self.dt_label, 1, 2)
        
        layout.addWidget(sim_group)
        
        return tab
    
    def _create_zeta_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        zeta_group = QGroupBox("Zeta Fractal")
        zeta_layout = QGridLayout(zeta_group)
        
        zeta_layout.addWidget(QLabel("s parameter:"), 0, 0)
        self.s_slider = QSlider(Qt.Horizontal)
        self.s_slider.setRange(5, 30)
        self.s_slider.setValue(12)
        self.s_slider.valueChanged.connect(lambda v: self.update_zeta_param('s', v/10.0))
        zeta_layout.addWidget(self.s_slider, 0, 1)
        self.s_label = QLabel("1.2")
        zeta_layout.addWidget(self.s_label, 0, 2)
        
        zeta_layout.addWidget(QLabel("Number of modes:"), 1, 0)
        self.n_modes_slider = QSlider(Qt.Horizontal)
        self.n_modes_slider.setRange(10, 200)
        self.n_modes_slider.setValue(50)
        self.n_modes_slider.valueChanged.connect(lambda v: self.update_zeta_param('N_modes', v))
        zeta_layout.addWidget(self.n_modes_slider, 1, 1)
        self.n_modes_label = QLabel("50")
        zeta_layout.addWidget(self.n_modes_label, 1, 2)
        
        zeta_layout.addWidget(QLabel("Frequency scaling:"), 2, 0)
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(1, 50)
        self.freq_slider.setValue(10)
        self.freq_slider.valueChanged.connect(lambda v: self.update_zeta_param('freq_scale', v/10.0))
        zeta_layout.addWidget(self.freq_slider, 2, 1)
        self.freq_label = QLabel("1.0")
        zeta_layout.addWidget(self.freq_label, 2, 2)
        
        zeta_layout.addWidget(QLabel("Amplitude:"), 3, 0)
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(1, 30)
        self.amp_slider.setValue(10)
        self.amp_slider.valueChanged.connect(lambda v: self.update_zeta_param('amp_mod', v/10.0))
        zeta_layout.addWidget(self.amp_slider, 3, 1)
        self.amp_label = QLabel("1.0")
        zeta_layout.addWidget(self.amp_label, 3, 2)
        
        layout.addWidget(zeta_group)
        
        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.schrodinger_radio = QRadioButton("Schrödinger Mode")
        self.zeta_radio = QRadioButton("Zeta Function Mode")
        self.schrodinger_radio.setChecked(True)
        
        self.schrodinger_radio.toggled.connect(lambda: self.switch_mode('schrodinger'))
        self.zeta_radio.toggled.connect(lambda: self.switch_mode('zeta'))
        
        mode_layout.addWidget(self.schrodinger_radio)
        mode_layout.addWidget(self.zeta_radio)
        
        layout.addWidget(mode_group)
        
        return tab
    
    def _create_viz_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        viz_type_group = QGroupBox("Visualization Type")
        viz_type_layout = QVBoxLayout(viz_type_group)
        
        self.particle_radio = QRadioButton("Particle Representation")
        self.isosurface_radio = QRadioButton("Isosurface Representation")
        self.particle_radio.setChecked(True)
        
        self.particle_radio.toggled.connect(lambda: self.set_visualization_type("particles"))
        self.isosurface_radio.toggled.connect(lambda: self.set_visualization_type("isosurface"))
        
        viz_type_layout.addWidget(self.particle_radio)
        viz_type_layout.addWidget(self.isosurface_radio)
        
        viz_type_group.setLayout(viz_type_layout)
        layout.addWidget(viz_type_group)
        
        particle_group = QGroupBox("Particle Options")
        particle_layout = QGridLayout(particle_group)
        
        particle_layout.addWidget(QLabel("Particles:"), 0, 0)
        self.particles_slider = QSlider(Qt.Horizontal)
        self.particles_slider.setRange(100, 5000)
        self.particles_slider.setValue(1000)
        self.particles_slider.valueChanged.connect(lambda v: self.visualizer.set_num_particles(v))
        particle_layout.addWidget(self.particles_slider, 0, 1)
        self.particles_label = QLabel("1000")
        particle_layout.addWidget(self.particles_label, 0, 2)
        
        particle_layout.addWidget(QLabel("Point Size:"), 1, 0)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 20)
        self.size_slider.setValue(5)
        self.size_slider.valueChanged.connect(lambda v: self.visualizer.set_point_size(v))
        particle_layout.addWidget(self.size_slider, 1, 1)
        self.size_label = QLabel("5")
        particle_layout.addWidget(self.size_label, 1, 2)
        
        particle_layout.addWidget(QLabel("Effects:"), 2, 0)
        effects_layout = QHBoxLayout()
        
        self.glow_checkbox = QCheckBox("Glow")
        self.glow_checkbox.setChecked(True)
        self.glow_checkbox.stateChanged.connect(lambda s: self.visualizer.toggle_glow_effect(s == Qt.Checked))
        effects_layout.addWidget(self.glow_checkbox)
        
        self.trails_checkbox = QCheckBox("Trails")
        self.trails_checkbox.setChecked(True)
        self.trails_checkbox.stateChanged.connect(lambda s: self.visualizer.toggle_trail_effect(s == Qt.Checked))
        effects_layout.addWidget(self.trails_checkbox)
        
        self.phase_checkbox = QCheckBox("Phase Color")
        self.phase_checkbox.setChecked(True)
        self.phase_checkbox.stateChanged.connect(lambda s: self.visualizer.toggle_color_by_phase(s == Qt.Checked))
        effects_layout.addWidget(self.phase_checkbox)
        
        particle_layout.addLayout(effects_layout, 2, 1, 1, 2)
        
        layout.addWidget(particle_group)
        
        iso_group = QGroupBox("Isosurface Options")
        iso_layout = QGridLayout(iso_group)
        
        iso_layout.addWidget(QLabel("Threshold:"), 0, 0)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setValue(30)
        self.threshold_slider.valueChanged.connect(lambda v: self.visualizer.set_isosurface_threshold(v/100.0))
        iso_layout.addWidget(self.threshold_slider, 0, 1)
        self.threshold_label = QLabel("0.30")
        iso_layout.addWidget(self.threshold_label, 0, 2)
        
        layout.addWidget(iso_group)
        
        return tab
    
    def _create_nd_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        dim_group = QGroupBox("Dimensions")
        dim_layout = QGridLayout(dim_group)
        
        dim_layout.addWidget(QLabel("Dimensions:"), 0, 0)
        self.dim_slider = QSlider(Qt.Horizontal)
        self.dim_slider.setRange(3, 4)
        self.dim_slider.setValue(self.quantum_system.params['dimensions'])
        self.dim_slider.valueChanged.connect(self.update_dimensions)
        dim_layout.addWidget(self.dim_slider, 0, 1)
        self.dim_label = QLabel(f"{self.quantum_system.params['dimensions']}D")
        dim_layout.addWidget(self.dim_label, 0, 2)
        
        layout.addWidget(dim_group)
        
        nd_viz_group = QGroupBox("ND Visualization")
        nd_viz_layout = QGridLayout(nd_viz_group)
        
        nd_viz_layout.addWidget(QLabel("Probability Threshold:"), 0, 0)
        self.nd_threshold_slider = QSlider(Qt.Horizontal)
        self.nd_threshold_slider.setRange(1, 100)
        self.nd_threshold_slider.setValue(int(self.nd_visualizer.threshold * 100))
        self.nd_threshold_slider.valueChanged.connect(lambda v: self.update_nd_threshold(v/100.0))
        nd_viz_layout.addWidget(self.nd_threshold_slider, 0, 1)
        
        nd_viz_layout.addWidget(QLabel("Point Scale:"), 1, 0)
        self.nd_point_scale_slider = QSlider(Qt.Horizontal)
        self.nd_point_scale_slider.setRange(5, 30)
        self.nd_point_scale_slider.setValue(int(self.nd_visualizer.point_scale * 10))
        self.nd_point_scale_slider.valueChanged.connect(lambda v: self.update_nd_point_scale(v/10.0))
        nd_viz_layout.addWidget(self.nd_point_scale_slider, 1, 1)
        
        layout.addWidget(nd_viz_group)
        
        return tab
    
    def update_controls_from_system(self):
        sigma = self.quantum_system.params.get('sigma', 1e-9)
        self.width_slider.setValue(int(sigma * 1e10))
        self.width_label.setText(f"{sigma * 1e9:.1f} nm")
        
        k0 = self.quantum_system.params.get('k0', [1e9, 0, 0, 0])
        self.momentum_slider.setValue(int(k0[0] * 1e-8))
        self.momentum_label.setText(f"{k0[0] * 1e-9:.1f} nm⁻¹")
        
        pot_type = self.quantum_system.params.get('potential_type', 'harmonic')
        index = self.potential_combo.findText(pot_type.capitalize(), Qt.MatchFixedString)
        if index >= 0:
            self.potential_combo.setCurrentIndex(index)
        
        pot_height = self.quantum_system.params.get('potential_height', 1.602e-19)
        self.potential_slider.setValue(int(pot_height / 1.602e-19 * 10))
        self.potential_label.setText(f"{pot_height / 1.602e-19:.1f} eV")
        
        s = self.quantum_system.params.get('s', 1.25)
        self.s_slider.setValue(int(s * 10))
        self.s_label.setText(f"{s:.2f}")
        
        n_modes = self.quantum_system.params.get('N_modes', 50)
        self.n_modes_slider.setValue(int(n_modes))
        self.n_modes_label.setText(f"{n_modes}")
        
        freq_scale = self.quantum_system.params.get('freq_scale', 1.0)
        self.freq_slider.setValue(int(freq_scale * 10))
        self.freq_label.setText(f"{freq_scale:.1f}")
        
        amp_mod = self.quantum_system.params.get('amp_mod', 1.0)
        self.amp_slider.setValue(int(amp_mod * 10))
        self.amp_label.setText(f"{amp_mod:.1f}")
        
        if self.quantum_system.mode == 'schrodinger':
            self.schrodinger_radio.setChecked(True)
        else:
            self.zeta_radio.setChecked(True)
            
        if hasattr(self, 'dim_slider'):
            dimensions = self.quantum_system.params.get('dimensions', 3)
            self.dim_slider.setValue(dimensions)
            self.dim_label.setText(f"{dimensions}D")
    
    def set_visualization_type(self, viz_type):
        if hasattr(self.visualizer, 'set_visualization_type'):
            self.visualizer.set_visualization_type(viz_type)
    
    def update_wave_param(self, param_name, value):
        self.quantum_system.params[param_name] = value
        if param_name == 'sigma':
            self.width_label.setText(f"{value * 1e9:.1f} nm")
        if self.quantum_system.mode == 'schrodinger':
            self.quantum_system.initialize()
            self.update_visualization()
    
    def update_momentum(self, value):
        self.quantum_system.params['k0'][0] = value
        self.momentum_label.setText(f"{value * 1e-9:.1f} nm⁻¹")
        if self.quantum_system.mode == 'schrodinger':
            self.quantum_system.initialize()
            self.update_visualization()
    
    def update_potential_type(self, pot_type):
        self.quantum_system.params['potential_type'] = pot_type.lower()
        if self.quantum_system.mode == 'schrodinger':
            self.quantum_system.initialize()
            self.update_visualization()
    
    def update_potential_height(self, value):
        self.quantum_system.params['potential_height'] = value
        self.potential_label.setText(f"{value / 1.602e-19:.1f} eV")
        if self.quantum_system.mode == 'schrodinger':
            self.quantum_system.initialize()
            self.update_visualization()
    
    def update_time_step(self, scale_factor):
        self.quantum_system.params['dt'] = 1e-18 * scale_factor
        self.dt_label.setText(f"{scale_factor:.1f}x")
    
    def update_zeta_param(self, param_name, value):
        self.quantum_system.params[param_name] = value
        if param_name == 's':
            self.s_label.setText(f"{value:.2f}")
        elif param_name == 'N_modes':
            self.n_modes_label.setText(f"{value}")
        elif param_name == 'freq_scale':
            self.freq_label.setText(f"{value:.1f}")
        elif param_name == 'amp_mod':
            self.amp_label.setText(f"{value:.1f}")
        
        if self.quantum_system.mode == 'zeta':
            self.quantum_system.initialize()
            self.update_visualization()
    
    def update_dimensions(self, dimensions):
        if dimensions != self.quantum_system.params['dimensions']:
            self.quantum_system.params['dimensions'] = dimensions
            self.dim_label.setText(f"{dimensions}D")
            self.quantum_system.initialize()
            self.update_visualization()
    
    def update_nd_threshold(self, value):
        if self.nd_visualizer:
            self.nd_visualizer.threshold = value
            self.nd_visualizer.update_matrix()
            self.nd_visualizer.update()
    
    def update_nd_point_scale(self, value):
        if self.nd_visualizer:
            self.nd_visualizer.point_scale = value
            self.nd_visualizer.update()
    
    def switch_mode(self, mode):
        if mode != self.quantum_system.mode:
            self.quantum_system.mode = mode
            self.quantum_system.initialize()
            self.update_visualization()
            self.status_label.setText(f"Switched to {mode.capitalize()} mode")
    
    def reset_simulation(self):
        self.quantum_system.initialize()
        self.update_visualization()
        self.status_label.setText("Simulation reset")
    
    def export_quantum_state(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Quantum State", "", 
                                                 "Quantum State Files (*.qst);;All Files (*)")
        if file_path:
            success = self.quantum_system.export_state(file_path)
            if success:
                self.status_label.setText(f"State exported to {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "Export Error", "Failed to export quantum state")
    
    def import_quantum_state(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Quantum State", "", 
                                                 "Quantum State Files (*.qst);;All Files (*)")
        if file_path:
            success = self.quantum_system.import_state(file_path)
            if success:
                self.update_controls_from_system()
                self.update_visualization()
                self.status_label.setText(f"State imported from {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "Import Error", "Failed to import quantum state")
    
    def update_visualization(self):
        if hasattr(self.visualizer, 'initialize_particles'):
            self.visualizer.initialize_particles()
        if hasattr(self.visualizer, 'compute_isosurface'):
            self.visualizer.compute_isosurface()
        if hasattr(self.visualizer, 'update_vbo'):
            self.visualizer.update_vbo()
        self.visualizer.update()
        if self.nd_visualizer:
            self.nd_visualizer.update_matrix()
            self.nd_visualizer.update()


class QuantumZetaSimulatorApp(QMainWindow):
    """
    Main application window for the Enhanced Quantum Fractal Simulator.
    Organizes and connects all visualization components and controls.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Quantum Fractal Simulator")
        self.resize(1200, 800)

        self.time_scale = 1.0
        
        try:
            use_gpu = HAS_CUPY
            self.quantum_system = QuantumZetaSystem(use_gpu=use_gpu)
        except Exception as e:
            print(f"Error: {e}")
            self.quantum_system = QuantumZetaSystem(use_gpu=False)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        viz_tabs = QTabWidget()
        viz_tabs.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(viz_tabs, stretch=3)
        
        self.quantum_3d_viz = QuantumVisualizerWidget(self.quantum_system)
        viz_tabs.addTab(self.quantum_3d_viz, "3D Visualization")
        
        slice_widget = QWidget()
        slice_layout = QGridLayout(slice_widget)
        
        self.prob_viz = WaveFunctionVisualizer2D(self.quantum_system)
        self.prob_viz.set_display_mode("probability")
        slice_layout.addWidget(self.prob_viz, 0, 0)
        
        self.phase_viz = WaveFunctionVisualizer2D(self.quantum_system)
        self.phase_viz.set_display_mode("phase")
        self.phase_viz.set_colormap("phase")
        slice_layout.addWidget(self.phase_viz, 0, 1)
        
        self.real_viz = WaveFunctionVisualizer2D(self.quantum_system)
        self.real_viz.set_display_mode("real")
        slice_layout.addWidget(self.real_viz, 1, 0)
        
        self.imag_viz = WaveFunctionVisualizer2D(self.quantum_system)
        self.imag_viz.set_display_mode("imag")
        slice_layout.addWidget(self.imag_viz, 1, 1)
        
        viz_tabs.addTab(slice_widget, "2D Slices")
        
        nd_tab = QWidget()
        nd_layout = QGridLayout(nd_tab)
        
        self.nd_viz = NDVisualizer(self.quantum_system)
        nd_layout.addWidget(self.nd_viz, 0, 0)
        
        self.matrix_viz1 = MatrixVisualizerWidget()
        self.matrix_viz1.set_dimensions(0, 1)
        nd_layout.addWidget(self.matrix_viz1, 0, 1)
        
        self.matrix_viz2 = MatrixVisualizerWidget()
        self.matrix_viz2.set_dimensions(0, 2)
        nd_layout.addWidget(self.matrix_viz2, 1, 0, 1, 2)
        
        viz_tabs.addTab(nd_tab, "N-Dimensional View")
        
        self.controls = EnhancedQuantumControlsWidget(
            self.quantum_system, 
            self.quantum_3d_viz,
            self.nd_viz
        )
        main_layout.addWidget(self.controls, stretch=1)
        
        self.statusBar().showMessage("Quantum Fractal Simulator Ready")
        
        self._create_menu()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(50)
        
        self.last_update_time = time.time()
        self.frame_count = 0
        self.fps = 0
    
    def _create_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('File')
        
        new_action = file_menu.addAction('New Simulation')
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_simulation)
        
        export_action = file_menu.addAction('Export State')
        export_action.setShortcut('Ctrl+S')
        export_action.triggered.connect(self.controls.export_quantum_state)
        
        import_action = file_menu.addAction('Import State')
        import_action.setShortcut('Ctrl+O')
        import_action.triggered.connect(self.controls.import_quantum_state)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('Exit')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        
        sim_menu = menubar.addMenu('Simulation')
        
        mode_menu = sim_menu.addMenu('Mode')
        schrodinger_action = mode_menu.addAction('Schrödinger Mode')
        schrodinger_action.triggered.connect(lambda: self.controls.switch_mode('schrodinger'))
        zeta_action = mode_menu.addAction('Zeta Fractal Mode')
        zeta_action.triggered.connect(lambda: self.controls.switch_mode('zeta'))
        
        presets_menu = sim_menu.addMenu('Presets')
        
        free_action = presets_menu.addAction('Free Particle')
        free_action.triggered.connect(lambda: self.load_preset('free'))
        
        harmonic_action = presets_menu.addAction('Harmonic Oscillator')
        harmonic_action.triggered.connect(lambda: self.load_preset('harmonic'))
        
        box_action = presets_menu.addAction('Infinite Well')
        box_action.triggered.connect(lambda: self.load_preset('box'))
        
        barrier_action = presets_menu.addAction('Tunneling Barrier')
        barrier_action.triggered.connect(lambda: self.load_preset('barrier'))
        
        presets_menu.addSeparator()
        
        zeta_critical_action = presets_menu.addAction('Zeta (Critical)')
        zeta_critical_action.triggered.connect(lambda: self.load_preset('zeta_critical'))
        
        zeta_complex_action = presets_menu.addAction('Zeta (Complex)')
        zeta_complex_action.triggered.connect(lambda: self.load_preset('zeta_complex'))
        
        viz_menu = menubar.addMenu('Visualization')
        
        particles_action = viz_menu.addAction('Particles')
        particles_action.triggered.connect(lambda: self.controls.set_visualization_type('particles'))
        
        isosurface_action = viz_menu.addAction('Isosurface')
        isosurface_action.triggered.connect(lambda: self.controls.set_visualization_type('isosurface'))
        
        view_menu = menubar.addMenu('View')
        
        reset_view_action = view_menu.addAction('Reset View')
        reset_view_action.triggered.connect(self.reset_all_views)
        
        help_menu = menubar.addMenu('Help')
        
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about_dialog)
        
        help_action = help_menu.addAction('Help')
        help_action.triggered.connect(self.show_help_dialog)
    
    def on_tab_changed(self, index):
        current_tab = self.centralWidget().findChild(QTabWidget).tabText(index)
        
        if current_tab == "N-Dimensional View":
            valid_dims = [d for d in self.nd_viz.selected_dims if d < self.quantum_system.params['dimensions']]
            if len(valid_dims) < 2:
                valid_dims = list(range(min(4, self.quantum_system.params['dimensions'])))[:2]
            self.nd_viz.selected_dims = valid_dims
            self.nd_viz.update_matrix()
    
    def new_simulation(self):
        self.quantum_system.initialize()
        self.controls.update_controls_from_system()
        self.update_all_visualizations()
        self.statusBar().showMessage("New simulation started")
    
    def load_preset(self, preset_name):
        if preset_name == 'free':
            self.quantum_system.params.update({
                'potential_type': 'free',
                'sigma': 0.8e-9,
                'k0': [3e9, 0, 0, 0]
            })
            self.quantum_system.mode = 'schrodinger'
        elif preset_name == 'harmonic':
            self.quantum_system.params.update({
                'potential_type': 'harmonic',
                'sigma': 0.5e-9,
                'k0': [0, 0, 0, 0]
            })
            self.quantum_system.mode = 'schrodinger'
        elif preset_name == 'box':
            self.quantum_system.params.update({
                'potential_type': 'box',
                'sigma': 0.8e-9,
                'k0': [1e9, 1e9, 0, 0]
            })
            self.quantum_system.mode = 'schrodinger'
        elif preset_name == 'barrier':
            self.quantum_system.params.update({
                'potential_type': 'barrier',
                'sigma': 0.5e-9,
                'k0': [5e9, 0, 0, 0],
                'potential_height': 5 * 1.602e-19
            })
            self.quantum_system.mode = 'schrodinger'
        elif preset_name == 'zeta_critical':
            self.quantum_system.params.update({
                's': 0.5,
                'freq_scale': 1.0,
                'amp_mod': 1.0,
                'N_modes': 100
            })
            self.quantum_system.mode = 'zeta'
        elif preset_name == 'zeta_complex':
            self.quantum_system.params.update({
                's': 1.5,
                'freq_scale': 2.0,
                'amp_mod': 1.5,
                'N_modes': 150
            })
            self.quantum_system.mode = 'zeta'
        
        self.quantum_system.initialize()
        self.controls.update_controls_from_system()
        self.update_all_visualizations()
        self.statusBar().showMessage(f"Loaded preset: {preset_name}")
    
    def update_simulation(self):
        try:
            self.quantum_system.update()
            
            current_tab = self.centralWidget().findChild(QTabWidget).currentIndex()
            
            if current_tab == 0:  # 3D
                self.quantum_3d_viz.initialize_particles()
                self.quantum_3d_viz.update()
            elif current_tab == 1:  # 2D Slices
                self.prob_viz.update()
                self.phase_viz.update()
                self.real_viz.update()
                self.imag_viz.update()
            elif current_tab == 2:  # ND
                self.nd_viz.update_matrix()
                self.nd_viz.update()
                
                # Update matrix visualizers
                prob = self.quantum_system.probability_density()
                if prob is not None:
                    self.matrix_viz1.update_matrix(prob, self.quantum_system.params['dimensions'], self.quantum_system.params['time'])
                    self.matrix_viz2.update_matrix(prob, self.quantum_system.params['dimensions'], self.quantum_system.params['time'])
            
            self.frame_count += 1
            elapsed = time.time() - self.last_update_time
            
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.last_update_time = time.time()
                self.frame_count = 0
                
                memory = self.quantum_system.performance_stats['memory_usage']
                time_fs = self.quantum_system.params['time']*1e15
                self.statusBar().showMessage(f"FPS: {self.fps:.1f} | Memory: {memory:.1f} MB | Time: {time_fs:.1f} fs")
        except Exception as e:
            print(f"Error in update_simulation: {e}")
            import traceback
            traceback.print_exc()
    
    def update_all_visualizations(self):
        self.quantum_3d_viz.initialize_particles()
        self.quantum_3d_viz.compute_isosurface()
        self.quantum_3d_viz.update_vbo()
        self.quantum_3d_viz.update()
        
        self.prob_viz.update()
        self.phase_viz.update()
        self.real_viz.update()
        self.imag_viz.update()
        
        self.nd_viz.update_matrix()
        self.nd_viz.update()
        
        # Update matrix visualizers
        prob = self.quantum_system.probability_density()
        if prob is not None:
            self.matrix_viz1.update_matrix(prob, self.quantum_system.params['dimensions'], self.quantum_system.params['time'])
            self.matrix_viz2.update_matrix(prob, self.quantum_system.params['dimensions'], self.quantum_system.params['time'])
    
    def reset_all_views(self):
        self.quantum_3d_viz.x_rot = 0
        self.quantum_3d_viz.y_rot = 0
        self.quantum_3d_viz.z_rot = 0
        self.quantum_3d_viz.z_trans = -15.0
        self.quantum_3d_viz.update()
        
        self.nd_viz.reset_view()
        
        self.statusBar().showMessage("All views reset")
    
    def show_about_dialog(self):
        QMessageBox.about(self, "About Quantum Fractal Simulator",
                         "Enhanced Quantum Fractal Simulator\n\n"
                         "Combined quantum mechanics simulator "
                         "with fractal zeta function visualization.\n\n"
                         "Features: Schrödinger solver, Zeta visualization, "
                         "3D/ND visualization, GPU acceleration, potentials.")
    
    def show_help_dialog(self):
        QMessageBox.information(self, "Help",
                               "Usage:\n"
                               "1. Select mode (Schrödinger or Zeta)\n"
                               "2. Adjust parameters in control panel\n"
                               "3. Use tabs for different views\n"
                               "4. Mouse: drag=rotate, wheel=zoom\n"
                               "5. 2D Slices: click to change position")
    
    def closeEvent(self, event):
        self.timer.stop()
        
        # Properly clean up OpenGL resources
        if hasattr(self, 'quantum_3d_viz'):
            self.quantum_3d_viz.cleanup()
            
        event.accept()


def main():
    """Application entry point"""
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    try:
        has_opengl = HAS_OPENGL
        if not has_opengl:
            print("WARNING: PyOpenGL not available. Features limited.")
            QMessageBox.warning(None, "Missing Dependency", 
                           "PyOpenGL not installed. Features limited.\n"
                           "Install PyOpenGL for full functionality.")
    
        window = QuantumZetaSimulatorApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "Error", 
                            f"Error: {str(e)}\n\n"
                            "Required: PyQt5, PyOpenGL, numpy, scipy")
        raise


if __name__ == "__main__":
    main()
