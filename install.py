#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import subprocess
import platform
import venv
from pathlib import Path

def print_header(message):
    print(f"\n{'=' * 60}")
    print(f"  {message}")
    print(f"{'=' * 60}")

def check_python_version():
    print("• Checking Python version...")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print(f"Python {sys.version.split()[0]} detected")
    
def create_virtual_env():
    print("\n• Setting up virtual environment...")
    venv_path = Path("venv")
    
    if venv_path.exists():
        print(f"✓ Existing virtual environment found at {venv_path}")
    else:
        print(f"• Creating new virtual environment at {venv_path}...")
        venv.create(venv_path, with_pip=True)
        print(f"✓ Virtual environment created at {venv_path}")
    
    return venv_path

def get_pip_command(venv_path):
    if platform.system() == "Windows":
        pip_path = venv_path / "Scripts" / "pip"
    else:
        pip_path = venv_path / "bin" / "pip"
    
    return str(pip_path)

def install_requirements(pip_command):
    print_header("Installing required dependencies")
    
    basic_packages = [
        "numpy",
        "scipy",
        "scikit-image",
        "matplotlib",
        "pyqt5"
    ]
    
    print("\n• Installing basic dependencies...")
    for package in basic_packages:
        print(f"• Installing {package}...")
        subprocess.run([pip_command, "install", package], check=True)
    
    print("\n• Installing OpenGL dependencies...")
    try:
        subprocess.run([pip_command, "install", "pyopengl", "pyopengl_accelerate"], check=True)
        print("✓ PyOpenGL successfully installed")
    except subprocess.CalledProcessError:
        print("⚠️ Error installing PyOpenGL acceleration. Installing basic PyOpenGL only...")
        subprocess.run([pip_command, "install", "pyopengl"], check=True)
    
    print("\n• Checking support for GPU acceleration...")
    try:
        subprocess.run([pip_command, "install", "cupy"], check=True)
        print("✓ CuPy successfully installed - GPU acceleration available")
    except subprocess.CalledProcessError:
        print("ℹ️ Could not install CuPy. GPU acceleration will not be available.")
        print("ℹ️ The system will work correctly using CPU only.")

def finalize_setup(venv_path):
    print_header("Installation Completed")
    
    activate_cmd = "venv\\Scripts\\activate" if platform.system() == "Windows" else "source venv/bin/activate"
    
    print("\nTo activate the virtual environment:")
    print(f"  {activate_cmd}")
    print("\nTo run QFVCS:")
    print("  python QFVCS.py")
    
    print("\nDocumentation: https://github.com/RAPIDENN/QFVCS")
    print("\n✨ System ready to use! ✨\n")

def main():
    print_header("QFVCS: Quantum Fractal Visualization & Computation System")
    print("Dependency Installer v1.0")
    
    start_time = time.time()
    
    try:
        check_python_version()
        venv_path = create_virtual_env()
        pip_command = get_pip_command(venv_path)
        
        install_requirements(pip_command)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Installation completed in {elapsed_time:.1f} seconds")
        
        finalize_setup(venv_path)
        
    except Exception as e:
        print(f"\nERROR during installation: {str(e)}")
        print("Please verify system requirements and permissions.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
