# Copyright (C) 2024 Nathaniel (RAPIDENN)
# Part of QFVCS - Licensed under GPL-3.0
# See LICENSE and core.py for details

from setuptools import setup, find_packages

setup(
    name="qfvcs",
    version="0.1.0",
    author="RAPIDENN",
    description="Quantum Fractal Visualization & Computation System",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RAPIDENN/QFVCS",
    packages=find_packages(),  # searches for qfvcs/
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "matplotlib",
        "PyQt5",
        "PyOpenGL",
        "PyOpenGL_accelerate"
        # "cupy" optional if compatible with the environment
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
