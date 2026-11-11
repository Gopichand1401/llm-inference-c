"""Build and install the pocketllm Python package:   pip install .

The C sources in src/ are compiled into one shared library that sits inside
the package (pocketllm/_pocket*.so) and is loaded with ctypes. Needs a C
compiler and zlib (Ubuntu: sudo apt install build-essential zlib1g-dev).
Set POCKETLLM_NO_OPENMP=1 to build without OpenMP (single-threaded).
"""
import glob
import os

from setuptools import Extension, setup

openmp = [] if os.environ.get("POCKETLLM_NO_OPENMP") else ["-fopenmp"]

native = Extension(
    "pocketllm._pocket",
    sources=sorted(glob.glob("src/*.c")),
    include_dirs=["include"],
    libraries=["z", "m"],
    extra_compile_args=["-std=c11", "-O2"] + openmp,
    extra_link_args=openmp,
)

setup(
    name="pocketllm",
    version="0.3.0",
    description="A LLaMA-style inference engine in C with RAG over spreadsheets",
    packages=["pocketllm"],
    package_dir={"": "python"},
    ext_modules=[native],
    python_requires=">=3.8",
)
