import os
import sys
import platform
import subprocess
import shutil

try:
    import numpy
    from numpy.distutils.core import setup, Extension
    has_numpy_distutils = True
except ImportError:
    # Fallback: allow a pure-Python install even if numpy.distutils is not available
    from setuptools import setup
    has_numpy_distutils = False

is_windows = platform.system() == "Windows"
is_linux = platform.system() == "Linux"
is_macos = platform.system() == "Darwin"

# ------------------------------------------------------------------
# User-controlled MSYS2 Fortran flag
# ------------------------------------------------------------------
use_msys2 = os.environ.get("ANAKLASIS_USE_MSYS2", "0") == "1"


def run_f2py_build():
    """Build Fortran extension using f2py (Linux/macOS or MSYS2-if-enabled)."""
    cmd = [
        sys.executable, "-m", "numpy.f2py",
        "-m", "fortran_ref",
        "-c", "anaklasis/fortran_ref_functions.f90",
        "--opt=-O3", "--quiet"
    ]
    print("+++ Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("\n+++ Fortran module built successfully\n")


# ============================================================
# WINDOWS — DEFAULT: Numba-only (no Fortran)
# ============================================================
if is_windows and not use_msys2:
    print("+++ Windows detected — ANAKLASIS_USE_MSYS2 not set.")
    print("+++ Skipping Fortran build. Using Numba backend.")

    setup(
        name="anaklasis",
        version="1.6.0",
        description="Neutron and X-ray reflectivity calculations",
        packages=["anaklasis"],
        install_requires=[
            "numpy>=1.22",
            "scipy>=1.4",
            "matplotlib",
            "numdifftools>=0.9.39",
            "sympy>=1.6.2",
            "emcee>=3.0",
            "tqdm",
            "corner",
            "numba",
        ],
        zip_safe=False,
    )

    print("\n*** Installed WITHOUT Fortran — using Numba backend on Windows ***\n")
    sys.exit(0)


# ============================================================
# WINDOWS — OPTIONAL MSYS2 Fortran build
# (Only when user explicitly sets ANAKLASIS_USE_MSYS2=1)
# ============================================================
if is_windows and use_msys2:
    print("+++ MSYS2 Fortran build explicitly enabled by user.")
    print("+++ Attempting Fortran build using numpy.distutils.")

    ext = Extension(
        name="anaklasis.fortran_ref",
        sources=["anaklasis/fortran_ref_functions.f90"],
    )

    setup(
        name="anaklasis",
        version="1.6.0",
        description="Neutron and X-ray reflectivity calculations",
        packages=["anaklasis"],
        ext_modules=[ext],
        install_requires=[
            "numpy>=1.22",
            "scipy>=1.4",
            "matplotlib",
            "numdifftools>=0.9.39",
            "sympy>=1.6.2",
            "emcee>=3.0",
            "tqdm",
            "corner",
        ],
    )

    print("\n*** Windows MSYS2 Fortran build finished ***\n")
    sys.exit(0)


# ============================================================
# LINUX / MACOS — Fortran build with Python fallback
# ============================================================
print("+++ Linux/macOS detected — attempting full Fortran build with Python fallback.")

install_requires_common = [
    "numpy>=1.22",
    "scipy>=1.4",
    "matplotlib",
    "numdifftools>=0.9.39",
    "sympy>=1.6.2",
    "emcee>=3.0",
    "tqdm",
    "corner",
]

try:
    if has_numpy_distutils and sys.version_info < (3, 12):
        print("+++ Using legacy numpy.distutils build (Python < 3.12).")

        ext = Extension(
            name="anaklasis.fortran_ref",
            sources=["anaklasis/fortran_ref_functions.f90"],
        )

        setup(
            name="anaklasis",
            version="1.6.0",
            packages=["anaklasis"],
            ext_modules=[ext],
            install_requires=install_requires_common,
        )

        print("\n*** Fortran extension built successfully on Linux/macOS (numpy.distutils) ***\n")

    else:
        print("+++ Using modern f2py build (Python ≥ 3.12 or no numpy.distutils).")

        # Build the module in-place via f2py, then install package as pure Python
        run_f2py_build()

        setup(
            name="anaklasis",
            version="1.6.0",
            packages=["anaklasis"],
            install_requires=install_requires_common,
        )

        print("\n*** Fortran extension built successfully on Linux/macOS via f2py ***\n")

except Exception as e:
    # Option B: if Fortran build fails, fall back to pure Python install.
    print("\n*** Warning: Fortran build failed, installing anaklasis WITHOUT Fortran extension. ***")
    print("Reason:", e)
    print("*** The runtime will fall back to the pure Python/Numba engine. ***\n")

    setup(
        name="anaklasis",
        version="1.6.0",
        packages=["anaklasis"],
        install_requires=install_requires_common,
    )
