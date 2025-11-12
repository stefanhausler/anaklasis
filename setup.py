import os
import sys
import platform
import subprocess

try:
    import numpy
    from numpy.distutils.core import setup, Extension
    has_numpy_distutils = True
except ImportError:
    has_numpy_distutils = False

def run_f2py_build():
    cmd = [
        sys.executable, "-m", "numpy.f2py",
        "-m", "fortran_ref",
        "-c", "anaklasis/fortran_ref_functions.f90",
        "--opt=-O3", "--quiet"
    ]

    if platform.system() == "Windows":
        print("+++ Configuring Windows build (MSYS2 compilers)...")
        os.environ["CC"] = r"C:\msys64\ucrt64\bin\gcc.exe"
        os.environ["FC"] = r"C:\msys64\ucrt64\bin\gfortran.exe"
        pyhome = sys.prefix.replace("\\", "/")
        os.environ["CFLAGS"]  = f"-I{pyhome}/include"
        os.environ["LDFLAGS"] = f"-L{pyhome}/libs"
        os.environ["PATH"] = r"C:\msys64\ucrt64\bin;" + os.environ["PATH"]

    print("+++ Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("\n+++ Fortran module built successfully\n")

if has_numpy_distutils and sys.version_info < (3, 12):
    print("+++ Using legacy numpy.distutils build...")
    ext = Extension(
        name="anaklasis.fortran_ref",
        sources=["anaklasis/fortran_ref_functions.f90"],
    )
    setup(name="anaklasis", ext_modules=[ext])
else:
    print("+++ Using modern f2py/Meson build...")
    run_f2py_build()
