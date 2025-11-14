# anaklasis 

[![Anaklasis](https://github.com/reflectivity/analysis/workflows/ORSO%20Val.%20Anaklasis/badge.svg)](https://github.com/reflectivity/analysis/actions/workflows/validate_anaklasis.yml)

[_anaklasis_](https://github.com/alexandros-koutsioumpas/anaklasis) is a set of open-source _Python3_ scripts (with _FORTRAN 90_ extensions) that facilitate a range of specular neutron and x-ray reflectivity calculations, involving the generation of theoretical curves and the comparison/fit of interfacial model reflectivity against experimental datasets.  The _ref_ module, contains three callable functions, _ref.calculate_ for generating theoretical reflectivity curves, _ref.compare_ for comparison of experimental data with theoretical curves and _ref.fit_ for refinement of experimental data against a defined model. Execution takes place by defining the interfacial model and instrumental parameters as lists in a simple _Python_ script and by passing them as arguments to the desired function.

In the [examples folder](https://github.com/alexandros-koutsioumpas/anaklasis/tree/main/examples) many scripts with calculations and refinements performed by the program can be found. Addiitonaly some [_Jupyter notebooks_](https://github.com/alexandros-koutsioumpas/anaklasis/tree/main/examples-Jupyter) explaining the input and output are included.

Full description of used methods is reported in an open-access article at the [_Journal of Applied Crystallography_](https://doi.org/10.1107/S1600576721009262).

## Installation

It is adviced to install a _FORTRAN_ compiler on your system before proceeding with installation, so that the extensions for reflectivity calculations can be compiled. If you perform the installation without a _FORTRAN_ compiler, a _Python_ calculation engine will be used by the package. The engine can be considerably accelerated by installing the [_Numba_ package](https://numba.readthedocs.io/en/stable/user/installing.html) after installing _anaklasis_.

Note that calculations with _Numba_ are 30-40% slower than with the _FORTRAN_ extensions, while _Python_ calculation engine without _Numba_ installed can be more than 20-30 times slower than the _FORTRAN_ extensions.

**Linux**

- Install _Python_ >= 3.7 and optionally _gfortran_ 
- then download the latest _anaklasis_ release, unzip the archive and install through the terminal

```bash
python -m pip install numpy scipy matplotlib sympy numdifftools emcee tqdm corner meson ninja 
python setup.py install --user 
```

Test if the installation with gfortran succeeded.

```bash
python -c "from anaklasis import ref; print('Fortran OK:', getattr(ref, 'engine', None) == 'fortran' and hasattr(ref, 'f_realref'))"
```

- If _gfortran_ was not present, install _Numba_ package.

**macOS**

- Install _python_ >= 3.7 from [python.org](https://www.python.org/downloads/)
- Optionally install _gfortran_ compiler. An easy way is to use the installers provided by _fxcoudert_ at [github](https://github.com/fxcoudert/gfortran-for-macOS)
- Install _NumPy_  

```bash
python3 -m pip install numpy
```

- then download _anaklasis_, navigate to the proper folder  and install throught the terminal

```bash
python3 setup.py install
```

- If _gfortran_ was not present, install _Numba_ package.

```bash
python3 -m pip install numba
```

Note that if you prefer you may use [MacPorts](https://www.macports.org) or [Homebrew](https://brew.sh) for the installation of _gfortran_ compiler.

**Windows 10 (_Numba_)**


Anaklasis supports a lightweight installation mode on Windows that uses the Numba JIT backend. This mode requires no Fortran compiler and is the recommended installation path for Windows users. Due to compatibility constraints with NumPy, Numba, llvmlite, and distutils,  Windows installations MUST use Python 3.11.

To ensure compatibility with Python 3.11 on Windows, the following versions must be used:

- `setuptools < 60`
- `numpy < 2.0`
- `numba` + `llvmlite`


#### I. Install the required downgraded core packages:

```bash
pip install "setuptools<60"
pip install "numpy<2"
```

#### II. Install remaining runtime dependencies:

```bash
pip install scipy matplotlib sympy numdifftools emcee tqdm corner numba llvmlite
```

#### III. Navigate to the *anaklasis* directory and install:

```bash
python setup.py install
```

Since the environment variable `ANAKLASIS_USE_MSYS2` is not set,  
Windows automatically installs Anaklasis in Numba mode and skips all Fortran build steps.


#### IV. Verification (Windows)

```bash
python -c "from anaklasis import ref; print('Engine:', ref.engine)"
```

Expected:

```bash
Engine: numba
```


**Windows 10 (_Python_ 3.9 only with provided _wheel_ )**

For convenience a _wheel_ with a pre-compiled _fortran_ extension is provided (folder `\win_wheel`) for _Python_ version 3.9. For installing it follow the steps below:

- Install _python_ 3.9 from [python.org](https://www.python.org/downloads/) and do not forget to include _python_ in the systen path.
- Install _NumPy_ and upgrade _setuptools_
```bash
py -m pip install --upgrade pip setuptools

py -m pip install numpy
```

then navigate in the `\win_wheel` folder and install through the command prompt

```bash
py -m pip install anaklasis-1.6.0-cp39-cp39m-win_amd64.whl
```

In case you prefer *Anaconda* on *Windows* just make sure you have _setuptools_ and _NumPy_ on the system and just install the wheel.

**Windows 10 (using WSL)**

An additional option for Windows 10 users is to use the Windows Subsystem for Linux (WSL), install a Linux distribution (like Ubuntu) and follow the installation instructions presented above for Linux.


**Windows 11 (Python ≥ 3.12, MSYS2 Installation)**

Uses MSYS2 (MINGW64) as the unified Python + compiler toolchain.

#### I. Install MSYS2

Download from https://www.msys2.org/

Install to:
C:\msys64

Then open **MSYS2 MinGW 64-bit** (not UCRT64) and update:

```
pacman -Syu
# restart MSYS2
pacman -Su
```

---

#### II. Install required scientific packages (system-wide)

```
pacman -S --needed     mingw-w64-x86_64-python     mingw-w64-x86_64-python-pip     mingw-w64-x86_64-python-setuptools     mingw-w64-x86_64-python-wheel     mingw-w64-x86_64-python-numpy     mingw-w64-x86_64-python-scipy     mingw-w64-x86_64-python-matplotlib     mingw-w64-x86_64-python-sympy     mingw-w64-x86_64-gcc     mingw-w64-x86_64-gcc-fortran     git
```

These system packages avoid heavy pip builds and ensure compatibility with the compiler.

---

#### III. Create a virtual environment with system packages

```
python -m venv --system-site-packages ~/anaklasis
source ~/anaklasis/bin/activate
```

Test:

```
echo $MSYSTEM
which python
python -c "import numpy, matplotlib; print(numpy.__version__, matplotlib.__version__)"
```

Expected:
- MINGW64
- /mingw64/bin/python
- Working numpy + matplotlib imports

---

#### IV. Install light pure-Python packages inside the venv

```
pip install numdifftools emcee tqdm corner
```

These contain no C extensions and install cleanly.

---

#### V. Clone Anaklasis


```
git clone https://github.com/stefanhausler/anaklasis.git
cd anaklasis
```

---

#### VI. Install Anaklasis

```
export ANAKLASIS_USE_MSYS2=1
python setup.py install
```

---

#### VII. Verify Fortran backend

```
python -c "from anaklasis import ref; print('Fortran OK:', getattr(ref, 'engine', None) == 'fortran' and hasattr(ref, 'f_realref'))"
```

Expected:

```
Fortran OK: True
```




## Running _anaklasis_ in the cloud

You may use the following **Binder** link in order to perform _anaklasis_ calculations inside _jupyter notebooks_ that run on the cloud .

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alexandros-koutsioumpas/anaklasis/HEAD?filepath=templates_binder)

You may find templates for calculations and refinements in the form of _jupyter notebooks_ that you can modify according to your needs. You may also upload data files in order to use them in your refinements. Resources on **Binder** are a bit limited but you are able to experiment with _anaklasis_ without performing a local installation on your machine.

## Getting help

In the `/docs` folder you may find the program's API in various formats.

Also after installing *anaklasis*, you may install [pdoc](https://pdoc3.github.io/pdoc/) and from the console run

```python
pdoc anaklasis
```

that will open a html version of the API in your browser.

In this [YouTube link](https://www.youtube.com/watch?v=ieulImJUK5o) you may also find a video tutorial for fitting the [ORSO example data](https://github.com/reflectivity/reflectivity.github.io/blob/master/workshops/workshop_2021/ORSO_example.ort) with _anaklasis_.

## Templates

In the directory `/script_templates` you may find commented scripts that may guide you for writing _anaklasis_ analysis scripts.

## Post-installation tests

In the folder `/tests` you will find a _Python_ script `tests.py` that runs tests of the core calculations used by all three _anaklasis_ functions and reports if the results are the same as those obtained by version 1.5.2 of the package. You may run the tests using

```bash
python3 tests.py
```

If the run is succesful you will receive a `All anaklasis tests passed!` message at the end.

Also a script that runs only the [ORSO validation](https://github.com/andyfaff/orso/tree/master/reflectivity/test/unpolarised) tests `ORSO_tests.py` can be found in `/tests`.

## Note about running Bootstrap analysis

On _macOS_ and _Linux_ before performing a bootstrap analysis you might need to increase the user resource limit using the command

```bash
ulimit -n 2048
```
## Jupyter notebook examples

Jupyter notebook examples of model calculations

- [XRR of supported lipid membrane at the water/Si interface](https://github.com/alexandros-koutsioumpas/anaklasis/blob/main/examples-Jupyter/XRR_lipid_membrane_calculations.ipynb)
- [X-ray reflectivity calculations (2 layers)](https://github.com/alexandros-koutsioumpas/anaklasis/blob/main/examples-Jupyter/XRR_calculations_2_layers.ipynb)
- [Data/Theory compasion for NR of supported lipid membrane](https://github.com/alexandros-koutsioumpas/anaklasis/blob/main/examples-Jupyter/NR_membrane_data_theory_comparison.ipynb)


Jupyter notebook examples of data refinements

- [Polymer Brush (single neutron reflection curve)](https://github.com/alexandros-koutsioumpas/anaklasis/blob/main/examples-Jupyter/Brush_Neutron_reflectivity_fit.ipynb)
- [Polydisperse polymer brush (single neutron reflection curve)](https://github.com/alexandros-koutsioumpas/anaklasis/blob/main/examples-Jupyter/Polydisperse_Brush_Neutron_reflectivity_fit.ipynb)
- [adsorbed protein (two neutron reflection curves)](https://github.com/alexandros-koutsioumpas/anaklasis/blob/main/examples-Jupyter/lysozyme_fit.ipynb)
- [compact polymer layer (ORSO example)](https://github.com/alexandros-koutsioumpas/anaklasis/blob/main/examples-Jupyter/ORSO_example.ipynb)
- [supported lipid bilayer (3 neutron reflection curves)](https://github.com/alexandros-koutsioumpas/anaklasis/blob/main/examples-Jupyter/lipid_membrane_3_NR_contrast_fit.ipynb)
- [Co-refinement of PNR and XRR data](https://github.com/alexandros-koutsioumpas/anaklasis/blob/main/examples-Jupyter/NR_XRR_Fit_AlO3_Co_Pt.ipynb)


## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
