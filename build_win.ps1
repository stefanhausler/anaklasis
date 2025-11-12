# Activate Conda env
conda activate anaklasis
Set-Location 'C:\Users\haeusler\Documents\Projects\LARS\code\anaklasis'

# --- Environment configuration ---
$env:MSYS2_PATH = 'C:\msys64\ucrt64\bin'
$env:CONDA_PY = 'C:\Users\haeusler\anaconda3\envs\anaklasis'

# Ensure MSYS2 tools are visible first
$env:PATH = "$env:MSYS2_PATH;$env:PATH"

# Explicitly tell Meson/F2PY which compilers and Python to use
$env:CC = "$env:MSYS2_PATH\gcc.exe"
$env:FC = "$env:MSYS2_PATH\gfortran.exe"
$env:CFLAGS = "-I$env:CONDA_PY\include"
$env:LDFLAGS = "-L$env:CONDA_PY\libs"
$env:PKG_CONFIG_PATH = "$env:CONDA_PY\lib\pkgconfig"

# --- Clean previous builds ---
Remove-Item -Recurse -Force build,dist,*.egg-info,anaklasis\*.pyd -ErrorAction SilentlyContinue

Write-Host '+++ Using modern f2py/Meson build...'
Write-Host '+++ Configuring Windows build (forcing Conda Python)...'

# Force Meson to use Conda’s Python explicitly
& "$env:CONDA_PY\python.exe" -m numpy.f2py -m fortran_ref -c anaklasis/fortran_ref_functions.f90 --opt=-O3 --quiet

# --- Verify build (escaped properly for PowerShell) ---
& "$env:CONDA_PY\python.exe" -c "import anaklasis; print('Fortran OK:', hasattr(anaklasis,'fortran_ref'))"

