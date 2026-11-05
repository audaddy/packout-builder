@echo off
rem Packout insert builder - one-step launcher for Windows.
rem First run: creates a private Python environment in .venv and installs the dependencies (a few minutes).
rem Every run: starts the app and opens it in your browser at http://localhost:8501
rem Needs Python 3.9 - 3.12 from https://www.python.org/downloads/  (tick "Add python.exe to PATH" or keep the "py" launcher).
setlocal
cd /d "%~dp0"
if "%PORT%"=="" set PORT=8501

if not exist ".venv\Scripts\python.exe" (
  set "PYCMD="
  for %%V in (3.11 3.12 3.10 3.9) do (
    if not defined PYCMD (
      py -%%V -c "import sys" >nul 2>nul && set "PYCMD=py -%%V"
    )
  )
  if not defined PYCMD (
    python -c "import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul && set "PYCMD=python"
  )
  if not defined PYCMD (
    echo Could not find Python 3.9 - 3.12. Install Python 3.11 from https://www.python.org/downloads/ and run this again.
    pause
    exit /b 1
  )
  echo Setting up ^(first run only^) ...
  %PYCMD% -m venv .venv
  if errorlevel 1 ( echo Could not create the Python environment. & pause & exit /b 1 )
)

fc /b requirements.txt .venv\requirements.installed.txt >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies - this takes a few minutes the first time ...
  .venv\Scripts\python.exe -m pip install --upgrade pip >nul
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 ( echo Installing the dependencies failed. & pause & exit /b 1 )
  copy /y requirements.txt .venv\requirements.installed.txt >nul
)

if not exist static mkdir static
if not exist outputs mkdir outputs

if not defined NO_BROWSER start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:%PORT%"
echo Packout builder is starting at http://localhost:%PORT%  ^(close this window to stop^)
.venv\Scripts\python.exe -m streamlit run app.py --server.port %PORT%
pause
