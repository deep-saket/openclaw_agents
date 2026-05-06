@echo off
setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0"
set "REQUIRED_PYTHON_LINE=3.11"
set "REQUIRED_PYTHON_VERSION=3.11.9"
cd /d "%ROOT_DIR%"

echo [setup] repo root: %ROOT_DIR%

set "PYTHON_CMD="
set "PYTHON_LINE_FOUND="

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_LINE_FOUND="
  for /f %%v in ('py -3.11 -c "import platform; print(platform.python_version())" 2^>nul') do set "PYTHON_LINE_FOUND=%%v"
  if /I "%PYTHON_LINE_FOUND%"=="%REQUIRED_PYTHON_VERSION%" (
    set "PYTHON_CMD=py -3.11"
  )
)

if not defined PYTHON_CMD (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PYTHON_LINE_FOUND="
    for /f %%v in ('python -c "import platform; print(platform.python_version())" 2^>nul') do set "PYTHON_LINE_FOUND=%%v"
    if /I "%PYTHON_LINE_FOUND%"=="%REQUIRED_PYTHON_VERSION%" (
      set "PYTHON_CMD=python"
    )
  )
)

if not defined PYTHON_CMD (
    echo [setup] Python %REQUIRED_PYTHON_VERSION% not found. Attempting install ...

    where winget >nul 2>nul
    if %errorlevel%==0 (
      winget install --id Python.Python.3.11 -e --version %REQUIRED_PYTHON_VERSION% --silent --accept-package-agreements --accept-source-agreements
    ) else (
      where choco >nul 2>nul
      if %errorlevel%==0 (
        choco install python311 -y
      ) else (
        echo [error] Could not auto-install Python.
        echo [error] Install Python 3.11+ manually, then re-run this script.
        exit /b 1
      )
    )

  where py >nul 2>nul
  if %errorlevel%==0 (
    set "PYTHON_LINE_FOUND="
    for /f %%v in ('py -3.11 -c "import platform; print(platform.python_version())" 2^>nul') do set "PYTHON_LINE_FOUND=%%v"
    if /I "%PYTHON_LINE_FOUND%"=="%REQUIRED_PYTHON_VERSION%" set "PYTHON_CMD=py -3.11"
  )
  if not defined PYTHON_CMD (
    where python >nul 2>nul
    if %errorlevel%==0 (
      set "PYTHON_LINE_FOUND="
      for /f %%v in ('python -c "import platform; print(platform.python_version())" 2^>nul') do set "PYTHON_LINE_FOUND=%%v"
      if /I "%PYTHON_LINE_FOUND%"=="%REQUIRED_PYTHON_VERSION%" set "PYTHON_CMD=python"
    )
  )
  if not defined PYTHON_CMD (
    echo [error] Python install command completed but Python %REQUIRED_PYTHON_VERSION% is still unavailable on PATH.
    echo [error] Open a new Command Prompt and run this script again.
    exit /b 1
  )
)

echo [setup] using python launcher: %PYTHON_CMD%

if exist ".venv" (
  echo [setup] deleting existing virtual environment at .venv
  rmdir /s /q ".venv"
)

echo [setup] creating fresh virtual environment at .venv
%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
  echo [error] Failed to create virtual environment with: %PYTHON_CMD%
  exit /b 1
)

set "VENV_PY=%ROOT_DIR%.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [error] Could not find venv python executable: %VENV_PY%
  exit /b 1
)

echo [setup] upgrading pip
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

if exist "requirements.txt" (
  echo [setup] installing requirements.txt
  "%VENV_PY%" -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)

echo [setup] installing project in editable mode
"%VENV_PY%" -m pip install -e .
if errorlevel 1 exit /b 1

if not exist ".env" (
  if exist ".env.example" (
    echo [setup] creating .env from .env.example
    copy /Y ".env.example" ".env" >nul
  ) else (
    type nul > ".env"
  )
)

findstr /b /c:"NVIDIA_API_KEY=" .env >nul || echo NVIDIA_API_KEY=>>.env
findstr /b /c:"NVIDIA_BASE_URL=" .env >nul || echo NVIDIA_BASE_URL=https://integrate.api.nvidia.com>>.env
findstr /b /c:"OPENAI_API_KEY=" .env >nul || echo OPENAI_API_KEY=>>.env

echo.
echo [done] Collection Agent setup complete.
echo.
echo Next steps:
echo 1) Add your API key(s) to %ROOT_DIR%.env
echo 2) Activate venv:
echo    .venv\Scripts\activate
echo 3) Run UI:
echo    python -m agents.collection_agent.ui.server
echo 4) Open:
echo    http://127.0.0.1:8060/

exit /b 0
