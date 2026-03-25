@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  Skyulf - One-Click Starter for Windows
REM  This script sets up and runs Skyulf with zero configuration.
REM  The window stays open so you can always see what happened.
REM ============================================================

REM --- Always cd to the folder where start.bat lives ---
cd /d "%~dp0"

echo.
echo  ============================
echo   Skyulf - Quick Start
echo  ============================
echo.
echo  Working directory: %CD%
echo.

REM ---------------------------------------------------------------
REM  Step 1: Find the BEST Python (prefer 3.12 > 3.11 > 3.10)
REM  Uses the "py" launcher to pick a specific version even if
REM  the default "python" is 3.13+.
REM ---------------------------------------------------------------
set "PYTHON_CMD="
set "PY_FULL_VER="

REM --- Strategy A: Use "py -3.X" launcher to pick exact version ---
REM     This works even if "python" on PATH is 3.13
for %%V in (3.12 3.11 3.10) do (
    if not defined PYTHON_CMD (
        py -%%V --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=py -%%V"
            for /f "tokens=2 delims= " %%A in ('py -%%V --version 2^>^&1') do set "PY_FULL_VER=%%A"
        )
    )
)

REM --- Strategy B: Check common install paths directly ---
if not defined PYTHON_CMD (
    for %%P in (
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Python310\python.exe"
    ) do (
        if not defined PYTHON_CMD (
            if exist %%P (
                set "PYTHON_CMD=%%~P"
                for /f "tokens=2 delims= " %%A in ('"%%~P" --version 2^>^&1') do set "PY_FULL_VER=%%A"
            )
        )
    )
)

REM --- Strategy C: Fall back to whatever "python" / "py" / "python3" is ---
if not defined PYTHON_CMD (
    for %%C in (python py python3) do (
        if not defined PYTHON_CMD (
            %%C --version >nul 2>&1
            if !errorlevel! equ 0 (
                set "PYTHON_CMD=%%C"
                for /f "tokens=2 delims= " %%A in ('%%C --version 2^>^&1') do set "PY_FULL_VER=%%A"
            )
        )
    )
)

REM ---------------------------------------------------------------
REM  Python not found — show clear instructions
REM ---------------------------------------------------------------
if not defined PYTHON_CMD (
    echo.
    echo  =====================================================
    echo   [ERROR] Python was not found on this computer.
    echo  =====================================================
    echo.
    echo  Skyulf needs Python 3.10 or newer to run.
    echo.
    echo  HOW TO FIX:
    echo.
    echo    1. Download Python from: https://www.python.org/downloads/
    echo    2. Run the installer
    echo    3. IMPORTANT: Check the box "Add Python to PATH" at the
    echo       bottom of the installer (this is the most common mistake^)
    echo    4. After installing, close this window and double-click
    echo       start.bat again
    echo.
    echo  If Python IS installed but this still fails:
    echo    - Open a new terminal and type: python --version
    echo    - If that doesn't work, Python is not in your PATH.
    echo    - Re-run the Python installer and choose "Modify" then
    echo      check "Add Python to environment variables".
    echo.
    set /p "OPEN_BROWSER=Open the Python download page now? [Y/n]: "
    if /i "!OPEN_BROWSER!" neq "n" (
        start https://www.python.org/downloads/
    )
    pause
    exit /b 1
)

REM --- Show what we found ---
echo [OK] Found Python %PY_FULL_VER% (using: %PYTHON_CMD%^)

REM ---------------------------------------------------------------
REM  Step 1b: Validate version
REM ---------------------------------------------------------------
for /f "tokens=1,2 delims=." %%A in ("%PY_FULL_VER%") do (
    set "PY_MAJOR=%%A"
    set "PY_MINOR=%%B"
)

if %PY_MAJOR% neq 3 (
    echo [ERROR] Python 3 is required. You have Python %PY_MAJOR%.
    pause
    exit /b 1
)

if %PY_MINOR% lss 10 (
    echo.
    echo  [ERROR] Python %PY_FULL_VER% is too old.
    echo  Skyulf requires Python 3.10, 3.11, or 3.12.
    echo  Download from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM --- Skip version warning if Python is 3.10-3.12 ---
if %PY_MINOR% leq 12 goto :version_ok

echo.
echo  =====================================================
echo   [WARNING] Python %PY_FULL_VER% detected.
echo  =====================================================
echo.
echo  Skyulf is tested with Python 3.10-3.12.
echo  Some dependencies (numba, llvmlite) may not work on 3.%PY_MINOR%.
echo.
echo  Options:
echo    [1] Auto-install Python 3.12 side-by-side (recommended)
echo    [2] Continue with Python %PY_FULL_VER% anyway (may fail)
echo    [3] Exit
echo.
set /p "PY_CHOICE=Choose [1/2/3]: "

if "!PY_CHOICE!" equ "2" (
    echo [INFO] Continuing with Python %PY_FULL_VER% — some packages may fail.
    goto :version_ok
)
if "!PY_CHOICE!" neq "1" (
    pause
    exit /b 1
)

echo.
echo [SETUP] Downloading Python 3.12 installer...
echo         This will install alongside your existing Python %PY_FULL_VER%.
echo         Please wait, this may take a few minutes...
echo.
set "PY_INSTALLER=%TEMP%\python-3.12-installer.exe"

REM Delete any leftover partial download
if exist "!PY_INSTALLER!" del "!PY_INSTALLER!" >nul 2>&1

REM Use curl first (faster, built into Win10+), fall back to PowerShell
curl --version >nul 2>&1
if !errorlevel! equ 0 (
    curl -L --connect-timeout 15 --max-time 300 --retry 3 --retry-delay 5 -o "!PY_INSTALLER!" "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
) else (
    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile '!PY_INSTALLER!' -TimeoutSec 300 }"
)

REM Verify the file actually downloaded (should be ~25MB+)
if not exist "!PY_INSTALLER!" (
    echo.
    echo [ERROR] Download failed — file not found.
    goto :download_failed
)
for %%F in ("!PY_INSTALLER!") do set "DLSIZE=%%~zF"
if !DLSIZE! lss 10000000 (
    echo.
    echo [ERROR] Download incomplete — file is only !DLSIZE! bytes.
    del "!PY_INSTALLER!" >nul 2>&1
    goto :download_failed
)

echo [OK] Download complete (!DLSIZE! bytes).
echo.
echo [SETUP] Installing Python 3.12 (this takes about 1 minute)...
echo         Installing for current user only, won't affect your Python %PY_FULL_VER%.
echo.
"!PY_INSTALLER!" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=1 Include_test=0
if !errorlevel! neq 0 (
    echo [ERROR] Installation failed. Try downloading manually:
    echo         https://www.python.org/downloads/release/python-31210/
    pause
    exit /b 1
)
del "!PY_INSTALLER!" >nul 2>&1
echo [OK] Python 3.12 installed successfully!
echo.

REM Re-detect Python — should now find 3.12 via py launcher
set "PYTHON_CMD="
py -3.12 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=py -3.12"
    for /f "tokens=2 delims= " %%A in ('py -3.12 --version 2^>^&1') do set "PY_FULL_VER=%%A"
    for /f "tokens=1,2 delims=." %%A in ("!PY_FULL_VER!") do (
        set "PY_MAJOR=%%A"
        set "PY_MINOR=%%B"
    )
    echo [OK] Now using Python !PY_FULL_VER!
) else (
    echo [WARN] py -3.12 not found after install. Continuing with %PY_FULL_VER%.
)

REM Delete old venv if it was created with the wrong version
if exist ".venv\Scripts\activate.bat" (
    echo [SETUP] Removing old virtual environment...
    rmdir /s /q .venv >nul 2>&1
)
goto :version_ok

:download_failed
echo.
echo  =====================================================
echo   Download failed or timed out.
echo  =====================================================
echo.
echo  Your internet may be slow or python.org is unreachable.
echo.
echo  MANUAL FIX (takes 2 minutes):
echo    1. Open in browser: https://www.python.org/downloads/release/python-31210/
echo    2. Download "Windows installer (64-bit)"
echo    3. Run the installer (default settings are fine)
echo    4. Close this window, delete .venv folder, re-run start.bat
echo.
pause
exit /b 1

:version_ok

REM ---------------------------------------------------------------
REM  Step 2: Create .env if missing
REM ---------------------------------------------------------------
if not exist ".env" (
    echo [SETUP] Creating .env with safe defaults...
    (
        echo # Auto-generated by start.bat
        echo USE_CELERY=false
        echo DB_TYPE=sqlite
        echo DEBUG=true
    ) > .env
    echo [OK] .env created (SQLite, no Redis required^)
)

REM ---------------------------------------------------------------
REM  Step 3: Create or reuse virtual environment
REM  - If .venv doesn't exist, create it
REM  - If .venv exists but was made with a different Python (e.g.
REM    3.13 but we now prefer 3.12), recreate it automatically
REM ---------------------------------------------------------------
set "NEED_NEW_VENV=0"

if not exist ".venv\Scripts\activate.bat" (
    set "NEED_NEW_VENV=1"
) else (
    REM Check what Python version the existing venv uses
    for /f "tokens=2 delims= " %%A in ('".venv\Scripts\python.exe" --version 2^>^&1') do set "VENV_PY_VER=%%A"
    for /f "tokens=1,2 delims=." %%A in ("!VENV_PY_VER!") do set "VENV_MINOR=%%B"

    if "!VENV_MINOR!" neq "%PY_MINOR%" (
        echo [INFO] Existing .venv uses Python !VENV_PY_VER! but we selected %PY_FULL_VER%.
        echo        Recreating virtual environment...
        rmdir /s /q .venv >nul 2>&1
        set "NEED_NEW_VENV=1"
    ) else (
        echo [OK] Reusing existing virtual environment (Python !VENV_PY_VER!^).
    )
)

if "!NEED_NEW_VENV!" equ "1" (
    echo [SETUP] Creating virtual environment with Python %PY_FULL_VER%...
    %PYTHON_CMD% -m venv .venv
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Failed to create virtual environment.
        echo         Try running manually: %PYTHON_CMD% -m venv .venv
        echo.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM ---------------------------------------------------------------
REM  Step 4: Activate venv
REM ---------------------------------------------------------------
echo [SETUP] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment active.

REM ---------------------------------------------------------------
REM  Step 5: Install dependencies if needed
REM  (After activation, "python" and "pip" point to the venv)
REM ---------------------------------------------------------------
python -c "import uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    REM --- Check internet connectivity before attempting install ---
    REM     Retries up to 5 times with 10s wait (handles transient drops)
    echo [CHECK] Verifying internet connection...
    set "INET_OK=0"
    for /L %%R in (1,1,5) do (
        if !INET_OK! equ 0 (
            ping -n 1 -w 3000 pypi.org >nul 2>&1
            if !errorlevel! equ 0 (
                set "INET_OK=1"
            ) else (
                if %%R lss 5 (
                    echo [RETRY] Attempt %%R/5 failed — waiting 10 seconds...
                    timeout /t 10 /nobreak >nul
                )
            )
        )
    )
    if !INET_OK! equ 0 (
        echo.
        echo  =====================================================
        echo   [ERROR] No internet connection after 5 attempts.
        echo  =====================================================
        echo.
        echo  Skyulf needs to download packages from pypi.org on
        echo  first run, but your computer cannot reach the internet.
        echo.
        echo  COMMON CAUSES:
        echo    - Wi-Fi or Ethernet is disconnected
        echo    - VPN is blocking the connection (try disconnecting^)
        echo    - Corporate firewall/proxy is blocking pip
        echo    - DNS is not resolving (try: ipconfig /flushdns^)
        echo.
        echo  FIXES TO TRY:
        echo    1. Check your browser can open https://pypi.org
        echo    2. If behind a proxy, run manually:
        echo       .venv\Scripts\activate.bat
        echo       pip install -r requirements-fastapi.txt --proxy http://YOUR_PROXY:PORT
        echo    3. If DNS fails, try:  ipconfig /flushdns
        echo    4. If on VPN, disconnect and retry
        echo.
        pause
        exit /b 1
    )
    echo [OK] Internet connection verified.
    echo.
    echo [SETUP] Installing dependencies (first run only; may take 3-5 minutes^)...
    echo         Please wait...
    echo.
    python -m pip install --upgrade pip -q
    python -m pip install --prefer-binary -r requirements-fastapi.txt
    if %errorlevel% neq 0 (
        echo.
        echo  =====================================================
        echo   [ERROR] Dependency installation failed.
        echo  =====================================================
        echo.
        echo  Try running these commands manually:
        echo    .venv\Scripts\activate.bat
        echo    pip install -r requirements-fastapi.txt
        echo.
        echo  If you see "connection" errors, check your internet
        echo  or proxy settings (see above^).
        echo.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Dependencies installed successfully.
)

REM ---------------------------------------------------------------
REM  Step 6: Launch!
REM ---------------------------------------------------------------
echo.
echo  =====================================================
echo   [START] Launching Skyulf...
echo  =====================================================
echo.
echo   API docs:  http://127.0.0.1:8000/docs
echo   Health:    http://127.0.0.1:8000/health
echo.
echo   Press Ctrl+C to stop the server.
echo.

python run_skyulf.py

REM ---------------------------------------------------------------
REM  If we get here, the server stopped (crashed or Ctrl+C)
REM  Keep the window open so the user can read the error
REM ---------------------------------------------------------------
echo.
echo  =====================================================
echo   Server stopped. See any errors above.
echo  =====================================================
echo.
pause
