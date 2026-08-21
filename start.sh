#!/usr/bin/env bash
# ============================================================
#  Skyulf - One-Click Starter for macOS / Linux
#  This script sets up and runs Skyulf with zero configuration.
#  Same logic as start.bat — prefers Python 3.12, auto-recovers.
# ============================================================

set -e

# --- Always cd to the folder where start.sh lives ---
cd "$(dirname "$0")"

echo ""
echo " ============================"
echo "  Skyulf - Quick Start"
echo " ============================"
echo ""
echo " Working directory: $(pwd)"
echo ""

# ---------------------------------------------------------------
#  Step 1: Find the BEST Python (prefer 3.12 > 3.11 > 3.10)
# ---------------------------------------------------------------
PYTHON_CMD=""
PY_MINOR=""

# Strategy A: Try specific versions first (best compatibility)
for ver in 3.12 3.11 3.10; do
    for cmd in "python$ver" "python${ver%%.*}"; do
        if command -v "$cmd" &> /dev/null; then
            actual=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
            if [ "$actual" = "$ver" ]; then
                PYTHON_CMD="$cmd"
                PY_MINOR="${ver#3.}"
                break 2
            fi
        fi
    done
done

# Strategy B: Fall back to any python3/python and check version
if [ -z "$PYTHON_CMD" ]; then
    for cmd in python3 python; do
        if command -v "$cmd" &> /dev/null; then
            actual=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
            major=$(echo "$actual" | cut -d. -f1)
            minor=$(echo "$actual" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
                PYTHON_CMD="$cmd"
                PY_MINOR="$minor"
                break
            fi
        fi
    done
fi

# ---------------------------------------------------------------
#  Python not found
# ---------------------------------------------------------------
if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo " ====================================================="
    echo "  [ERROR] Python 3.10+ was not found on this computer."
    echo " ====================================================="
    echo ""
    echo " HOW TO FIX:"
    echo ""
    if [ "$(uname)" = "Darwin" ]; then
        echo "   brew install python@3.12"
    elif command -v apt &> /dev/null; then
        echo "   sudo apt update && sudo apt install python3.12 python3.12-venv"
    elif command -v dnf &> /dev/null; then
        echo "   sudo dnf install python3.12"
    elif command -v pacman &> /dev/null; then
        echo "   sudo pacman -S python"
    else
        echo "   Install Python 3.12 from: https://www.python.org/downloads/"
    fi
    echo ""
    echo " After installing, run this script again: ./start.sh"
    echo ""
    exit 1
fi

PY_FULL_VER=$("$PYTHON_CMD" --version 2>&1)
echo "[OK] Found $PY_FULL_VER (using: $PYTHON_CMD)"

# ---------------------------------------------------------------
#  Step 1b: Warn if Python > 3.12 and offer to install 3.12
# ---------------------------------------------------------------
if [ "$PY_MINOR" -gt 12 ]; then
    echo ""
    echo " ====================================================="
    echo "  [WARNING] Python 3.$PY_MINOR detected."
    echo " ====================================================="
    echo ""
    echo " Skyulf is tested with Python 3.10-3.12."
    echo " Some dependencies (numba, llvmlite) may not work on 3.$PY_MINOR."
    echo ""

    # Offer auto-install based on OS
    CAN_AUTO_INSTALL=false
    INSTALL_CMD=""
    if [ "$(uname)" = "Darwin" ] && command -v brew &> /dev/null; then
        CAN_AUTO_INSTALL=true
        INSTALL_CMD="brew install python@3.12"
    elif command -v apt &> /dev/null; then
        CAN_AUTO_INSTALL=true
        INSTALL_CMD="sudo apt update && sudo apt install -y python3.12 python3.12-venv"
    elif command -v dnf &> /dev/null; then
        CAN_AUTO_INSTALL=true
        INSTALL_CMD="sudo dnf install -y python3.12"
    elif command -v pacman &> /dev/null; then
        CAN_AUTO_INSTALL=true
        INSTALL_CMD="sudo pacman -S --noconfirm python"
    fi

    if [ "$CAN_AUTO_INSTALL" = true ]; then
        echo " Options:"
        echo "   [1] Auto-install Python 3.12 side-by-side (recommended)"
        echo "   [2] Continue with Python 3.$PY_MINOR anyway (may fail)"
        echo "   [3] Exit"
        echo ""
        read -rp " Choose [1/2/3]: " PY_CHOICE

        if [ "$PY_CHOICE" = "1" ]; then
            echo ""
            echo "[SETUP] Installing Python 3.12..."
            echo "        Running: $INSTALL_CMD"
            echo "        This may take a few minutes and may ask for your password."
            echo ""

            # Temporarily disable set -e so we can handle failure
            set +e
            eval "$INSTALL_CMD"
            INSTALL_EXIT=$?
            set -e

            if [ $INSTALL_EXIT -ne 0 ]; then
                echo ""
                echo " ====================================================="
                echo "  [ERROR] Installation failed (exit code: $INSTALL_EXIT)"
                echo " ====================================================="
                echo ""
                echo " MANUAL FIX:"
                if [ "$(uname)" = "Darwin" ]; then
                    echo "   brew install python@3.12"
                    echo "   (If brew itself fails, try: xcode-select --install first)"
                elif command -v apt &> /dev/null; then
                    echo "   sudo apt update && sudo apt install python3.12 python3.12-venv"
                    echo "   (If python3.12 is not found, try adding deadsnakes PPA:)"
                    echo "   sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update"
                elif command -v dnf &> /dev/null; then
                    echo "   sudo dnf install python3.12"
                fi
                echo ""
                echo " After installing, run this script again: ./start.sh"
                echo ""
                exit 1
            fi

            # Verify it actually works
            if command -v python3.12 &> /dev/null; then
                PYTHON_CMD="python3.12"
                PY_MINOR="12"
                echo ""
                echo "[OK] Now using $(python3.12 --version 2>&1)"
                # Remove old venv
                if [ -d ".venv" ]; then
                    echo "[SETUP] Removing old virtual environment..."
                    rm -rf .venv
                fi
            else
                echo ""
                echo "[WARN] python3.12 command not found after install."
                # On macOS brew, python may be at a different path
                if [ "$(uname)" = "Darwin" ]; then
                    BREW_PY="$(brew --prefix python@3.12 2>/dev/null)/bin/python3.12"
                    if [ -x "$BREW_PY" ]; then
                        PYTHON_CMD="$BREW_PY"
                        PY_MINOR="12"
                        echo "[OK] Found at: $BREW_PY"
                        if [ -d ".venv" ]; then
                            rm -rf .venv
                        fi
                    else
                        echo "[WARN] Continuing with Python 3.$PY_MINOR."
                    fi
                else
                    echo "[WARN] Continuing with Python 3.$PY_MINOR."
                fi
            fi
        elif [ "$PY_CHOICE" = "2" ]; then
            echo "[INFO] Continuing with Python 3.$PY_MINOR — some packages may fail."
        else
            exit 1
        fi
    else
        echo " No supported package manager found (brew/apt/dnf/pacman)."
        echo " Install Python 3.12 manually from: https://www.python.org/downloads/"
        echo ""
        read -rp " Continue with Python 3.$PY_MINOR anyway? [y/N]: " CONT
        if [ "$CONT" != "y" ] && [ "$CONT" != "Y" ]; then
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------
#  Step 2: Create .env if missing
# ---------------------------------------------------------------
if [ ! -f ".env" ]; then
    echo "[SETUP] Creating .env with safe defaults..."
    cat > .env <<EOF
# Auto-generated by start.sh
USE_CELERY=false
DB_TYPE=sqlite
DEBUG=true
EOF
    echo "[OK] .env created (SQLite, no Redis required)"
fi

# ---------------------------------------------------------------
#  Step 3: Create or reuse virtual environment
# ---------------------------------------------------------------
NEED_NEW_VENV=false

if [ ! -d ".venv" ] || [ ! -f ".venv/bin/activate" ]; then
    NEED_NEW_VENV=true
else
    # Check if existing venv matches selected Python version
    VENV_MINOR=$(.venv/bin/python -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    if [ "$VENV_MINOR" != "$PY_MINOR" ]; then
        echo "[INFO] Existing .venv uses Python 3.$VENV_MINOR but we selected 3.$PY_MINOR."
        echo "       Recreating virtual environment..."
        rm -rf .venv
        NEED_NEW_VENV=true
    else
        echo "[OK] Reusing existing virtual environment (Python 3.$VENV_MINOR)."
    fi
fi

if [ "$NEED_NEW_VENV" = true ]; then
    echo "[SETUP] Creating virtual environment with $("$PYTHON_CMD" --version 2>&1)..."
    "$PYTHON_CMD" -m venv .venv || {
        echo ""
        echo "[ERROR] Failed to create virtual environment."
        if command -v apt &> /dev/null; then
            echo "        Try: sudo apt install python3.$PY_MINOR-venv"
        fi
        echo ""
        exit 1
    }
    echo "[OK] Virtual environment created."
fi

# ---------------------------------------------------------------
#  Step 4: Activate venv
# ---------------------------------------------------------------
echo "[SETUP] Activating virtual environment..."
source .venv/bin/activate
echo "[OK] Virtual environment active."

# ---------------------------------------------------------------
#  Step 5: Install dependencies if needed
# ---------------------------------------------------------------
if ! python -c "import uvicorn" 2>/dev/null; then
    # Check internet connectivity — retry up to 5 times with 10s wait
    echo "[CHECK] Verifying internet connection..."
    INET_OK=false
    for ATTEMPT in 1 2 3 4 5; do
        if ping -c 1 -W 3 pypi.org &>/dev/null || curl -s --connect-timeout 3 https://pypi.org >/dev/null 2>&1; then
            INET_OK=true
            break
        fi
        if [ "$ATTEMPT" -lt 5 ]; then
            echo "[RETRY] Attempt $ATTEMPT/5 failed — waiting 10 seconds..."
            sleep 10
        fi
    done

    if [ "$INET_OK" = false ]; then
        echo ""
        echo " ====================================================="
        echo "  [ERROR] No internet connection after 5 attempts."
        echo " ====================================================="
        echo ""
        echo " Skyulf needs to download packages from pypi.org."
        echo " Check your network, VPN, or proxy settings."
        echo ""
        exit 1
    fi
    echo "[OK] Internet connection verified."
    echo ""
    echo "[SETUP] Installing dependencies (first run only; may take 3-5 minutes)..."
    echo "        Please wait..."
    echo ""
    pip install --upgrade pip -q
    pip install --prefer-binary -r requirements-fastapi.txt
    echo ""
    echo "[OK] Dependencies installed successfully."
fi

# ---------------------------------------------------------------
#  Step 6: Launch!
# ---------------------------------------------------------------
echo ""
echo " ====================================================="
echo "  [START] Launching Skyulf..."
echo " ====================================================="
echo ""
echo "  API docs:  http://127.0.0.1:8000/docs"
echo "  Health:    http://127.0.0.1:8000/health"
echo ""
echo "  Press Ctrl+C to stop the server."
echo ""

python run_skyulf.py
