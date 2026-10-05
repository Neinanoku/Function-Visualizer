@echo off

set PROJ=%~dp0
set PYTHON=%PROJ%.venv\Scripts\python.exe

echo.
echo === FuncVisualizer Builder ===
echo.

if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    pause
    exit /b 1
)

"%PYTHON%" --version
echo.

echo [1/3] Installing dependencies...
"%PYTHON%" -m pip install matplotlib numpy sympy scipy ttkbootstrap pillow pyinstaller -q
echo Done.
echo.

echo [2/3] Generating spec file...
"%PYTHON%" generate_spec.py
if errorlevel 1 (
    echo ERROR: Failed to generate spec
    pause
    exit /b 1
)

echo [3/3] Building .exe...
"%PYTHON%" -m PyInstaller FuncVisualizer.spec

if errorlevel 1 (
    echo ERROR: PyInstaller failed
    pause
    exit /b 1
)

echo.
echo === DONE! ===
echo Output: dist\FuncVisualizer.exe
echo.
pause
