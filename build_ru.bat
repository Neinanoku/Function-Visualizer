@echo off

set PROJ=%~dp0
set PYTHON=%PROJ%.venv\Scripts\python.exe
rem Работаем из папки проекта независимо от того, откуда запущен скрипт
cd /d "%PROJ%"

echo.
echo === FuncVisualizer Russian Builder ===
echo.

if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    pause
    exit /b 1
)

"%PYTHON%" --version
echo.

echo [1/3] Installing dependencies...
"%PYTHON%" -m pip install matplotlib numpy sympy scipy pillow pyinstaller -q
echo Done.
echo.

echo [2/3] Generating spec file...
"%PYTHON%" generate_spec_ru.py
if errorlevel 1 (
    echo ERROR: Failed to generate spec
    pause
    exit /b 1
)

echo [3/3] Building .exe...
"%PYTHON%" -m PyInstaller FuncVisualizer_ru.spec

if errorlevel 1 (
    echo ERROR: PyInstaller failed
    pause
    exit /b 1
)

echo.
echo === DONE! ===
echo Output: dist\FuncVisualizer_ru.exe
echo.
pause
