@echo off
setlocal
cd /d "%~dp0"
if defined PHYS_META_PYTHON (
  "%PHYS_META_PYTHON%" "%~dp0install_update.py" %*
  goto finished
)
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0install_update.py" %*
  goto finished
)
where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%~dp0install_update.py" %*
  goto finished
)
where python >nul 2>nul
if not errorlevel 1 (
  python "%~dp0install_update.py" %*
  goto finished
)
echo Python not found. Set PHYS_META_PYTHON to your existing CUDA Python executable.
pause
exit /b 1
:finished
set "NODE_EXIT=%ERRORLEVEL%"
if not "%NODE_EXIT%"=="0" pause
exit /b %NODE_EXIT%
