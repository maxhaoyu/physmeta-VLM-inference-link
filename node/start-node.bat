@echo off
setlocal enabledelayedexpansion
rem ============================================================
rem  PhysMeta inference node launcher (double-click to start)
rem  Keeps a persistent process that long-polls the queue and
rem  runs YOLO + VLM locally on the NVIDIA GPU.
rem
rem  用法：把本文件与 agent.py、inference-node.json 放在同一目录，
rem  改下面 PY 指向本机 venv 的 python.exe，然后双击即可。
rem  （agent.py / inference-node.json 用 %~dp0 自动定位，无需再改）
rem ============================================================

rem 改成本机 venv 的 python.exe（需已装 CUDA 版 torch + requirements-windows.txt）
set "PY=C:\PhysMetaInference\.venv\Scripts\python.exe"

set "AGENT=%~dp0agent.py"
set "CFG=%~dp0inference-node.json"

if not exist "%PY%" (
    echo [ERROR] Python venv not found:
    echo     %PY%
    echo     Edit this file and set PY to your venv python.exe.
    goto :fail
)
if not exist "%AGENT%" (
    echo [ERROR] agent.py not found:
    echo     %AGENT%
    goto :fail
)
if not exist "%CFG%" (
    echo [ERROR] Config file not found:
    echo     %CFG%
    goto :fail
)

rem Warn if the token placeholder is still present (token not filled yet)
findstr /C:"在这里粘贴" "%CFG%" >nul 2>&1
if %errorlevel%==0 (
    echo [ERROR] INFERENCE_NODE_TOKEN is not filled yet.
    echo     Edit the file: %CFG%
    echo     Put the token issued by server deployment into the "token" field,
    echo     then double-click this file again.
    goto :fail
)

echo ============================================================
echo   PhysMeta inference node starting...
echo   Config: %CFG%
echo   Monitor: http://127.0.0.1:8901  (opens automatically)
echo   Keep this window open. Press Ctrl+C to stop.
echo ============================================================
echo.
set "restarts=0"
:loop
echo [node] starting (restart #%restarts%)...
"%PY%" "%AGENT%" --config "%CFG%"
set "code=%ERRORLEVEL%"
echo.
echo [node] process exited with code %code%.

if "%code%"=="2" (
    set /a restarts+=1
    if !restarts! gtr 10 (
        echo [node] crashed over 10 times, auto-restart stopped. Check the window above.
        goto :done
    )
    echo [node] crashed, restarting in 5s (attempt !restarts!)...
    timeout /t 5 /nobreak >nul
    goto :loop
)

if "%code%"=="0" (
    echo [node] exited normally.
) else (
    echo [node] exited with a config/auth error, no auto-restart. Check the window above.
)
goto :done

:done
pause
exit /b 0

:fail
echo.
pause
exit /b 1
