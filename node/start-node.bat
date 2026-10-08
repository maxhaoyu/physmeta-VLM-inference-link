@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0inference-node.json" (
  echo.
  echo [无法启动] 当前目录不是已配置的 PhysMeta 节点目录。
  echo 缺少文件：%~dp0inference-node.json
  echo.
  echo 如果你刚解压了更新包，请返回解压目录运行“升级现有节点.bat”，
  echo 并指向原节点目录。不要在公开更新包中新建或复制生产 token。
  echo 新节点必须先由管理员注册并生成本机配置。
  echo.
  pause
  exit /b 1
)
if defined PHYS_META_PYTHON (
  "%PHYS_META_PYTHON%" "%~dp0launch_node.py" %*
  goto finished
)
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0launch_node.py" %*
  goto finished
)
where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%~dp0launch_node.py" %*
  goto finished
)
where python >nul 2>nul
if not errorlevel 1 (
  python "%~dp0launch_node.py" %*
  goto finished
)
echo Python not found. Set PHYS_META_PYTHON to your existing CUDA Python executable.
pause
exit /b 1
:finished
set "NODE_EXIT=%ERRORLEVEL%"
if not "%NODE_EXIT%"=="0" pause
exit /b %NODE_EXIT%
