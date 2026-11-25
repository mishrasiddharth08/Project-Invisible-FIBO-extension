@echo off
setlocal
set "FIBO_PYTHON=%~dp0..\..\venv\Scripts\python.exe"
if defined FORGE_PYTHON set "FIBO_PYTHON=%FORGE_PYTHON%"
if not exist "%FIBO_PYTHON%" (
  echo Forge Python not found. Set FORGE_PYTHON to the python.exe used by Forge.
  exit /b 1
)
"%FIBO_PYTHON%" "%~dp0setup_fibo2.py" %*
set "FIBO_RESULT=%ERRORLEVEL%"
if not "%FIBO_RESULT%"=="0" echo FIBO setup failed. Review the error above.
exit /b %FIBO_RESULT%
