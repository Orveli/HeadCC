@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Siirry tähän kansioon
cd /d "%~dp0"

REM Tarkista venv
if not exist ".venv\Scripts\python.exe" (
  echo [ERR] Venv ei loydy: .venv\Scripts\python.exe
  echo Aja ensin: setup_venv.bat
  exit /b 1
)

REM Aktivoi venv
call ".venv\Scripts\run.bat"

REM Vähennä TFLite-lokia (valinnainen)
set TF_CPP_MIN_LOG_LEVEL=2
set GLOG_minloglevel=2

REM Aja skripti. Välittää komentoriviargumentit edelleen.
python -u "HeadCC.py" %*
set EXITCODE=%ERRORLEVEL%

REM Sulje
deactivate >nul 2>&1
endlocal & exit /b %EXITCODE%
