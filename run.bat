@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" goto START_APP

where python >nul 2>nul
if errorlevel 1 goto NO_PYTHON

echo First launch: creating the local Python environment...
python -m venv "%~dp0.venv"
if errorlevel 1 goto INSTALL_FAILED

echo Installing required packages. This may take a few minutes...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto INSTALL_FAILED
"%VENV_PYTHON%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto INSTALL_FAILED

:START_APP
echo Starting Genshin Avatar Recognizer...
echo Open http://localhost:8501 if the browser does not open automatically.
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
"%VENV_PYTHON%" -m streamlit run "%~dp0app.py" --browser.gatherUsageStats=false
goto END

:NO_PYTHON
echo Python was not found.
echo Please install Python 3.14 and enable "Add Python to PATH".
pause
goto END

:INSTALL_FAILED
echo Installation failed. Please check the messages above.
pause

:END
endlocal
