@echo off
setlocal
set "ANIMA_PROJECT_ROOT=%~dp0."
"%ANIMA_PROJECT_ROOT%\.runtime-build\runtimes\core\python.exe" -B -I "%ANIMA_PROJECT_ROOT%\packaging\scripts\ocr_component.py" install --app-root "%ANIMA_PROJECT_ROOT%" --mode cpu --model-root "%ANIMA_PROJECT_ROOT%\ocr-model-archives"
exit /b %ERRORLEVEL%
