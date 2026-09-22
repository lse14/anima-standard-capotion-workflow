@echo off
setlocal
call "%~dp0Install-WebUI.bat" -IncludeQualityStack %*
exit /b %ERRORLEVEL%
