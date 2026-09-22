@echo off
setlocal
call "%~dp0Install-WebUI.bat" -IncludePolicy %*
exit /b %ERRORLEVEL%
