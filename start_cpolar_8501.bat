@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Starting cpolar tunnel...
echo Local web address: http://127.0.0.1:8501
echo.
echo If the public URL is not shown in this window, open:
echo http://127.0.0.1:4042
echo Then copy the https public URL from the cpolar tunnel page and send it to participants.
echo.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:4042'"

cpolar http 8501

echo.
echo cpolar has stopped.
pause
