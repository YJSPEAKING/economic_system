@echo off
chcp 65001 >nul

echo Checking cpolar public URL...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$ErrorActionPreference='Stop';" ^
"try {" ^
"  $data = Invoke-RestMethod -Uri 'http://127.0.0.1:4042/api/tunnels' -TimeoutSec 5;" ^
"  $urls = @($data.tunnels | ForEach-Object { $_.public_url } | Where-Object { $_ -like 'https://*' });" ^
"  if (-not $urls -or $urls.Count -eq 0) { $urls = @($data.tunnels | ForEach-Object { $_.public_url } | Where-Object { $_ }); }" ^
"  if (-not $urls -or $urls.Count -eq 0) { Write-Host 'No public URL found. Make sure start_cpolar_8501.bat is still running.'; exit 1; }" ^
"  Write-Host 'Send this URL to participants:';" ^
"  Write-Host $urls[0];" ^
"  Set-Clipboard $urls[0];" ^
"  Write-Host '';" ^
"  Write-Host 'The URL has also been copied to your clipboard.';" ^
"} catch {" ^
"  Write-Host 'Could not read cpolar URL from http://127.0.0.1:4042/api/tunnels';" ^
"  Write-Host 'Please make sure start_cpolar_8501.bat is running, then try again.';" ^
"  exit 1;" ^
"}"

echo.
pause
