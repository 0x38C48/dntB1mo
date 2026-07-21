$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$port = if ($env:CHATBOT_PORT) { [int]$env:CHATBOT_PORT } else { 8765 }

$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) {
  $conn | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force
  }
}

if (-not $env:SOPHNET_API_KEY) {
  Write-Host "SOPHNET_API_KEY is not set; starting in local fallback mode."
}

if (-not $env:SOPHNET_MODEL) {
  $env:SOPHNET_MODEL = "DeepSeek-V4-Flash"
}

Start-Process `
  -FilePath "python" `
  -ArgumentList "app.py" `
  -WorkingDirectory $root `
  -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $root "server.log") `
  -RedirectStandardError (Join-Path $root "server.err.log")

Start-Sleep -Seconds 3
Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/status" | ConvertTo-Json -Depth 5
