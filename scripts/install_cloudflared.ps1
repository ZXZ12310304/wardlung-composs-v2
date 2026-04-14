param(
    [string]$DestDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($DestDir)) {
    $DestDir = Join-Path $PSScriptRoot "..\tools"
}

$DestDir = [System.IO.Path]::GetFullPath($DestDir)
New-Item -ItemType Directory -Path $DestDir -Force | Out-Null

$url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
$destFile = Join-Path $DestDir "cloudflared.exe"

Write-Host "[install] Downloading cloudflared from $url"
Invoke-WebRequest -Uri $url -OutFile $destFile -UseBasicParsing
Unblock-File -Path $destFile -ErrorAction SilentlyContinue

Write-Host "[install] Installed: $destFile"
& $destFile --version

Write-Host ""
Write-Host "[next] Run app with cloudflared tunnel:"
Write-Host '$env:PUBLIC_TUNNEL_PROVIDER="cloudflared"'
Write-Host '$env:ENABLE_PUBLIC_TUNNEL="1"'
Write-Host "python app.py"
