# Public demo from this machine: starts the app and a free Cloudflare quick tunnel.
# The tunnel prints a https://<random>.trycloudflare.com URL; it works while this window stays open.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot)

# The link is public, so upload/delete must require the admin token (paste it into the UI token field).
$m = Select-String -Path .env -Pattern '^DEMO_ADMIN_TOKEN=(.+)$'
if (-not $m) { throw "Set DEMO_ADMIN_TOKEN in .env first: the demo is public and write endpoints must be locked" }
$env:ADMIN_TOKEN = $m.Matches[0].Groups[1].Value.Trim()
$env:TRUST_PROXY = "1"  # rate-limit by the visitor's IP from X-Forwarded-For, not the tunnel's

$cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cf) { $cf = "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe" }
if (-not (Test-Path $cf)) { throw "cloudflared not found: winget install --id Cloudflare.cloudflared" }

$app = Start-Process .venv\Scripts\python.exe -ArgumentList "-m", "uvicorn", "app.main:app", "--port", "8000" -PassThru -NoNewWindow
try {
    & $cf tunnel --no-autoupdate --url http://localhost:8000
} finally {
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
}
