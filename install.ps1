# Install VoiceSTT as a Claude Code skill (one-click).
# Usage: .\install.ps1            (install skill + prepare venv/models now)
#        .\install.ps1 -SkipSetup (install skill only; env is built on first use)
# After install, open `claude` anywhere and hand it a recording - the
# `meeting-minutes` skill triggers automatically.
# Keep this file ASCII-only.
param([switch]$SkipSetup)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$dst = Join-Path $env:USERPROFILE ".claude\skills\meeting-minutes"
New-Item -ItemType Directory -Force (Join-Path $dst "scripts") | Out-Null
Copy-Item (Join-Path $here "SKILL.md") $dst -Force
Copy-Item (Join-Path $here "scripts\*") (Join-Path $dst "scripts") -Force
Write-Host "Skill installed to $dst"

if (-not $SkipSetup) {
    Write-Host "Preparing venv and models (idempotent, ~500MB on first run)..."
    pwsh -File (Join-Path $dst "scripts\setup.ps1")
    if ($LASTEXITCODE -ne 0) { throw "setup failed" }
}
Write-Host "DONE. Open 'claude' in any folder and give it a recording, e.g.:"
Write-Host '  claude "D:\rec\meeting.m4a 產逐字稿和會議紀錄"'
