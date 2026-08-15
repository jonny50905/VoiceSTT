# VoiceSTT one-command entry point.
# Usage: .\voicestt.ps1 <audio-file> [extra pipeline args...]
#   e.g. .\voicestt.ps1 "D:\rec\meeting.m4a"
#        .\voicestt.ps1 "D:\rec\talk.m4a" --stage merge --speakers 1
# First run downloads models (~500MB) and creates a venv; later runs skip both.
# Keep this file ASCII-only.
param(
    [Parameter(Mandatory = $true)][string]$Audio,
    [Parameter(ValueFromRemainingArguments = $true)]$Rest
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

pwsh -File (Join-Path $here "scripts\setup.ps1")
if ($LASTEXITCODE -ne 0) { throw "setup failed" }

$py = Join-Path $env:LOCALAPPDATA "meeting-minutes\venv\Scripts\python.exe"
& $py (Join-Path $here "scripts\pipeline.py") $Audio @Rest
exit $LASTEXITCODE
