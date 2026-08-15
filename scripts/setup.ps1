# Idempotent bootstrap for the meeting-minutes pipeline.
# Creates %LOCALAPPDATA%\meeting-minutes\{venv,models}; skips anything already present.
# Keep this file ASCII-only (Windows PowerShell 5.1 misreads UTF-8 without BOM).
$ErrorActionPreference = "Stop"

$root   = Join-Path $env:LOCALAPPDATA "meeting-minutes"
$venv   = Join-Path $root "venv"
$models = Join-Path $root "models"
New-Item -ItemType Directory -Force $root, $models | Out-Null

if (-not (Test-Path "$venv\Scripts\python.exe")) {
    # Bare "python" on PATH may be Python 2.7 on this machine - probe for a real 3.10+.
    $basePy = $null
    foreach ($cand in @(@("py", "-3.12"), @("py", "-3"), @("python3"), @("python"))) {
        try {
            $v = & $cand[0] @($cand[1..($cand.Count)] | Where-Object { $_ }) -c "import sys; print(sys.version_info[0]*100 + sys.version_info[1])" 2>$null
            if ($LASTEXITCODE -eq 0 -and [int]$v -ge 310) { $basePy = $cand; break }
        } catch {}
    }
    if (-not $basePy) { throw "No Python >= 3.10 found (tried py -3.12 / py -3 / python3 / python)" }
    Write-Host "Creating venv with $($basePy -join ' ')..."
    & $basePy[0] @($basePy[1..($basePy.Count)] | Where-Object { $_ }) -m venv $venv
    if (-not (Test-Path "$venv\Scripts\python.exe")) { throw "venv creation failed" }
}
$py = "$venv\Scripts\python.exe"

# Versions validated together on 2026-07-29 (Python 3.12, RTX 4070 SUPER).
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet `
    faster-whisper==1.2.1 `
    opencc-python-reimplemented==0.1.7 `
    scikit-learn==1.9.0 `
    huggingface_hub==1.25.1 `
    nvidia-cublas-cu12==12.9.2.10 `
    nvidia-cudnn-cu12==9.24.0.43
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# sherpa-onnx: CUDA wheel on NVIDIA machines (diarize ~3.6x faster, validated
# bit-identical to CPU output on 2026-08-15), plain CPU wheel otherwise.
# The CUDA wheel needs cudart + cufft DLLs on top of cublas/cudnn above.
# NOTE: older installs used the split packages sherpa-onnx + sherpa-onnx-core
# sharing one lib dir - always remove BOTH before switching variants.
$hasGpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$wantSherpa = if ($hasGpu) { "1.13.5+cuda12.cudnn9" } else { "1.13.4" }
$haveSherpa = ""
try {
    $line = & $py -m pip show sherpa-onnx 2>$null | Select-String "^Version:"
    if ($line) { $haveSherpa = $line.ToString().Split(" ")[-1] }
} catch {}
if ($haveSherpa -ne $wantSherpa) {
    Write-Host "Installing sherpa-onnx $wantSherpa (was: '$haveSherpa')..."
    & $py -m pip uninstall --quiet -y sherpa-onnx sherpa-onnx-core 2>$null
    if ($hasGpu) {
        & $py -m pip install --quiet nvidia-cuda-runtime-cu12 nvidia-cufft-cu12
        if ($LASTEXITCODE -ne 0) { throw "pip install cuda runtime failed" }
    }
    & $py -m pip install --quiet "sherpa-onnx==$wantSherpa" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
    if ($LASTEXITCODE -ne 0) { throw "pip install sherpa-onnx failed" }
}

# Speaker embedding model MUST be eres2net...16k-common (200k-speaker corpus).
# The one in sherpa-onnx docs examples (eres2net_base_sv_zh-cn_3dspeaker_16k) is a known
# bad choice that over-segments (70 speakers from a 4-person meeting).
$downloads = @(
    @{ url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
       check = "$models\sherpa-onnx-pyannote-segmentation-3-0\model.onnx"; extract = $true },
    @{ url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_sv_zh-cn_16k-common.onnx"
       check = "$models\embed_eres2net_common.onnx"; extract = $false },
    @{ url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
       check = "$models\embed_campplus_zhen.onnx"; extract = $false },
    @{ url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2"
       check = "$models\sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12\model.onnx"; extract = $true }
)
foreach ($d in $downloads) {
    if (Test-Path $d.check) { continue }
    $file = Join-Path $root ([System.IO.Path]::GetFileName($d.url))
    Write-Host "Downloading $($d.url)..."
    Invoke-WebRequest $d.url -OutFile $file
    if ($d.extract) {
        tar -xjf $file -C $models       # Windows 10+ ships bsdtar; handles .tar.bz2
        if ($LASTEXITCODE -ne 0) { throw "tar extract failed: $file" }
        Remove-Item $file
    } else {
        Move-Item $file $d.check -Force
    }
}

# The ASR model (SoybeanMilk/faster-whisper-Breeze-ASR-25) downloads on first
# pipeline run into the DEFAULT HuggingFace cache - do not redirect it to a deep
# custom path (Windows 260-char path limit).

& $py -c "import ctranslate2; n = ctranslate2.get_cuda_device_count(); print(f'CUDA devices: {n} (0 = will fall back to CPU int8)')"
Write-Host "SETUP OK"
Write-Host "python: $py"
