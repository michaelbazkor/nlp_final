# Official MulTaBench evaluation for BIN_TEXT_SEC_MDA_DISTRESS.
# Five learners x four conditions x folds 0-4. Does not log in to Weights & Biases.
# Joint TAR fine-tunes E5-small-v2 with LoRA at batch size 8.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:USERPROFILE ".venvs\nlp-multabench\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing virtualenv at $Python"
}
$env:PYTHONPATH = Join-Path $Root "MulTaBench"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
& $Python -u (Join-Path $PSScriptRoot "train_curation.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
