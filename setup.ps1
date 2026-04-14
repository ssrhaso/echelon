# ECHELON Windows Setup
#
# Usage (from inside the extracted repo):
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# Installs torch (cu128), Atari deps, ROMs, logs into W&B, disables sleep,
# and runtime-patches nnet/envs/__init__.py so the missing dm_control import
# doesn't crash on Windows. Stops before launching training.

$ErrorActionPreference = "Stop"

Write-Host "ECHELON Windows setup" -ForegroundColor Cyan

Write-Host "[1/7] Installing torch + torchvision (CUDA 12.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "[2/7] Installing Atari + training deps..."
pip install gymnasium ale-py opencv-python wandb tqdm av tensorboard pyyaml autorom

Write-Host "[3/7] Adding user Scripts dir to PATH for this session..."
$userScripts = Join-Path $env:APPDATA "Python\Python313\Scripts"
if (Test-Path $userScripts) {
    $env:PATH += ";$userScripts"
} else {
    Write-Host "  (skipped, $userScripts does not exist)" -ForegroundColor Yellow
}

Write-Host "[4/7] Downloading Atari ROMs..."
AutoROM --accept-license

Write-Host "[5/7] Logging into Weights & Biases..."
python -m wandb login

Write-Host "[6/7] Patching nnet/envs/__init__.py to tolerate missing dm_control..."
$envInit = "nnet/envs/__init__.py"
$content = Get-Content $envInit -Raw
if ($content -notmatch "try:\s*\r?\n\s*from \. import dm_control") {
    $patched = $content -replace "from \. import dm_control", "try:`n    from . import dm_control`nexcept ImportError:`n    pass"
    Set-Content -Path $envInit -Value $patched -NoNewline
    Write-Host "  patched."
} else {
    Write-Host "  already patched, skipping."
}

Write-Host "[7/7] Disabling sleep/hibernate on AC..."
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

Write-Host ""
Write-Host "Setup complete" -ForegroundColor Green
Write-Host ""
Write-Host "Launch training with (pick your seed/game):" -ForegroundColor Cyan
Write-Host '  $env:env_name="atari100k-pong"; $env:run_name="hrvq"; python main.py --wandb --seed 0 --eval_period_epoch 5 --keep_last_k 3'
Write-Host ""
Write-Host "Games: pong, breakout, alien, ... (lowercase). Swap --seed N as needed."
