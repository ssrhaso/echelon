# ECHELON Transfer-Freezing Experiment Setup (Pong -> Breakout)
#
# Usage (from inside the extracted repo):
#   powershell -ExecutionPolicy Bypass -File setup_transfer.ps1
#
# Runs the base environment setup, then stops before launching training.
# Below are the launch commands for each freezing configuration in the
# Pong-checkpoint -> Breakout transfer sweep. Pick ONE and run it.
#
# Set $env:TRANSFER_CKPT to the path of your trained Pong checkpoint before
# launching, e.g.:
#   $env:TRANSFER_CKPT = "callbacks/hrvq/pong/seed0/checkpoints_swa_epoch_100_step_xxxx.ckpt"

$ErrorActionPreference = "Stop"

Write-Host "ECHELON transfer-freezing setup (Pong -> Breakout)" -ForegroundColor Cyan

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
Write-Host "=== Transfer-freezing launch commands (Pong -> Breakout) ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "First set the source checkpoint path:" -ForegroundColor Yellow
Write-Host '  $env:TRANSFER_CKPT = "callbacks/hrvq/pong/seed0/checkpoints_swa_epoch_XXX_step_YYY.ckpt"'
Write-Host ""
Write-Host "Common env vars for every run:" -ForegroundColor Yellow
Write-Host '  $env:env_name = "atari100k-breakout"'
Write-Host '  $env:run_name = "hrvq_transfer"'
Write-Host ""
Write-Host "--- (A) Transfer codebooks only, nothing frozen (baseline warm-start) ---"
Write-Host '  python main.py --wandb --seed 0 --eval_period_epoch 5 --keep_last_k 3 `'
Write-Host '    --transfer_checkpoint $env:TRANSFER_CKPT `'
Write-Host '    --wandb_name "transfer/breakout/warmstart/seed0"'
Write-Host ""
Write-Host "--- (B) Transfer + freeze VQ level 0 (coarsest) ---"
Write-Host '  python main.py --wandb --seed 0 --eval_period_epoch 5 --keep_last_k 3 `'
Write-Host '    --transfer_checkpoint $env:TRANSFER_CKPT --freeze_levels "0" `'
Write-Host '    --wandb_name "transfer/breakout/freeze-L0/seed0"'
Write-Host ""
Write-Host "--- (C) Transfer + freeze VQ levels 0,1 ---"
Write-Host '  python main.py --wandb --seed 0 --eval_period_epoch 5 --keep_last_k 3 `'
Write-Host '    --transfer_checkpoint $env:TRANSFER_CKPT --freeze_levels "0,1" `'
Write-Host '    --wandb_name "transfer/breakout/freeze-L01/seed0"'
Write-Host ""
Write-Host "--- (D) Transfer + freeze all VQ levels 0,1,2 ---"
Write-Host '  python main.py --wandb --seed 0 --eval_period_epoch 5 --keep_last_k 3 `'
Write-Host '    --transfer_checkpoint $env:TRANSFER_CKPT --freeze_levels "0,1,2" `'
Write-Host '    --wandb_name "transfer/breakout/freeze-L012/seed0"'
Write-Host ""
Write-Host "--- (E) Transfer encoder CNN + freeze encoder (VQ trainable) ---"
Write-Host '  python main.py --wandb --seed 0 --eval_period_epoch 5 --keep_last_k 3 `'
Write-Host '    --transfer_checkpoint $env:TRANSFER_CKPT --freeze_encoder `'
Write-Host '    --wandb_name "transfer/breakout/freeze-enc/seed0"'
Write-Host ""
Write-Host "--- (F) Full frozen backbone: encoder + all VQ levels (dynamics-only finetune) ---"
Write-Host '  python main.py --wandb --seed 0 --eval_period_epoch 5 --keep_last_k 3 `'
Write-Host '    --transfer_checkpoint $env:TRANSFER_CKPT --freeze_encoder --freeze_levels "0,1,2" `'
Write-Host '    --wandb_name "transfer/breakout/freeze-all/seed0"'
Write-Host ""
Write-Host "Swap --seed N to run multiple seeds per configuration."
