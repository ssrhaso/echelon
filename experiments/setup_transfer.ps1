# ECHELON Transfer-Freezing Experiment Setup (Pong -> Breakout)
#
# Usage (run from anywhere - the script chdirs to the repo root):
#   powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1              # setup + print commands
#   powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1 -Run         # setup + run ALL experiments
#   powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1 -Only freeze-L0   # setup + run one
#   powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1 -SkipSetup -Run   # skip env setup, just run
#
# Experiment definitions live in experiments\transfer_freezing.yaml.

param(
    [switch]$Run,
    [string]$Only = "",
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"

Write-Host "ECHELON transfer-freezing setup (Pong -> Breakout)" -ForegroundColor Cyan

# Resolve repo root (parent of this script's folder) and chdir there so all
# relative paths (nnet/, main.py, transfer_ckpt/) resolve correctly no matter
# where the script was invoked from.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot" -ForegroundColor DarkGray

# Pong checkpoint to transfer FROM (W&B artifact pinned to the seed5 Pong run)
$PongArtifact = "haso-university-of-the-west-of-england/nnet/best-checkpoint:v50"
$ConfigFile = Join-Path $PSScriptRoot "transfer_freezing.yaml"

if (-not $SkipSetup) {

    Write-Host "[1/8] Installing torch + torchvision (CUDA 12.8)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

    Write-Host "[2/8] Installing Atari + training deps..."
    pip install gymnasium ale-py opencv-python wandb tqdm av tensorboard pyyaml autorom

    Write-Host "[3/8] Adding user Scripts dir to PATH for this session..."
    $userScripts = Join-Path $env:APPDATA "Python\Python313\Scripts"
    if (Test-Path $userScripts) {
        $env:PATH += ";$userScripts"
    } else {
        Write-Host "  (skipped, $userScripts does not exist)" -ForegroundColor Yellow
    }

    Write-Host "[4/8] Downloading Atari ROMs..."
    try {
        AutoROM --accept-license
    } catch {
        Write-Host "  AutoROM failed: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "  Retrying via python -m AutoROM..." -ForegroundColor Yellow
        try {
            python -m AutoROM --accept-license
        } catch {
            Write-Host "  AutoROM still failing - continuing anyway. Install ROMs manually if training errors." -ForegroundColor Red
        }
    }

    Write-Host "[5/8] Logging into Weights & Biases..."
    python -m wandb login

    Write-Host "[6/8] Patching nnet/envs/__init__.py to tolerate missing dm_control..."
    $envInit = "nnet/envs/__init__.py"
    $content = Get-Content $envInit -Raw
    if ($content -notmatch "try:\s*\r?\n\s*from \. import dm_control") {
        $patched = $content -replace "from \. import dm_control", "try:`n    from . import dm_control`nexcept ImportError:`n    pass"
        Set-Content -Path $envInit -Value $patched -NoNewline
        Write-Host "  patched."
    } else {
        Write-Host "  already patched, skipping."
    }

    Write-Host "[7/8] Disabling sleep/hibernate on AC..."
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0

    Write-Host "[8/8] Downloading Pong checkpoint from W&B ($PongArtifact)..."
    python -c @"
import wandb, shutil, os
api = wandb.Api()
art = api.artifact('$PongArtifact')
d = art.download(root='transfer_ckpt')
src = os.path.join(d, 'best.ckpt')
dst = 'transfer_ckpt/pong_seed5_best.ckpt'
if os.path.abspath(src) != os.path.abspath(dst):
    shutil.copyfile(src, dst)
print('DOWNLOADED:', dst)
"@
}

$env:TRANSFER_CKPT = (Resolve-Path "transfer_ckpt/pong_seed5_best.ckpt").Path
Write-Host "  TRANSFER_CKPT = $env:TRANSFER_CKPT" -ForegroundColor Green

# ---- Parse experiments YAML via python and emit a flat job list ----
# Each line: <exp_name>|<seed>|<freeze_levels_or_empty>|<freeze_encoder 0|1>|<env_name>|<run_name>|<eval_period>|<keep_last_k>|<description>
$ConfigFileFwd = $ConfigFile -replace '\\', '/'
$parser = @"
import yaml, sys
cfg = yaml.safe_load(open('$ConfigFileFwd'))
env_name = cfg['env_name']
run_name = cfg['run_name']
base = cfg.get('base_args', {})
eval_p = base.get('eval_period_epoch', 5)
keep_k = base.get('keep_last_k', 3)
for exp in cfg['experiments']:
    fl = exp.get('freeze_levels') or ''
    fe = 1 if exp.get('freeze_encoder') else 0
    desc = exp.get('description', '')
    for seed in exp['seeds']:
        print(f'{exp[\"name\"]}|{seed}|{fl}|{fe}|{env_name}|{run_name}|{eval_p}|{keep_k}|{desc}')
"@
$jobLines = python -c $parser
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to parse $ConfigFile" -ForegroundColor Red
    exit 1
}

$jobs = @()
foreach ($line in $jobLines) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $p = $line.Split('|')
    $jobs += [pscustomobject]@{
        Name           = $p[0]
        Seed           = [int]$p[1]
        FreezeLevels   = $p[2]
        FreezeEncoder  = ($p[3] -eq '1')
        EnvName        = $p[4]
        RunName        = $p[5]
        EvalPeriod     = [int]$p[6]
        KeepLastK      = [int]$p[7]
        Description    = $p[8]
    }
}

function Get-LaunchArgs($job) {
    $launch = @(
        "main.py", "--wandb",
        "--seed", $job.Seed,
        "--eval_period_epoch", $job.EvalPeriod,
        "--keep_last_k", $job.KeepLastK,
        "--transfer_checkpoint", $env:TRANSFER_CKPT
    )
    if ($job.FreezeLevels -ne "") {
        $launch += @("--freeze_levels", $job.FreezeLevels)
    }
    if ($job.FreezeEncoder) {
        $launch += "--freeze_encoder"
    }
    $launch += @("--wandb_name", "transfer/breakout/$($job.Name)/seed$($job.Seed)")
    return ,$launch
}

Write-Host ""
Write-Host "Setup complete" -ForegroundColor Green
Write-Host ""
Write-Host "=== Experiments loaded from $ConfigFile ===" -ForegroundColor Cyan
foreach ($job in $jobs) {
    $tag = "[{0}] seed={1}  freeze_levels='{2}'  freeze_encoder={3}" -f `
        $job.Name, $job.Seed, $job.FreezeLevels, $job.FreezeEncoder
    Write-Host "  $tag  -- $($job.Description)"
}
Write-Host ""

# ---- Filter jobs if -Only was passed ----
if ($Only -ne "") {
    $jobs = $jobs | Where-Object { $_.Name -eq $Only }
    if ($jobs.Count -eq 0) {
        Write-Host "No experiment named '$Only' in $ConfigFile" -ForegroundColor Red
        exit 1
    }
}

# ---- Execute or print ----
$env:env_name = ($jobs | Select-Object -First 1).EnvName
$env:run_name = ($jobs | Select-Object -First 1).RunName

if ($Run -or $Only -ne "") {
    Write-Host "Running $($jobs.Count) experiment(s)..." -ForegroundColor Cyan
    Write-Host "  env_name = $env:env_name" -ForegroundColor Yellow
    Write-Host "  run_name = $env:run_name" -ForegroundColor Yellow
    foreach ($job in $jobs) {
        $launchArgs = Get-LaunchArgs $job
        Write-Host ""
        Write-Host ">>> [$($job.Name) seed=$($job.Seed)] python $($launchArgs -join ' ')" -ForegroundColor Cyan
        & python @launchArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: $($job.Name) seed=$($job.Seed) (exit $LASTEXITCODE)" -ForegroundColor Red
            Write-Host "Continuing to next experiment..." -ForegroundColor Yellow
        }
    }
    Write-Host ""
    Write-Host "All experiments finished." -ForegroundColor Green
} else {
    Write-Host "Launch commands (copy-paste one, or re-run with -Run / -Only <name>):" -ForegroundColor Cyan
    Write-Host "  `$env:env_name = `"$env:env_name`""
    Write-Host "  `$env:run_name = `"$env:run_name`""
    Write-Host ""
    foreach ($job in $jobs) {
        $launchArgs = Get-LaunchArgs $job
        Write-Host "# [$($job.Name) seed=$($job.Seed)] $($job.Description)"
        Write-Host "python $($launchArgs -join ' ')"
        Write-Host ""
    }
    Write-Host "To edit seeds/configs, modify $ConfigFile and re-run this script." -ForegroundColor Yellow
}
