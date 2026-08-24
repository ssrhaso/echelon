# ECHELON codebook transfer sweep, run locally on one GPU.
#
# Usage (run from anywhere; the script chdirs to the repo root):
#   powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1                    # setup + print commands
#   powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1 -Run               # setup + run every condition
#   powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1 -Only freeze-CB0   # setup + run one condition
#   powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1 -SkipSetup -Run    # skip env setup, just run
#   powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1 -Run -LowVRAM      # 8GB GPUs: bf16 + memory tweaks
#
# Conditions and target game are defined in experiments\transfer_freezing.yaml.

param(
    [switch]$Run,
    [string]$Only = "",
    [switch]$SkipSetup,
    [switch]$LowVRAM   # 8GB GPUs: bf16 + memory tweaks. Opt-in; off by default.
)

$ErrorActionPreference = "Stop"

Write-Host "ECHELON codebook transfer setup" -ForegroundColor Cyan

# Low-VRAM profile for 8GB cards, opt-in so default launches are unchanged.
# bf16 keeps the HRVQ logit cascade off the fp16 overflow path while engaging
# Ampere tensor cores, expandable_segments lets the allocator reclaim memory
# instead of fragmenting, and figure logging is disabled per run below.
if ($LowVRAM) {
    $env:override_config = '{"precision":"bfloat16"}'
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    Write-Host "LowVRAM profile ON: precision=bfloat16, expandable_segments=True, figure logging disabled" -ForegroundColor Magenta
}

# Resolve the repo root and chdir there so relative paths (nnet/, main.py,
# transfer_ckpt/) resolve wherever the script was invoked from.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot" -ForegroundColor DarkGray

$ConfigFile = Join-Path $PSScriptRoot "transfer_freezing.yaml"
$ConfigFileFwd = $ConfigFile -replace '\\', '/'

# Source checkpoint and its W&B artifact, both named by the sweep definition.
$srcSpec = python -c @"
import yaml
cfg = yaml.safe_load(open('$ConfigFileFwd'))
print(cfg['transfer_checkpoint']); print(cfg['transfer_checkpoint_artifact'])
"@
$PongCkpt = $srcSpec[0]
$PongArtifact = $srcSpec[1]

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

    Write-Host "[8/8] Downloading the source checkpoint from W&B ($PongArtifact)..."
    python -c @"
import glob, os, shutil, wandb
d = wandb.Api().artifact('$PongArtifact').download(root='transfer_ckpt')
ckpts = glob.glob(os.path.join(d, '*.ckpt'))
if not ckpts:
    raise FileNotFoundError(f'No .ckpt in {d}')
src = max(ckpts, key=os.path.getsize)
dst = '$PongCkpt'
if os.path.abspath(src) != os.path.abspath(dst):
    shutil.copyfile(src, dst)
print('DOWNLOADED:', dst)
"@
}

$env:TRANSFER_CKPT = (Resolve-Path $PongCkpt).Path
Write-Host "  TRANSFER_CKPT = $env:TRANSFER_CKPT" -ForegroundColor Green

# ---- Parse the sweep definition into a flat job list ----
# Each line: name|seed|freeze_levels|freeze_encoder|init_encoder|transfer|transfer_all|env_name|run_name|eval_period|keep_last_k|description
$parser = @"
import yaml
cfg = yaml.safe_load(open('$ConfigFileFwd'))
env_name = cfg['env_name']
run_name = cfg['run_name']
base = cfg.get('base_args', {})
eval_p = base.get('eval_period_epoch', 5)
keep_k = base.get('keep_last_k', 3)
for exp in cfg['experiments']:
    fields = [
        exp['name'], None,
        exp.get('freeze_levels') or '',
        int(bool(exp.get('freeze_encoder'))),
        int(bool(exp.get('init_encoder'))),
        int(exp.get('transfer', True)),
        int(bool(exp.get('transfer_all'))),
        env_name, run_name, eval_p, keep_k,
        exp.get('description', ''),
    ]
    for seed in exp['seeds']:
        fields[1] = seed
        print('|'.join(str(f) for f in fields))
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
        InitEncoder    = ($p[4] -eq '1')
        Transfer       = ($p[5] -eq '1')
        TransferAll    = ($p[6] -eq '1')
        EnvName        = $p[7]
        RunName        = $p[8]
        EvalPeriod     = [int]$p[9]
        KeepLastK      = [int]$p[10]
        Description    = $p[11]
    }
}

function Get-LaunchArgs($job) {
    $launch = @(
        "main.py", "--wandb",
        "--seed", $job.Seed,
        "--eval_period_epoch", $job.EvalPeriod,
        "--keep_last_k", $job.KeepLastK
    )
    if ($job.Transfer) {
        $launch += @("--transfer_checkpoint", $env:TRANSFER_CKPT)
    }
    if ($job.TransferAll) {
        $launch += "--transfer_all"
    }
    if ($job.FreezeLevels -ne "") {
        $launch += @("--freeze_levels", $job.FreezeLevels)
    }
    if ($job.FreezeEncoder) {
        $launch += "--freeze_encoder"
    }
    if ($job.InitEncoder) {
        $launch += "--init_encoder"
    }
    if ($LowVRAM) {
        # Per-epoch figure logging is the biggest periodic VRAM spike and does
        # not affect the transfer metrics.
        $launch += @("--log_figure_period_epoch", 9999)
    }
    $target = $job.EnvName -replace '^atari100k-', ''
    $launch += @("--wandb_name", "transfer/$target/$($job.Name)/seed$($job.Seed)")
    return ,$launch
}

Write-Host ""
Write-Host "Setup complete" -ForegroundColor Green
Write-Host ""
Write-Host "=== Experiments loaded from $ConfigFile ===" -ForegroundColor Cyan
foreach ($job in $jobs) {
    $tag = "[{0}] seed={1}  freeze_levels='{2}'  freeze_encoder={3}  init_encoder={4}" -f `
        $job.Name, $job.Seed, $job.FreezeLevels, $job.FreezeEncoder, $job.InitEncoder
    Write-Host "  $tag  $($job.Description)"
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
    if ($LowVRAM) {
        Write-Host "  `$env:override_config = '$env:override_config'"
        Write-Host "  `$env:PYTORCH_CUDA_ALLOC_CONF = `"$env:PYTORCH_CUDA_ALLOC_CONF`""
    }
    Write-Host ""
    foreach ($job in $jobs) {
        $launchArgs = Get-LaunchArgs $job
        Write-Host "# [$($job.Name) seed=$($job.Seed)] $($job.Description)"
        Write-Host "python $($launchArgs -join ' ')"
        Write-Host ""
    }
    Write-Host "To edit seeds/configs, modify $ConfigFile and re-run this script." -ForegroundColor Yellow
}
