# ECHELON: Spatial Hierarchical Residual VQ for Transformer World Models

Companion code for the ECHELON paper. ECHELON is a transformer world model that
replaces the flat categorical stochastic state with a spatial Hierarchical
Residual VQ (HRVQ) tokenizer, and uses per-level codebook freezing as the unit
of cross-game transfer.

![ECHELON architecture: encoder CNN, 3-level HRVQ tokenizer, TSSM and decoder, with per-level codebook freezing for cross-game transfer](assets/echelon_arch.gif)

This repository holds the model, its configuration and the sweep definitions.
The manuscript, its figures, the derived result tables and the analysis code
that produces them are kept with the paper rather than here.

## Method

ECHELON inherits the transformer world model and action-conditioned Contrastive
Predictive Coding (AC-CPC) objective from TWISTER. It replaces TWISTER's flat
categorical stochastic state with a 3-level spatial HRVQ tokenizer applied at
each position of the encoder CNN feature map. Each level quantizes the residual
left by the level above it, with EMA codebook updates and dead-code revival.

The encoder, transformer state-space model and decoder consume per-position
cascade-summed embeddings from the three HRVQ levels. The transfer utilities in
[nnet/modules/twister/hrvq/transfer.py](nnet/modules/twister/hrvq/transfer.py)
load pre-trained codebooks from a source game into a target run and freeze a
chosen prefix of levels. That is the knob the freeze-depth and codebook-stitching
sweeps vary.

## Repository Layout

| Path | Contents |
| --- | --- |
| [main.py](main.py), [configs/](configs/) | Entry point and model configuration |
| [nnet/](nnet/) | The model: HRVQ tokenizer, transformer world model, actor-critic, training loop |
| [experiments/](experiments/) | Sweep definitions, setup scripts and SLURM array jobs |
| [IRIS/](IRIS/) | The cross-architecture HRVQ port |

Run output is deliberately not versioned. Training writes checkpoints and replay
buffers to `callbacks/` and raw stdout to `logs/`, both ignored. The W&B run
history and the SLURM stdout logs are the source of truth for every result the
paper reports.

## Installation

Requires Python 3.12 or newer.

```
git clone https://github.com/ssrhaso/ECHELON.git && cd ECHELON
pip install -r requirements.txt
```

## Training

The training environment is selected with the `env_name` variable. Logs, replay
buffer and checkpoints are written to `callbacks/run_name/env_name`.

```
env_name=atari100k-alien run_name=atari100k python3 main.py
```

Hyperparameters can be overridden per run:

```
env_name=atari100k-alien run_name=atari100k override_config='{"num_envs": 4, "epochs": 100}' python3 main.py
```

`precision` accepts `float32` (the atari100k default), `float16` or `bfloat16`.
On 8GB GPUs prefer `bfloat16`: it engages Ampere tensor cores for a large
speedup and, unlike `float16`, its fp32-range exponent avoids the HRVQ
logit-cascade overflow that produces a NaN on the first forward pass. Pair it
with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
`--log_figure_period_epoch 9999` to stay inside the VRAM budget;
`experiments\setup_transfer.ps1 -LowVRAM` applies all three. Keep precision
constant within an ablation set so conditions stay comparable.

## Evaluation

`--mode evaluation` evaluates a trained agent. `--load_last` scans the log
directory for the most recent checkpoint, and `--checkpoint` loads a specific
`.ckpt` file.

## Reproducing the Experiments

### Freeze-depth transfer sweep

[experiments/setup_transfer.ps1](experiments/setup_transfer.ps1) installs
dependencies, fetches the Pong source checkpoint and emits the per-condition
launch commands for the sweep defined in
[experiments/transfer_freezing.yaml](experiments/transfer_freezing.yaml).

```
powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1            # setup + print commands
powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1 -Run       # setup + run the sweep
powershell -ExecutionPolicy Bypass -File experiments\setup_transfer.ps1 -LowVRAM   # 8GB GPUs: bf16 + memory tweaks
```

On SLURM, [experiments/isca_setup.sh](experiments/isca_setup.sh) builds the
environment and the `isca_*.slurm` array jobs run the sweeps. Each array task
maps to one (game, condition, seed) cell, in the order the sweep definition
lists them.

### Codebook stitching

[experiments/codebook_stitching.yaml](experiments/codebook_stitching.yaml)
freezes each HRVQ level from a different donor game, isolating which level
carries transferable game specificity. This is orthogonal to freeze depth: codes
are read L0 L1 L2, with D meaning the target's own codebook and P meaning the
foreign Pong codebook, so PPP reduces to the whole-Pong freeze-L012 condition.

## Cross-Architecture Replication (IRIS)

To test whether the freeze-transfer behaviour is specific to the ECHELON world
model or holds more generally, the 3-level HRVQ tokenizer is also ported into
IRIS (Micheli et al., ICLR 2023), an autoregressive-transformer world model with
a different architecture. The port lives under [IRIS/](IRIS/) and keeps the
upstream layout, so `python src/main.py` runs it. In that host the world model
consumes the level-0 token stream only, so the transfer ladder reduces to the
rungs that affect dynamics: scratch, warmstart, freeze-L0, freeze-enc and
freeze-all.

## Script Options

```
# Args
-c / --config_file           type=str   default="configs/echelon.py"    help="Python configuration file containing model hyperparameters"
-m / --mode                  type=str   default="training"              help="Mode: training, evaluation, pass"
-i / --checkpoint            type=str   default=None                    help="Load model from checkpoint name"
--cpu                        action="store_true"                        help="Load model on cpu"
--load_last                  action="store_true"                        help="Load last model checkpoint"
--wandb                      action="store_true"                        help="Initialize wandb logging"
--verbose_progress_bar       type=int   default=1                       help="Verbose level of progress bar display"

# Training
--saving_period_epoch        type=int   default=1                       help="Model saving every n epochs"
--log_figure_period_step     type=int   default=None                    help="Log figure every n steps"
--log_figure_period_epoch    type=int   default=1                       help="Log figure every n epochs"
--step_log_period            type=int   default=100                     help="Training step log period"
--keep_last_k                type=int   default=3                       help="Keep last k checkpoints"

# Eval
--eval_period_epoch          type=int   default=1                       help="Model evaluation every n epochs"
--eval_period_step           type=int   default=None                    help="Model evaluation every n steps"

# Info
--show_dict                  action="store_true"                        help="Show model dict summary"
--show_modules               action="store_true"                        help="Show model named modules"

# Debug
--detect_anomaly             action="store_true"                        help="Enable or disable the autograd anomaly detection"
```

## Citation

This repository is the companion code for the ECHELON paper (under review).
Until the paper is public, cite the repository directly via
[CITATION.cff](CITATION.cff), which GitHub also exposes through the "Cite this
repository" sidebar.

If you build on ECHELON, please also cite TWISTER, the architecture it extends:

```bibtex
@inproceedings{burchi2025twister,
  title     = {Learning Transformer-based World Models with Contrastive Predictive Coding},
  author    = {Burchi, Maxime and Timofte, Radu},
  booktitle = {The Thirteenth International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=YK9G4Htdew}
}
```

## Acknowledgments

Built on TWISTER. World-model design follows DreamerV3, and tokenizer design
draws on IRIS and VQ-VAE.

## License

ECHELON is released under the Apache License, Version 2.0; see
[LICENSE](LICENSE) and [NOTICE](NOTICE) for full terms and attribution.

ECHELON derives from [TWISTER](https://github.com/burchim/TWISTER) (Burchi,
ICLR 2025), also Apache 2.0. Per Apache 2.0 section 4, original copyright
notices are retained on TWISTER-derived files; ECHELON modifications carry an
additional notice.
