# ECHELON: Spatial Hierarchical Residual VQ for Transformer World Models

Research code for ECHELON, a fork of TWISTER that replaces the flat stochastic-state tokens with a spatial Hierarchical Residual VQ (HRVQ) tokenizer.

## Method

ECHELON inherits the transformer-based world model and action-conditioned Contrastive Predictive Coding (AC-CPC) objective from TWISTER. It replaces TWISTER's flat categorical stochastic state with a 3-level spatial Hierarchical Residual VQ tokenizer applied at each spatial position from the encoder CNN. Each VQ level quantizes the residual from the previous level, with EMA codebook updates and dead-code revival.

The encoder, transformer state-space model, and decoder are adapted to consume per-position cascade-summed embeddings from the HRVQ levels. Codebook transfer utilities under nnet/modules/twister/hrvq/transfer.py allow pre-trained codebooks from one game to be loaded and frozen in another for cross-game transfer ablations.
## Installation

Clone GitHub repository and set up environment:

```
git clone https://github.com/ssrhaso/echelon.git && cd echelon
./install.sh
```

### Atari100k Benchmark

The agent can be trained on specific tasks using the 'env_name' variable, which defines the training environment. Training logs, replay buffer and checkpoints will be saved to callbacks/run_name/env_name.

```
env_name=atari100k-alien run_name=atari100k python3 main.py
```

### Override hyperparameters

Overriding model config hyperparameters:

```
env_name=atari100k-alien run_name=atari100k override_config='{"num_envs": 4, "epochs": 100, "eval_episode_saving_path": "./videos"}' python3 main.py
```

## Evaluation

'--mode evaluation' can be used to evaluate agents. The '--load_last' flag will scan the log directory to load the last checkpoint. '--checkpoint' can also be used to load a specific '.ckpt' checkpoint file.

## Script options

```
# Args
-c / --config_file           type=str   default="configs/twister.py"    help="Python configuration file containing model hyperparameters"
-m / --mode                  type=str   default="training"              help="Mode: training, evaluation, pass"
-i / --checkpoint            type=str   default=None                    help="Load model from checkpoint name"
--cpu                        action="store_true"                        help="Load model on cpu"
--load_last                  action="store_true"                        help="Load last model checkpoint"
--wandb                      action="store_true",                       help="Initialize wandb logging"
--verbose_progress_bar       type=int,  default=1,                      help="Verbose level of progress bar display"

# Training
--saving_period_epoch        type=int   default=1                       help="Model saving every 'n' epochs"
--log_figure_period_step     type=int   default=None                    help="Log figure every 'n' steps"
--log_figure_period_epoch    type=int   default=1                       help="Log figure every 'n' epochs"
--step_log_period            type=int   default=100                     help="Training step log period"
--keep_last_k                type=int,  default=3,                      help="Keep last k checkpoints"

# Eval
--eval_period_epoch          type=int   default=1                       help="Model evaluation every 'n' epochs"
--eval_period_step           type=int   default=None                    help="Model evaluation every 'n' steps"

# Info
--show_dict                  action="store_true"                        help="Show model dict summary"
--show_modules               action="store_true"                        help="Show model named modules"

# Debug
--detect_anomaly             action="store_true"                        help="Enable or disable the autograd anomaly detection"
```

## Citation

If this code or paper is helpful in your research, please use the following citation:

Work in Progress!

## Acknowledgments

Dreamer V3

IRIS

TWISTER
