# Copyright 2025, Maxime Burchi.
# Modifications Copyright 2025-2026, Hasaan Ahmad, adapted for ECHELON.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Solve dm_control bug (EGL is Linux-only)
import os
import sys
if sys.platform == "linux":
    os.environ["MUJOCO_GL"] = "egl"

# PyTorch
import torch

# Functions
import functions

# Other
import os
import random
import numpy as np
import argparse
import importlib
import warnings

# Disable Warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    """Seed all RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):

    ###############################################################################
    # Init
    ###############################################################################

    # Seed
    if args.seed is not None:
        seed_everything(args.seed)
        print("Seed: {}".format(args.seed))

    # Print Mode
    print("Mode: {}".format(args.mode))

    # Load Config
    args.config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))

    # Load Model
    model = functions.load_model(args)

    # Codebook Cross-Transfer and Freezing
    if (args.transfer_checkpoint is not None or args.freeze_levels is not None
            or args.freeze_encoder or args.init_encoder):
        from nnet.modules.twister.hrvq.transfer import (
            _load_source_state, load_and_transfer_codebooks, load_and_transfer_encoder,
            load_and_transfer_all, print_transfer_provenance, print_parameter_audit,
        )

        freeze_levels = [int(x) for x in args.freeze_levels.split(",")] if args.freeze_levels else []

        # Load the source checkpoint once and share the state dict
        if args.transfer_checkpoint is not None:
            source_state = _load_source_state(model, args.transfer_checkpoint)

            if args.transfer_all:
                # Whole-model transfer warm-starts every task-agnostic weight
                # and re-initialises only the action-conditioned modules. It
                # carries the encoder, so the codebook-only path is skipped.
                summary = load_and_transfer_all(model, args.transfer_checkpoint, source_state)
                print_transfer_provenance(summary, args.transfer_checkpoint)
            else:
                transfer_levels = freeze_levels if freeze_levels else [0, 1, 2]
                load_and_transfer_codebooks(model, args.transfer_checkpoint, transfer_levels, source_state)

                # Source-initialise the encoder CNN alongside the codebooks.
                # --init_encoder leaves it trainable, --freeze_encoder pins it.
                if args.freeze_encoder or args.init_encoder:
                    load_and_transfer_encoder(model, args.transfer_checkpoint, source_state)

            del source_state  # Free memory
        elif args.freeze_encoder:
            print("WARNING: --freeze_encoder without --transfer_checkpoint freezes the current encoder weights as-is")

        # Freeze specified VQ levels
        if freeze_levels:
            model.encoder_network.hrvq.freeze_levels(freeze_levels)
            print(f"Frozen VQ levels: {freeze_levels}")

        # Freeze the encoder CNN. train_step re-enables grads on whole networks
        # every step, so the _frozen flag, not requires_grad_, is what holds.
        if args.freeze_encoder:
            for p in model.encoder_network.cnn.parameters():
                p._frozen = True
                p.requires_grad_(False)
            print("Frozen encoder CNN")

        # Store freeze/transfer metadata for W&B logging
        model._freeze_levels = args.freeze_levels
        model._transfer_source = args.transfer_checkpoint
        model._freeze_encoder = args.freeze_encoder
        model._init_encoder = args.init_encoder
        model._transfer_all = args.transfer_all

        # Parameter audit
        print_parameter_audit(model)

    # Load Dataset
    dataset_train, dataset_eval = functions.load_datasets(args)

    ###############################################################################
    # Modes
    ###############################################################################

    # Training
    if args.mode == "training":

        model.fit(
            dataset_train=dataset_train, 
            epochs=getattr(args.config, "epochs", 1000), 
            dataset_eval=dataset_eval, 
            initial_epoch=int(args.checkpoint.split("_")[2]) if args.checkpoint != None else 0, 
            callback_path=args.config.callback_path,
            precision=getattr(args.config, "precision", torch.float32),
            accumulated_steps=getattr(args.config, "accumulated_steps", 1),
            eval_period_step=getattr(args.config, "eval_period_step", args.eval_period_step),
            eval_period_epoch=getattr(args.config, "eval_period_epoch", args.eval_period_epoch),
            saving_period_epoch=getattr(args.config, "saving_period_epoch", args.saving_period_epoch),
            log_figure_period_step=getattr(args.config, "log_figure_period_step", args.log_figure_period_step),
            log_figure_period_epoch=getattr(args.config, "log_figure_period_epoch", args.log_figure_period_epoch),
            step_log_period=args.step_log_period,
            grad_init_scale=getattr(args.config, "grad_init_scale", 65536.0),
            detect_anomaly=getattr(args.config, "detect_anomaly", args.detect_anomaly),
            recompute_metrics=getattr(args.config, "recompute_metrics", False),
            wandb_logging=args.wandb,
            wandb_name=args.wandb_name or (
                "{}/{}/seed{}".format(
                    os.environ.get("run_name", "run"),
                    os.environ.get("env_name", "env").split("-", 1)[-1],
                    args.seed
                ) if args.seed is not None else None
            ),
            verbose_progress_bar=args.verbose_progress_bar
        )

    # Evaluation
    elif args.mode == "evaluation":

        model._evaluate(
            dataset_eval, 
            writer=None,
            recompute_metrics=getattr(args.config, "recompute_metrics", False),
            verbose_progress_bar=args.verbose_progress_bar,
        )

    # Pass
    elif args.mode == "pass":
        pass

if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",          type=str,   default="configs/echelon.py",                                       help="Python configuration file containing model hyperparameters")
    parser.add_argument("-m", "--mode",                 type=str,   default="training", choices=["training", "evaluation", "pass"],     help="Mode: training, validation-clean, test-clean, eval_time-dev-clean, ...")
    parser.add_argument("-i", "--checkpoint",           type=str,   default=None,                                                       help="Load model from checkpoint name")
    parser.add_argument("--cpu",                        action="store_true",                                                            help="Load model on cpu")
    parser.add_argument("--load_last",                  action="store_true",                                                            help="Load last model checkpoint")
    parser.add_argument("--wandb",                      action="store_true",                                                            help="Initialize wandb logging")
    parser.add_argument("--wandb_name",                 type=str,   default=None,                                                       help="Custom W&B run name")
    parser.add_argument("--verbose_progress_bar",       type=int,   default=1,                                                          help="Verbose level of progress bar display")

    # Training
    parser.add_argument("--saving_period_epoch",        type=int,   default=1,                                                          help="Model saving every 'n' epochs")
    parser.add_argument("--log_figure_period_step",     type=int,   default=None,                                                       help="Log figure every 'n' steps")
    parser.add_argument("--log_figure_period_epoch",    type=int,   default=1,                                                          help="Log figure every 'n' epochs")
    parser.add_argument("--step_log_period",            type=int,   default=100,                                                        help="Training step log period")
    parser.add_argument("--keep_last_k",                type=int,   default=3,                                                          help="Keep last k checkpoints")

    # Eval
    parser.add_argument("--eval_period_epoch",          type=int,   default=5,                                                          help="Model evaluation every 'n' epochs")
    parser.add_argument("--eval_period_step",           type=int,   default=None,                                                       help="Model evaluation every 'n' steps")

    # Info
    parser.add_argument("--show_dict",                  action="store_true",                                                            help="Show model dict summary")
    parser.add_argument("--show_modules",               action="store_true",                                                            help="Show model named modules")
    
    # Reproducibility
    parser.add_argument("--seed",                       type=int,   default=None,                                                       help="Global random seed for reproducibility")

    # Codebook Freezing / Transfer
    parser.add_argument("--freeze_levels",              type=str,   default=None,                                                       help="Comma-separated VQ levels to freeze, e.g. '0,1'")
    parser.add_argument("--freeze_encoder",             action="store_true",                                                            help="Source-initialise the encoder CNN from --transfer_checkpoint and hold it fixed")
    parser.add_argument("--init_encoder",               action="store_true",                                                            help="Source-initialise the encoder CNN from --transfer_checkpoint but leave it trainable")
    parser.add_argument("--transfer_checkpoint",        type=str,   default=None,                                                       help="Path to checkpoint for VQ codebook cross-transfer")
    parser.add_argument("--transfer_all",               action="store_true",                                                            help="Weight-transfer baseline: warm-start every task-agnostic weight (encoder, codebooks, world model, decoder, reward/value/continue heads) from --transfer_checkpoint. Action-conditioned modules are re-initialised. Nothing is frozen unless --freeze_* is also given.")

    # Debug
    parser.add_argument("--detect_anomaly",             action="store_true",                                                            help="Enable or disable the autograd anomaly detection")
    
    # Parse Args
    args = parser.parse_args()

    # Validate the arguments at startup rather than failing part-way into a run
    if args.transfer_checkpoint is not None and not os.path.isfile(args.transfer_checkpoint):
        parser.error(
            "--transfer_checkpoint path does not exist: {}".format(args.transfer_checkpoint)
        )

    if args.freeze_levels is not None:
        try:
            parsed_levels = [int(x) for x in args.freeze_levels.split(",") if x != ""]
        except ValueError:
            parser.error(
                "--freeze_levels must be comma-separated integers, e.g. '0,1,2' "
                "(got {!r})".format(args.freeze_levels)
            )
        if not parsed_levels or any(lvl < 0 or lvl > 2 for lvl in parsed_levels):
            parser.error(
                "--freeze_levels values must be in [0, 2] (3 HRVQ levels); "
                "got {!r}".format(args.freeze_levels)
            )

    if args.freeze_encoder and args.transfer_checkpoint is None:
        parser.error(
            "--freeze_encoder requires --transfer_checkpoint (freezing a randomly "
            "initialised encoder is never intended)"
        )

    if args.init_encoder:
        if args.transfer_checkpoint is None:
            parser.error("--init_encoder requires --transfer_checkpoint (there is nothing to initialise from)")
        if args.freeze_encoder:
            parser.error("--init_encoder is redundant with --freeze_encoder, which already source-initialises the encoder")

    if args.transfer_all and args.transfer_checkpoint is None:
        parser.error("--transfer_all requires --transfer_checkpoint (there is nothing to transfer from)")

    # Run main
    main(args)
