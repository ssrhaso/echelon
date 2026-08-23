"""Port CLI surface: freezing-ladder + codebook-transfer flags.

Mirrors ECHELON's main.py argparse surface (--freeze_levels, --freeze_encoder,
--transfer_checkpoint) and its validation, so the GPU launch scripts drive the
IRIS port the same way they drive ECHELON. IRIS's own entrypoint is hydra-based;
these flags are parsed here and applied to the built tokenizer via
``apply_port_overrides`` (called from the launch wrapper / trainer hook).

The actual training loop is GPU/RL work and is NOT invoked here -- this module
only parses, validates, and wires freeze/transfer onto a CPU-built tokenizer.
"""
import argparse
import os
from typing import List, Optional

from models.tokenizer.transfer import (
    _load_source_state,
    load_and_transfer_codebooks,
    load_and_transfer_encoder,
    print_parameter_audit,
)

NUM_LEVELS = 3


def build_port_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IRIS+HRVQ port: freeze/transfer flags (ECHELON parity)")
    p.add_argument("--freeze_levels", type=str, default=None,
                   help="Comma-separated HRVQ levels to freeze, e.g. '0,1'")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze encoder CNN + pre_quant_conv (use with --transfer_checkpoint)")
    p.add_argument("--transfer_checkpoint", type=str, default=None,
                   help="Path to a source tokenizer checkpoint for codebook/encoder transfer")
    return p


def validate_port_args(args: argparse.Namespace) -> List[int]:
    """Validate flags exactly as ECHELON's main.py does. Returns parsed levels."""
    if args.transfer_checkpoint is not None and not os.path.isfile(args.transfer_checkpoint):
        raise FileNotFoundError(f"--transfer_checkpoint path does not exist: {args.transfer_checkpoint}")

    freeze_levels: List[int] = []
    if args.freeze_levels is not None:
        try:
            freeze_levels = [int(x) for x in args.freeze_levels.split(",") if x != ""]
        except ValueError:
            raise ValueError(
                "--freeze_levels must be comma-separated integers, e.g. '0,1,2' "
                "(got {!r})".format(args.freeze_levels)
            )
        if any(l < 0 or l >= NUM_LEVELS for l in freeze_levels):
            raise ValueError(
                "--freeze_levels values must be in [0, {}] ({} HRVQ levels); got {!r}".format(
                    NUM_LEVELS - 1, NUM_LEVELS, args.freeze_levels
                )
            )

    if args.freeze_encoder and args.transfer_checkpoint is None:
        raise ValueError(
            "--freeze_encoder requires --transfer_checkpoint (freezing a randomly "
            "initialised encoder is almost never intended)"
        )
    return freeze_levels


def apply_port_overrides(tokenizer, args: argparse.Namespace) -> None:
    """Apply transfer + freeze to a built tokenizer, mirroring main.py:69-107."""
    freeze_levels = validate_port_args(args)

    if args.transfer_checkpoint is not None:
        source_state = _load_source_state(tokenizer, args.transfer_checkpoint)
        transfer_levels = freeze_levels if freeze_levels else list(range(NUM_LEVELS))
        load_and_transfer_codebooks(tokenizer, args.transfer_checkpoint, transfer_levels, source_state)
        if args.freeze_encoder:
            load_and_transfer_encoder(tokenizer, args.transfer_checkpoint, source_state)

    if args.freeze_encoder:
        tokenizer.freeze_encoder()

    if freeze_levels:
        tokenizer.hrvq.freeze_levels(freeze_levels)
        print(f"Frozen HRVQ levels: {freeze_levels}")

    print_parameter_audit(tokenizer)
