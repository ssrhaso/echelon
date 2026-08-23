"""Codebook / encoder transfer for the IRIS + HRVQ port.

Mirrors the public surface of ECHELON's nnet/modules/twister/hrvq/transfer.py
(load source state once, transfer codebooks and optionally the encoder, audit
params) so the eventual ECHELON merge can share the freezing-ladder driver.

Mechanism difference: ECHELON's codebooks live in EMA buffers; here they are
gradient nn.Embedding parameters. Transfer is therefore a parameter copy of
``hrvq.quantizers[level].embedding.weight`` (and, for the encoder, the encoder
CNN + pre_quant_conv weights), not a buffer copy. Shapes are identical, so the
driver logic is unchanged.
"""
from typing import Dict, List

import torch


def _load_source_state(model: torch.nn.Module, checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load a source tokenizer state_dict from disk (CPU map)."""
    state = torch.load(checkpoint_path, map_location="cpu")
    # Accept either a bare state_dict or a {"tokenizer": state_dict} wrapper.
    if isinstance(state, dict) and "tokenizer" in state and isinstance(state["tokenizer"], dict):
        state = state["tokenizer"]
    return state


def load_and_transfer_codebooks(
    model: torch.nn.Module,
    checkpoint_path: str,
    levels: List[int],
    source_state: Dict[str, torch.Tensor] = None,
) -> None:
    """Copy the source codebook weights into ``model.hrvq`` for the given levels."""
    if source_state is None:
        source_state = _load_source_state(model, checkpoint_path)
    with torch.no_grad():
        for level in levels:
            key = f"hrvq.quantizers.{level}.embedding.weight"
            assert key in source_state, f"codebook {key} missing in source checkpoint"
            target = model.hrvq.quantizers[level].embedding.weight
            src = source_state[key]
            assert src.shape == target.shape, f"codebook {level} shape mismatch {src.shape} vs {target.shape}"
            target.copy_(src)


def load_and_transfer_encoder(
    model: torch.nn.Module,
    checkpoint_path: str,
    source_state: Dict[str, torch.Tensor] = None,
) -> None:
    """Copy the source encoder CNN + pre_quant_conv weights into ``model``."""
    if source_state is None:
        source_state = _load_source_state(model, checkpoint_path)
    enc_state = {k[len("encoder."):]: v for k, v in source_state.items() if k.startswith("encoder.")}
    pq_state = {k[len("pre_quant_conv."):]: v for k, v in source_state.items() if k.startswith("pre_quant_conv.")}
    assert enc_state, "no encoder.* params in source checkpoint"
    model.encoder.load_state_dict(enc_state, strict=True)
    model.pre_quant_conv.load_state_dict(pq_state, strict=True)


def print_parameter_audit(model: torch.nn.Module) -> Dict[str, int]:
    """Report trainable vs frozen parameter counts (ECHELON-parity logging)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    frozen_levels = model.hrvq.get_frozen_levels() if hasattr(model, "hrvq") else []
    print(f"[transfer audit] trainable={trainable:,} frozen={frozen:,} frozen_levels={frozen_levels}")
    return {"trainable": trainable, "frozen": frozen}
