"""Config + CLI parity, and an end-to-end CPU forward+backward.

Covers:
  * the tokenizer hydra config (now with hrvq_* fields) parses and builds a
    3-level tokenizer on CPU.
  * the ECHELON-parity CLI (--freeze_levels / --freeze_encoder /
    --transfer_checkpoint) parses and validates exactly like main.py.
  * codebook + encoder transfer copies source weights and freezes correctly.
  * tokenizer + world model + actor-critic build and run a forward+backward on a
    2-step tiny batch end to end (imagination included), all on CPU.
"""
import os

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from conftest import make_stock_tokenizer

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
ACT_VOCAB = 4
B, T = 2, 2


class _ZeroLPIPS(nn.Module):
    def forward(self, a, b):
        return a.new_zeros(a.shape[0], 1, 1, 1)


# ----------------------------- config ------------------------------------- #
def test_tokenizer_config_parses_and_builds_three_levels():
    cfg = OmegaConf.load(os.path.join(CONFIG, "tokenizer", "default.yaml"))
    assert list(cfg.hrvq_num_codes) == [512, 512, 512]
    tok = instantiate(cfg, with_lpips=False)  # override avoids VGG download
    assert tok.hrvq.num_levels == 3
    assert tok.vocab_size == 512


# ------------------------------- CLI -------------------------------------- #
def test_cli_parses_and_validates():
    from port_cli import build_port_argparser, validate_port_args

    p = build_port_argparser()
    args = p.parse_args(["--freeze_levels", "0,1"])
    assert validate_port_args(args) == [0, 1]


def test_cli_rejects_out_of_range_levels():
    from port_cli import build_port_argparser, validate_port_args

    args = build_port_argparser().parse_args(["--freeze_levels", "0,3"])
    try:
        validate_port_args(args)
        assert False, "expected ValueError for level 3"
    except ValueError:
        pass


def test_cli_freeze_encoder_requires_transfer():
    from port_cli import build_port_argparser, validate_port_args

    args = build_port_argparser().parse_args(["--freeze_encoder"])
    try:
        validate_port_args(args)
        assert False, "expected ValueError: --freeze_encoder requires --transfer_checkpoint"
    except ValueError:
        pass


# ----------------------------- transfer ----------------------------------- #
def test_transfer_copies_codebooks_and_encoder_and_freezes(tmp_path):
    from port_cli import build_port_argparser, apply_port_overrides

    source = make_stock_tokenizer()
    ckpt = tmp_path / "source_tokenizer.pt"
    torch.save(source.state_dict(), str(ckpt))

    target = make_stock_tokenizer()
    args = build_port_argparser().parse_args(
        ["--transfer_checkpoint", str(ckpt), "--freeze_levels", "0,1", "--freeze_encoder"]
    )
    apply_port_overrides(target, args)

    # Codebooks for the transferred/frozen levels now match the source.
    for level in (0, 1):
        assert torch.equal(
            target.hrvq.quantizers[level].embedding.weight,
            source.hrvq.quantizers[level].embedding.weight,
        ), f"level {level} codebook not transferred"
    # Encoder weights transferred.
    assert torch.equal(target.encoder.conv_in.weight, source.encoder.conv_in.weight)
    # Freeze state correct.
    assert target.hrvq.get_frozen_levels() == [0, 1]
    assert target.encoder.conv_in.weight.requires_grad is False


# ------------------------------- e2e -------------------------------------- #
def _tiny_batch():
    return {
        "observations": torch.rand(B, T, 3, 64, 64),
        "actions": torch.randint(0, ACT_VOCAB, (B, T)),
        "rewards": torch.randn(B, T),
        "ends": torch.zeros(B, T, dtype=torch.long),
        "mask_padding": torch.ones(B, T, dtype=torch.bool),
    }


def _build_models():
    from models.world_model import WorldModel
    from models.actor_critic import ActorCritic

    tok = make_stock_tokenizer()
    tok.lpips = _ZeroLPIPS()
    wm_cfg = OmegaConf.load(os.path.join(CONFIG, "world_model", "default.yaml"))
    world_model = WorldModel(obs_vocab_size=tok.vocab_size, act_vocab_size=ACT_VOCAB,
                             config=instantiate(wm_cfg))
    actor_critic = ActorCritic(use_original_obs=False, act_vocab_size=ACT_VOCAB)
    return tok, world_model, actor_critic


def test_end_to_end_forward_backward_on_cpu():
    torch.manual_seed(0)
    tok, world_model, actor_critic = _build_models()
    batch = _tiny_batch()

    # Tokenizer step.
    tok.zero_grad()
    tok.compute_loss(batch).loss_total.backward()
    assert tok.encoder.conv_in.weight.grad.abs().sum() > 0

    # World model step (consumes 16 level-0 tokens/frame + action => block of 17).
    world_model.zero_grad()
    world_model.compute_loss(batch, tok).loss_total.backward()
    assert world_model.pos_emb.weight.grad is not None

    # Actor-critic step (imagination through the world model env, horizon=2).
    actor_critic.zero_grad()
    ac_loss = actor_critic.compute_loss(
        batch, tok, world_model,
        imagine_horizon=T, gamma=0.99, lambda_=0.95, entropy_weight=0.01,
    )
    ac_loss.loss_total.backward()
    assert actor_critic.actor_linear.weight.grad is not None
    assert actor_critic.actor_linear.weight.grad.abs().sum() > 0
