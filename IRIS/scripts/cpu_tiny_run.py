"""CPU tiny-run: prove the full IRIS + 3-level HRVQ loop is WIRED before Slurm.

Runs entirely on CPU, with a tiny synthetic batch:
  1. one tokenizer grad step  (3-level residual VQ reconstruction + commitment)
  2. one world-model grad step (consumes 16 level-0 tokens/frame + action)
  3. one actor-critic step     (1 imagination "env step" through WorldModelEnv)

Exits 0 iff every stage builds and back-propagates. This is a WIRING proof, not a
quality signal: no scores, no RL judgement (those need GPU runs).

Run:  iris_hrvq_port> .venv/Scripts/python.exe scripts/cpu_tiny_run.py
"""
import os
import sys
import types

# --- make src importable. Import torch/torchvision FIRST (torch's import walks
# --- sys.modules' __file__, so the gym stub must not exist yet), then install a
# --- gym stub (Atari dep is GPU-only; imagination uses env=None and never calls
# --- a real gym API). Mirrors tests/conftest.py.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch
import torch.nn as nn
import torchvision  # noqa: F401  (force op registration once, before tokenizer import)

if "gym" not in sys.modules:
    _gym = types.ModuleType("gym")

    def _gym_getattr(name):
        if name.startswith("__"):  # never shadow dunders (e.g. __file__, __path__)
            raise AttributeError(name)
        return type(name, (), {"__getattr__": lambda self, _n: (lambda *a, **k: None)})

    _gym.__getattr__ = _gym_getattr
    sys.modules["gym"] = _gym

from models.tokenizer import Encoder, Decoder, EncoderDecoderConfig, Tokenizer
from models.world_model import WorldModel
from models.actor_critic import ActorCritic
from models.transformer import TransformerConfig

ACT_VOCAB = 4
B, T = 2, 2


class _ZeroLPIPS(nn.Module):
    def forward(self, a, b):
        return a.new_zeros(a.shape[0], 1, 1, 1)


def _build_tokenizer():
    cfg = EncoderDecoderConfig(resolution=64, in_channels=3, z_channels=512, ch=64,
                               ch_mult=[1, 1, 1, 1, 1], num_res_blocks=2,
                               attn_resolutions=[8, 16], out_ch=3, dropout=0.0)
    tok = Tokenizer(vocab_size=512, embed_dim=512, encoder=Encoder(cfg), decoder=Decoder(cfg),
                    with_lpips=False)
    tok.lpips = _ZeroLPIPS()
    return tok


def main() -> int:
    torch.manual_seed(0)
    tok = _build_tokenizer()
    world_model = WorldModel(
        obs_vocab_size=tok.vocab_size, act_vocab_size=ACT_VOCAB,
        config=TransformerConfig(tokens_per_block=17, max_blocks=20, attention="causal",
                                 num_layers=2, num_heads=4, embed_dim=256,
                                 embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1),
    )
    actor_critic = ActorCritic(use_original_obs=False, act_vocab_size=ACT_VOCAB)

    batch = {
        "observations": torch.rand(B, T, 3, 64, 64),
        "actions": torch.randint(0, ACT_VOCAB, (B, T)),
        "rewards": torch.randn(B, T),
        "ends": torch.zeros(B, T, dtype=torch.long),
        "mask_padding": torch.ones(B, T, dtype=torch.bool),
    }

    # 1) tokenizer grad step
    tok.zero_grad()
    tok_loss = tok.compute_loss(batch)
    tok_loss.loss_total.backward()
    usage = tok.hrvq.get_codebook_usage(tok.encode(batch["observations"], should_preprocess=True).tokens_levels)
    print(f"[1/3] tokenizer  loss={tok_loss.loss_total.item():.4f}  "
          f"perplexity/level={[round(usage[f'perplexity_{i}'], 1) for i in range(3)]}")

    # 2) world-model grad step
    world_model.zero_grad()
    wm_loss = world_model.compute_loss(batch, tok)
    wm_loss.loss_total.backward()
    print(f"[2/3] world_model loss={wm_loss.loss_total.item():.4f}")

    # 3) actor-critic step (1 imagination env step through the world model)
    actor_critic.zero_grad()
    ac_loss = actor_critic.compute_loss(batch, tok, world_model, imagine_horizon=T,
                                        gamma=0.995, lambda_=0.95, entropy_weight=0.001)
    ac_loss.loss_total.backward()
    print(f"[3/3] actor_critic loss={ac_loss.loss_total.item():.4f}")

    assert tok.encoder.conv_in.weight.grad is not None
    assert world_model.pos_emb.weight.grad is not None
    assert actor_critic.actor_linear.weight.grad is not None
    print("OK: full tokenizer+WM+AC loop wired on CPU (1 grad step each, 1 imagination step).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
