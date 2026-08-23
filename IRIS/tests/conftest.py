"""Shared pytest fixtures and path setup for the IRIS + 3-level HRVQ CPU port.

All tests run on CPU only. No GPU, no RL, no Atari env. The IRIS source modules
live under ``src/`` and use flat imports, so we put ``src/`` on ``sys.path`` here.
"""
import os
import sys

import pytest
import torch
# Force torchvision's lazy C-op meta-registration to run ONCE here, before any
# test module imports it transitively (via models.tokenizer.lpips). Under
# pytest's import machinery a later first-import can re-enter the registration
# and raise "roi_align already registered"; importing it up front avoids that.
import torchvision  # noqa: F401

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# IRIS's envs/world_model_env.py does `import gym` and uses `gym.Env` only as a
# type annotation. gym (0.21 + ale-py) is the fragile Atari/RL dependency that
# belongs to the GPU env, not CPU validation. We stub a minimal `gym` so the
# actor-critic import chain works on CPU; imagination runs with env=None and
# never calls a real gym API. The launch package documents the real gym build.
if "gym" not in sys.modules:
    import types

    _gym_stub = types.ModuleType("gym")

    def _gym_getattr(name):
        # Every gym.<X> resolves to a fresh dummy class. That serves both uses
        # in the import chain: base classes (gym.ObservationWrapper / Wrapper /
        # RewardWrapper / Env -> subclassed in envs/wrappers.py) and callables
        # (gym.make, gym.spaces.Box -> only invoked when a REAL env is built,
        # which never happens on the CPU imagination path with env=None).
        # Never shadow dunders (e.g. __file__) -- torch's import walks them.
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__getattr__": lambda self, _n: (lambda *a, **k: None)})

    _gym_stub.__getattr__ = _gym_getattr
    sys.modules["gym"] = _gym_stub


@pytest.fixture(autouse=True)
def _cpu_determinism():
    """Make every test deterministic. (Deterministic-algorithms toggling is
    intentionally NOT called here: on torch 2.11 it lazily imports the
    inductor/dynamo/fsdp stack, which double-registers a _c10d_functional kernel
    and errors. Plain manual_seed is enough for these CPU tests.)"""
    torch.manual_seed(0)
    yield


def make_encoder_decoder(z_channels: int = 512):
    """Build the stock IRIS Encoder/Decoder at the 64x64 -> 4x4 config.

    Mirrors config/tokenizer/default.yaml. Used by characterisation tests so
    they exercise the real downsampling stack that yields 16 (=4x4) tokens.
    """
    from models.tokenizer import Encoder, Decoder, EncoderDecoderConfig

    config = EncoderDecoderConfig(
        resolution=64,
        in_channels=3,
        z_channels=z_channels,
        ch=64,
        ch_mult=[1, 1, 1, 1, 1],
        num_res_blocks=2,
        attn_resolutions=[8, 16],
        out_ch=3,
        dropout=0.0,
    )
    return Encoder(config), Decoder(config)


def make_stock_tokenizer(vocab_size: int = 512, embed_dim: int = 512):
    """Stock single-level IRIS Tokenizer on CPU, no LPIPS (no network fetch).

    Matches config/tokenizer/default.yaml dims. with_lpips=False so the VGG
    perceptual weights are never downloaded in the test environment.
    """
    from models.tokenizer import Tokenizer

    encoder, decoder = make_encoder_decoder(z_channels=embed_dim)
    tok = Tokenizer(vocab_size=vocab_size, embed_dim=embed_dim,
                    encoder=encoder, decoder=decoder, with_lpips=False)
    return tok.train()
