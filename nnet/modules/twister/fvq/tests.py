# Copyright 2026, Hasaan Ahmad.
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

""" Smoke tests for SINGLE LEVEL FLAT VQ - nnet/modules/twister/fvq/ """

import sys
import os
import types
import torch

# Path + stubs (must come before any nnet import)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))


def _stub(*names):
    for name in names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


_stub("ale_py", "gymnasium", "gymnasium.wrappers")
_gym = sys.modules["gymnasium"]
_gym.wrappers = sys.modules["gymnasium.wrappers"]
_gym.register_envs = lambda *a: None

_stub("dm_control", "dm_control.suite", "dm_control.suite.wrappers")
sys.modules["dm_control"].suite = sys.modules["dm_control.suite"]
_dmc_w = sys.modules["dm_control.suite.wrappers"]
sys.modules["dm_control.suite"].wrappers = _dmc_w
_dmc_w.pixels = types.ModuleType("pixels")

# Stub tssm so encoder can be imported without a real TSSM
_tssm_stub = types.ModuleType("nnet.modules.twister.fvq.tssm")
_tssm_stub.TSSM = type("TSSM", (), {})
sys.modules["nnet.modules.twister.fvq.tssm"] = _tssm_stub

from nnet.modules.twister.fvq.vq import HRVQ, VectorQuantizerEMA
from nnet.modules.twister.fvq.encoder import EncoderNetwork


# VQ tests

def test_vq_train_forward():
    vq = HRVQ(embed_dim=1024, num_codes=[512], commitment_costs=[0.25])
    z_e = torch.randn(4, 8, 1024)
    vq.train()
    out = vq(z_e)
    assert out["z_q"].shape        == (4, 8, 1024), f"z_q shape wrong: {out['z_q'].shape}"
    assert out["indices"][0].shape == (4, 8),        f"indices shape wrong: {out['indices'][0].shape}"
    assert out["vq_loss"].shape    == (),             f"vq_loss should be scalar"
    print(f"VQ train forward             OK  z_q={out['z_q'].shape}  vq_loss={out['vq_loss']:.4f}  perp={out['perplexities'][0]:.1f}")


def test_vq_roundtrip():
    vq = HRVQ(embed_dim=1024, num_codes=[512], commitment_costs=[0.25])
    z_e = torch.randn(4, 8, 1024)
    vq.eval()
    out = vq(z_e)
    z_q_recon = vq.decode_from_indices(out["indices"])
    assert torch.allclose(out["z_q"].detach(), z_q_recon, atol=1e-6)
    print("VQ roundtrip (encode/decode) OK")


# Encoder tests

def test_encoder_batch_sequence():
    enc = EncoderNetwork()
    enc.train()
    out = enc(torch.rand(4, 8, 3, 64, 64))
    stoch   = out["stoch"]
    info    = out["hrvq_info"]
    indices = info["indices"][0]
    assert stoch.shape           == (4, 8, 32, 32)
    assert info["z_q"].shape     == (4, 8, 1024)
    assert indices.shape         == (4, 8)
    assert info["vq_loss"].shape == ()
    assert torch.allclose(stoch.flatten(-2, -1), info["z_q"])
    print(f"Encoder (B,L,C,H,W)          OK  stoch={stoch.shape}")


def test_encoder_single_frame():
    enc = EncoderNetwork()
    enc.eval()
    out = enc(torch.rand(4, 3, 64, 64))
    assert out["stoch"].shape == (4, 32, 32)
    print(f"Encoder (B,C,H,W)            OK  stoch={out['stoch'].shape}")


def test_encoder_gradient():
    enc = EncoderNetwork()
    enc.train()
    loss = enc(torch.rand(2, 4, 3, 64, 64))["stoch"].sum()
    loss.backward()
    assert enc.pre_vq_proj.weight.grad is not None
    assert not torch.isnan(enc.pre_vq_proj.weight.grad).any()
    print("Encoder gradient             OK")


def test_encoder_index_range():
    enc = EncoderNetwork()
    enc.eval()
    out     = enc(torch.rand(4, 8, 3, 64, 64))
    indices = out["hrvq_info"]["indices"][0]
    assert 0 <= indices.min() and indices.max() < 512
    print(f"Index range                  OK  [{indices.min().item()}, {indices.max().item()}] in [0, 511]")


if __name__ == "__main__":
    test_vq_train_forward()
    test_vq_roundtrip()
    test_encoder_batch_sequence()
    test_encoder_single_frame()
    test_encoder_gradient()
    test_encoder_index_range()
    print("\nAll smoke tests passed.")
