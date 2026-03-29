# Copyright 2025, Hasaan Ahmad.
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

"""Tests for spatial HRVQ: component tests + full WorldModel.forward integration.

Run with: python nnet/modules/twister/hrvq/tests.py
(from the ECHELON root directory)
"""

import sys
import os
import types
import torch
import torch.nn as nn
from collections import OrderedDict

# Path + stubs (must come before any nnet import)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))

def _stub(*names):
    for name in names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

_stub("gym", "gym.envs", "gym.envs.atari", "gym.envs.registration", "gym.wrappers")
sys.modules["gym"].envs = sys.modules["gym.envs"]
sys.modules["gym.envs"].atari        = sys.modules["gym.envs.atari"]
sys.modules["gym.envs"].registration = sys.modules["gym.envs.registration"]

_stub("dm_control", "dm_control.suite", "dm_control.suite.wrappers")
sys.modules["dm_control"].suite = sys.modules["dm_control.suite"]
_dmc_w = sys.modules["dm_control.suite.wrappers"]
sys.modules["dm_control.suite"].wrappers = _dmc_w
_dmc_w.pixels = types.ModuleType("pixels")

from nnet.modules.twister.hrvq.vq import HRVQ, VectorQuantizerEMA
from nnet.modules.twister.hrvq.encoder import SpatialHRVQEncoder
from nnet.modules.twister.hrvq.tssm import SpatialHRVQTSSM
from nnet.modules.twister.hrvq.decoder import spatial_cascade_decode
from nnet.modules.twister.hrvq.losses import compute_world_model_losses
from nnet.modules.twister.decoder_network import DecoderNetwork
from nnet.modules.twister.reward_network import RewardNetwork
from nnet.modules.twister.continue_network import ContinueNetwork
from nnet.modules.twister.contrastive_network import ContrastiveNetwork
from nnet.structs import AttrDict


def test_1_hrvq_shape():
    """HRVQ shape correctness with 3 levels."""
    print("TEST 1: HRVQ shape correctness")
    hrvq = HRVQ(embed_dim=256, num_codes=[512, 512, 512], commitment_costs=[0.25, 0.5, 1.0])
    z_e = torch.randn(8, 16, 256)
    flat = z_e.reshape(-1, 256)
    hrvq.train()
    out = hrvq(flat)

    assert out["z_q"].shape == (128, 256), f"z_q shape: {out['z_q'].shape}"
    assert len(out["z_q_levels"]) == 3
    assert len(out["indices"]) == 3
    for i in range(3):
        assert out["z_q_levels"][i].shape == (128, 256), f"z_q_levels[{i}] shape: {out['z_q_levels'][i].shape}"
        assert out["indices"][i].shape == (128,), f"indices[{i}] shape: {out['indices'][i].shape}"
    assert out["vq_loss"].shape == ()
    assert len(out["perplexities"]) == 3

    z_q_sum = sum(zq.detach() for zq in out["z_q_levels"])
    assert torch.allclose(out["z_q"].detach(), z_q_sum, atol=1e-5), "Residual sum mismatch"

    idx = out["indices"]
    assert torch.allclose(hrvq.decode_partial(idx, 0), out["z_q_levels"][0].detach(), atol=1e-5)
    partial_01 = hrvq.decode_partial(idx, 1)
    expected_01 = out["z_q_levels"][0].detach() + out["z_q_levels"][1].detach()
    assert torch.allclose(partial_01, expected_01, atol=1e-5)

    print(f"  PASS  z_q={out['z_q'].shape}  vq_loss={out['vq_loss']:.4f}  perps=[{', '.join(f'{p:.1f}' for p in out['perplexities'])}]")


def test_2_hrvq_gradient():
    """HRVQ gradient flow through straight-through."""
    print("TEST 2: HRVQ gradient flow")
    hrvq = HRVQ(embed_dim=256, num_codes=[512, 512, 512], commitment_costs=[0.25, 0.5, 1.0])
    hrvq.train()
    z_e = torch.randn(8, 16, 256, requires_grad=True)
    flat = z_e.reshape(-1, 256)
    out = hrvq(flat)
    out["z_q"].sum().backward()
    assert z_e.grad is not None, "No gradient on z_e"
    assert z_e.grad.abs().sum() > 0, "Zero gradient on z_e"
    print(f"  PASS  grad norm={z_e.grad.norm():.4f}")


def test_3_encoder_shape():
    """Encoder shape correctness with (B, L, C, H, W) input."""
    print("TEST 3: Encoder (B, L, C, H, W) shape")
    encoder = SpatialHRVQEncoder()
    encoder.train()
    images = torch.randn(2, 4, 3, 64, 64)
    out = encoder(images)

    assert out["stoch"].shape == (2, 4, 32, 32), f"stoch: {out['stoch'].shape}"
    assert out["hrvq_info"]["z_q_spatial"].shape == (2, 4, 16, 256), f"z_q_spatial: {out['hrvq_info']['z_q_spatial'].shape}"
    assert len(out["hrvq_info"]["z_q_levels_spatial"]) == 3
    for i in range(3):
        assert out["hrvq_info"]["z_q_levels_spatial"][i].shape == (2, 4, 16, 256), f"z_q_levels[{i}]: {out['hrvq_info']['z_q_levels_spatial'][i].shape}"
        assert out["hrvq_info"]["indices"][i].shape == (2, 4, 16), f"indices[{i}]: {out['hrvq_info']['indices'][i].shape}"
    assert out["pre_vq_features"].shape == (2, 4, 1024), f"pre_vq_features: {out['pre_vq_features'].shape}"
    print(f"  PASS  stoch={out['stoch'].shape}  z_q_spatial={out['hrvq_info']['z_q_spatial'].shape}")


def test_4_encoder_single_image():
    """Encoder single-image mode (for env_step): (B, C, H, W) no L dim."""
    print("TEST 4: Encoder (B, C, H, W) single image")
    encoder = SpatialHRVQEncoder()
    encoder.eval()
    image = torch.randn(1, 3, 64, 64)
    out = encoder(image)
    assert out["stoch"].shape == (1, 32, 32), f"stoch: {out['stoch'].shape}"
    assert out["hrvq_info"]["indices"][0].shape == (1, 16), f"indices: {out['hrvq_info']['indices'][0].shape}"
    print(f"  PASS  stoch={out['stoch'].shape}")


def test_5_tssm_forward_img():
    """TSSM forward_img shape correctness."""
    print("TEST 5: TSSM forward_img shape")
    encoder = SpatialHRVQEncoder()
    tssm = SpatialHRVQTSSM(
        num_actions=18, hidden_size=512, num_blocks=2, ff_ratio=2,
        num_heads=8, drop_rate=0.0, att_context_left=8,
        num_positions=16, position_dim=256, num_codes=[512, 512, 512],
        hrvq=encoder.hrvq, spatial_proj_dim=128,
    )
    tssm.eval()

    B, L = 2, 1
    prev_state = tssm.initial(batch_size=B, seq_length=L)
    prev_actions = torch.randn(B, L, 18)

    from nnet import modules
    mask = modules.return_mask(seq_len=L, hidden_len=0, left_context=8, right_context=0, dtype=torch.float32, device="cpu")

    with torch.no_grad():
        out = tssm.forward_img(prev_state, prev_actions, mask)

    assert out["stoch"].shape == (B, L, 32, 32), f"stoch: {out['stoch'].shape}"
    assert out["deter"].shape == (B, L, 512), f"deter: {out['deter'].shape}"
    assert out["logits_l0"].shape == (B, L, 16, 512), f"logits_l0: {out['logits_l0'].shape}"
    assert out["logits_l1"].shape == (B, L, 16, 512), f"logits_l1: {out['logits_l1'].shape}"
    assert out["logits_l2"].shape == (B, L, 16, 512), f"logits_l2: {out['logits_l2'].shape}"
    print(f"  PASS  stoch={out['stoch'].shape}  deter={out['deter'].shape}  logits_l0={out['logits_l0'].shape}")


def test_6_tssm_get_feat():
    """TSSM get_feat unchanged: stoch(32,32) + deter(512) = 1536."""
    print("TEST 6: TSSM get_feat")
    encoder = SpatialHRVQEncoder()
    tssm = SpatialHRVQTSSM(
        num_actions=18, hidden_size=512, num_blocks=2, ff_ratio=2,
        num_heads=8, drop_rate=0.0, att_context_left=8,
        num_positions=16, position_dim=256, num_codes=[512, 512, 512],
        hrvq=encoder.hrvq, spatial_proj_dim=128,
    )
    state = {"stoch": torch.randn(2, 4, 32, 32), "deter": torch.randn(2, 4, 512)}
    feat = tssm.get_feat(state)
    assert feat.shape == (2, 4, 1536), f"feat: {feat.shape}"
    print(f"  PASS  feat={feat.shape}")


def test_7_cascade_decode():
    """spatial_cascade_decode shape correctness."""
    print("TEST 7: spatial_cascade_decode shape")
    decoder = DecoderNetwork(feat_size=1024, dim_cnn=32)
    z_q_levels = [torch.randn(2, 4, 16, 256) for _ in range(3)]
    dist = spatial_cascade_decode(decoder, z_q_levels, up_to_level=0, dim_cnn=32)
    assert dist.mode().shape == (2, 4, 3, 64, 64), f"L0 decode: {dist.mode().shape}"
    dist2 = spatial_cascade_decode(decoder, z_q_levels, up_to_level=2, dim_cnn=32)
    assert dist2.mode().shape == (2, 4, 3, 64, 64), f"L0+L1+L2 decode: {dist2.mode().shape}"
    print(f"  PASS  L0={dist.mode().shape}  L012={dist2.mode().shape}")


def test_8_tssm_initial_keys():
    """TSSM initial() state dict keys."""
    print("TEST 8: TSSM initial state dict keys")
    encoder = SpatialHRVQEncoder()
    tssm = SpatialHRVQTSSM(
        num_actions=18, hidden_size=512, num_blocks=2, ff_ratio=2,
        num_heads=8, drop_rate=0.0, att_context_left=8,
        num_positions=16, position_dim=256, num_codes=[512, 512, 512],
        hrvq=encoder.hrvq, spatial_proj_dim=128,
    )
    state = tssm.initial(batch_size=2, seq_length=1)
    assert "logits_l0" in state, "Missing logits_l0"
    assert "logits_l1" in state, "Missing logits_l1"
    assert "logits_l2" in state, "Missing logits_l2"
    assert state["logits_l0"].shape == (2, 1, 16, 512), f"logits_l0: {state['logits_l0'].shape}"
    assert state["stoch"].shape == (2, 1, 32, 32), f"stoch: {state['stoch'].shape}"
    assert state["deter"].shape == (2, 1, 512), f"deter: {state['deter'].shape}"
    print(f"  PASS  keys={list(state.keys())}  logits_l0={state['logits_l0'].shape}")


def test_9_tssm_imagine():
    """TSSM imagine() runs without error and produces correct keys."""
    print("TEST 9: TSSM imagine")
    encoder = SpatialHRVQEncoder()
    tssm = SpatialHRVQTSSM(
        num_actions=18, hidden_size=512, num_blocks=2, ff_ratio=2,
        num_heads=8, drop_rate=0.0, att_context_left=8,
        num_positions=16, position_dim=256, num_codes=[512, 512, 512],
        hrvq=encoder.hrvq, spatial_proj_dim=128,
    )
    tssm.eval()

    B = 2
    img_steps = 3
    prev_state = tssm.initial(batch_size=B, seq_length=1)

    class MockDist:
        def __init__(self, logits):
            self._logits = logits
        def rsample(self):
            return torch.softmax(self._logits, dim=-1)

    class MockPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1536, 18)
        def forward(self, x):
            return MockDist(self.linear(x))

    p_net = MockPolicy()

    with torch.no_grad():
        img_states = tssm.imagine(p_net, prev_state, img_steps=img_steps)

    expected_keys = {"stoch", "deter", "action", "logits_l0", "logits_l1", "logits_l2"}
    assert expected_keys.issubset(set(img_states.keys())), f"Missing keys: {expected_keys - set(img_states.keys())}"
    assert img_states["stoch"].shape == (B, 1 + img_steps, 32, 32), f"stoch: {img_states['stoch'].shape}"
    assert img_states["logits_l0"].shape == (B, 1 + img_steps, 16, 512), f"logits_l0: {img_states['logits_l0'].shape}"
    print(f"  PASS  stoch={img_states['stoch'].shape}  logits_l0={img_states['logits_l0'].shape}")


def test_10_ema_update():
    """EMA codebook updates: weights should change after training forward passes."""
    print("TEST 10: EMA codebook updates")
    vq = VectorQuantizerEMA(num_codes=512, embed_dim=256)
    vq.train()
    initial_emb = vq.embedding.clone()
    for _ in range(10):
        z = torch.randn(64, 256)
        vq(z)
    assert not torch.allclose(vq.embedding, initial_emb, atol=1e-6), "Codebook did not update"
    print(f"  PASS  embedding changed after 10 updates")


def test_11_dead_code_revival():
    """Dead code revival: force dead codes and verify they get replaced."""
    print("TEST 11: Dead code revival")
    vq = VectorQuantizerEMA(num_codes=512, embed_dim=256, revival_interval=1, revival_threshold=1.0)
    vq.train()
    vq.ema_cluster_size[:256] = 0.0
    dead_emb_before = vq.embedding[:256].clone()
    z = torch.randn(64, 256)
    vq(z)
    changed = not torch.allclose(vq.embedding[:256], dead_emb_before, atol=1e-6)
    assert changed, "Dead codes were not revived"
    print(f"  PASS  dead codes revived")


def test_12_world_model_forward():
    """Full WorldModel.forward on CPU with synthetic batch (B=2, L=4).

    Builds all real networks (encoder, TSSM, decoder, reward, continue,
    contrastive) and runs compute_world_model_losses end-to-end.
    Verifies: no crashes, correct loss keys, detached_posts shapes.
    """
    print("TEST 12: Full WorldModel.forward integration (B=2, L=4)")

    import torchvision

    B, L = 2, 4
    num_actions = 18
    feat_size = 32 * 32 + 512  # 1536

    encoder = SpatialHRVQEncoder(
        dim_input_cnn=3, dim_cnn=32,
        cnn_norm={"class": "LayerNorm", "params": {"eps": 1e-3, "convert_float32": True}},
        stoch_size=32, discrete=32,
        num_positions=16, position_dim=256,
        hrvq_num_codes=[512, 512, 512],
        hrvq_commitment_costs=[0.25, 0.5, 1.0],
        hrvq_ema_decay=0.99,
    )

    decoder = DecoderNetwork(
        dim_output_cnn=3,
        feat_size=32 * 32,  # 1024
        dim_cnn=32,
        cnn_norm={"class": "LayerNorm", "params": {"eps": 1e-3, "convert_float32": True}},
    )

    rssm = SpatialHRVQTSSM(
        num_actions=num_actions,
        stoch_size=32, discrete=32,
        learn_initial=True,
        norm={"class": "LayerNorm", "params": {"eps": 1e-3, "convert_float32": True}},
        hidden_size=512,
        num_blocks=2,  # smaller for CPU test
        ff_ratio=2,
        num_heads=8,
        drop_rate=0.0,
        att_context_left=4,  # must be <= L
        num_positions=16, position_dim=256,
        num_codes=[512, 512, 512],
        hrvq=encoder.hrvq,
        spatial_proj_dim=128,
    )

    reward_network = RewardNetwork(
        hidden_size=512, feat_size=feat_size, num_mlp_layers=2,
        norm={"class": "LayerNorm", "params": {"eps": 1e-3, "convert_float32": True}},
    )

    continue_network = ContinueNetwork(
        hidden_size=512, feat_size=feat_size, num_mlp_layers=2,
        norm={"class": "LayerNorm", "params": {"eps": 1e-3, "convert_float32": True}},
    )

    contrastive_steps = 2  # keep small for CPU test
    contrastive_network = nn.ModuleList([
        ContrastiveNetwork(
            feat_size=feat_size + t * num_actions,
            embed_size=32 * 32,  # 1024
            hidden_size=512, out_size=512, num_layers=2,
        )
        for t in range(contrastive_steps)
    ])

    class MockOuter:
        """Stands in for the TWISTER instance that WorldModel.outer references."""
        pass

    outer = MockOuter()
    outer.detached_posts = None
    outer.detached_is_firsts = None
    outer.detached_is_firsts_hidden = None

    class MockWorldModel:
        """Minimal mock of WorldModel with real networks and add_loss/add_info/add_metric."""

        def __init__(self):
            self.encoder_network = encoder
            self.decoder_network = decoder
            self.rssm = rssm
            self.reward_network = reward_network
            self.continue_network = continue_network
            self.contrastive_network = contrastive_network
            self.outer = outer

            # Config — subset needed by compute_world_model_losses
            self.config = AttrDict()
            self.config.L = L
            self.config.dim_cnn = 32
            self.config.att_context_left = 4
            self.config.hrvq_num_codes = [512, 512, 512]
            self.config.echelon_level_loss_weights = [1.0, 0.5, 0.1]
            self.config.loss_decoder_scale = 1.0
            self.config.loss_kl_prior_scale = 0.5
            self.config.loss_reward_scale = 1.0
            self.config.loss_discount_scale = 1.0
            self.config.loss_contrastive_scale = 1.0
            self.config.contrastive_steps = contrastive_steps
            self.config.contrastive_exp_lambda = 0.75
            self.config.contrastive_augments = torchvision.transforms.RandomResizedCrop(
                size=(64, 64), antialias=True, scale=(0.25, 1)
            )

            # Loss/metric/info storage (mimics Module base class)
            self.added_losses = OrderedDict()
            self.added_metrics = OrderedDict()
            self.infos = OrderedDict()

        def add_loss(self, name, loss, weight=1.0):
            self.added_losses[name] = {"loss": loss, "weight": weight}

        def add_metric(self, name, metric):
            self.added_metrics[name] = metric

        def add_info(self, name, info):
            self.infos[name] = info

        def compute_contrastive_loss(self, features_x, features_y):
            features_x = features_x.flatten(start_dim=0, end_dim=1)
            features_y = features_y.flatten(start_dim=0, end_dim=1)
            features = features_x.matmul(features_y.transpose(0, 1))
            features_pos = torch.diag(features)
            features_all = torch.logsumexp(features, dim=-1)
            info_nce_loss = features_pos - features_all
            acc_con = torch.mean(torch.where(
                features.argmax(dim=-1).cpu() == torch.arange(0, features.shape[0]),
                1.0, 0.0
            ))
            return info_nce_loss, acc_con

    wm = MockWorldModel()

    states = torch.randn(B, L, 3, 64, 64)
    actions = torch.randn(B, L, num_actions)
    rewards = torch.randn(B, L)
    dones = torch.zeros(B, L)
    is_firsts = torch.zeros(B, L)
    is_firsts[:, 0] = 1.0
    model_steps = torch.zeros(B, L)

    inputs = (states, actions, rewards, dones, is_firsts, model_steps)

    encoder.train()
    rssm.train()
    decoder.train()
    reward_network.train()
    continue_network.train()
    contrastive_network.train()

    outputs = compute_world_model_losses(wm, inputs)

    expected_losses = set()
    # Cascade reconstruction: model_image_l0, model_image_l1, model_image_l2
    for lvl in range(3):
        expected_losses.add(f"model_image_l{lvl}")
    # Per-level CE: ce_prior_l0, ce_prior_l1, ce_prior_l2
    for lvl in range(3):
        expected_losses.add(f"ce_prior_l{lvl}")
    # VQ commitment
    expected_losses.add("vq_commitment")
    # Reward, discount
    expected_losses.add("model_reward")
    expected_losses.add("model_discount")
    # Contrastive (contrastive_steps=2)
    for t in range(contrastive_steps):
        expected_losses.add(f"model_contrastive_{t}")

    actual_losses = set(wm.added_losses.keys())
    missing = expected_losses - actual_losses
    assert not missing, f"Missing losses: {missing}"
    print(f"  Losses registered: {sorted(actual_losses)}")

    # Verify all losses are finite scalars
    for name, entry in wm.added_losses.items():
        loss_val = entry["loss"]
        assert loss_val.dim() == 0, f"Loss '{name}' not scalar: shape={loss_val.shape}"
        assert torch.isfinite(loss_val), f"Loss '{name}' not finite: {loss_val.item()}"

    dp = outer.detached_posts
    assert dp is not None, "detached_posts not set"
    assert "stoch" in dp, "detached_posts missing 'stoch'"
    assert "deter" in dp, "detached_posts missing 'deter'"
    assert "hidden" in dp, "detached_posts missing 'hidden'"
    for lvl in range(3):
        assert f"logits_l{lvl}" in dp, f"detached_posts missing 'logits_l{lvl}'"

    # Shapes: (B*L, 1, ...)
    BL = B * L
    assert dp["stoch"].shape == (BL, 1, 32, 32), f"detached_posts stoch: {dp['stoch'].shape}"
    assert dp["deter"].shape == (BL, 1, 512), f"detached_posts deter: {dp['deter'].shape}"
    assert dp["logits_l0"].shape == (BL, 1, 16, 512), f"detached_posts logits_l0: {dp['logits_l0'].shape}"
    print(f"  detached_posts: stoch={dp['stoch'].shape}  deter={dp['deter'].shape}  logits_l0={dp['logits_l0'].shape}")

    # Verify detached_is_firsts
    assert outer.detached_is_firsts.shape == (BL, 1), f"detached_is_firsts: {outer.detached_is_firsts.shape}"
    print(f"  detached_is_firsts: {outer.detached_is_firsts.shape}")

    # Verify all detached_posts are actually detached
    for k, v in dp.items():
        if k == "hidden":
            for blk in v:
                assert not blk[0].requires_grad, f"hidden key not detached"
                assert not blk[1].requires_grad, f"hidden value not detached"
        else:
            assert not v.requires_grad, f"detached_posts['{k}'] still requires grad"

    assert "vq_loss" in wm.infos, "Missing info: vq_loss"
    for lvl in range(3):
        assert f"vq_perplexity_l{lvl}" in wm.infos, f"Missing info: vq_perplexity_l{lvl}"
    print(f"  Infos: {list(wm.infos.keys())}")

    assert "acc_con" in wm.added_metrics, "Missing metric: acc_con"
    print(f"  Metrics: {list(wm.added_metrics.keys())}")

    total_loss = sum(entry["loss"] * entry["weight"] for entry in wm.added_losses.values())
    total_loss.backward()

    # Check gradient flows to encoder CNN
    cnn_param = next(encoder.cnn.parameters())
    assert cnn_param.grad is not None, "No gradient on encoder CNN"
    assert torch.isfinite(cnn_param.grad).all(), "NaN/Inf gradient on encoder CNN"

    # Check gradient flows to TSSM spatial_proj
    assert rssm.spatial_proj.weight.grad is not None, "No gradient on TSSM spatial_proj"
    assert torch.isfinite(rssm.spatial_proj.weight.grad).all(), "NaN/Inf gradient on TSSM spatial_proj"

    print(f"  Backward pass OK  total_loss={total_loss.item():.4f}")
    print(f"  PASS")


# Main

if __name__ == "__main__":
    test_1_hrvq_shape()
    test_2_hrvq_gradient()
    test_3_encoder_shape()
    test_4_encoder_single_image()
    test_5_tssm_forward_img()
    test_6_tssm_get_feat()
    test_7_cascade_decode()
    test_8_tssm_initial_keys()
    test_9_tssm_imagine()
    test_10_ema_update()
    test_11_dead_code_revival()
    test_12_world_model_forward()
    print("\nAll 12 tests passed.")
