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

"""WorldModel.forward body for spatial HRVQ losses.

Replaces the forward method body in twister.py's WorldModel inner class.
Implements cascade reconstruction, per-level CE, VQ commitment, and
contrastive with pre-VQ features.
"""

import torch
import torch.nn.functional as F

from .decoder import spatial_cascade_decode


def compute_world_model_losses(wm, inputs):
    """WorldModel forward pass for spatial HRVQ.

    This function is called from WorldModel.forward(self, inputs) with wm=self.
    It uses wm.encoder_network, wm.rssm, wm.decoder_network, etc.

    Args:
        wm: WorldModel instance (has .encoder_network, .rssm, .decoder_network,
            .reward_network, .continue_network, .contrastive_network, .config, .outer,
            .add_loss, .add_metric, .add_info, .compute_contrastive_loss)
        inputs: tuple of (states, actions, rewards, dones, is_firsts, model_steps)
    Returns:
        outputs: dict (empty — losses registered via add_loss)
    """
    # Unpack Inputs (B, L, ...)
    states, actions, rewards, dones, is_firsts, model_steps = inputs

    # Outputs dict (losses registered via add_loss, not returned)
    outputs = {}

    assert actions.shape[1] == wm.config.L

    # 1. Encoder Forward

    # Encode observations: CNN -> spatial HRVQ -> aggregate -> stoch
    encoder_out = wm.encoder_network(states)

    # Split: TSSM only sees "stoch", hrvq_info and pre_vq_features stay local
    tssm_states = {"stoch": encoder_out["stoch"]}             # (B, L, 32, 32)
    hrvq_info = encoder_out["hrvq_info"]
    pre_vq_features = encoder_out["pre_vq_features"]          # (B, L, 1024)
    # hrvq_info["indices"][level]: (B, L, 16) — codebook indices per position per level
    # hrvq_info["vq_loss"]: scalar — total commitment loss
    # hrvq_info["z_q_spatial"]: (B, L, 16, 256) — quantized spatial tokens
    # hrvq_info["z_q_levels_spatial"]: list of 3, each (B, L, 16, 256)

    num_levels = len(hrvq_info["z_q_levels_spatial"])

    # 2. TSSM Observe

    # Run TSSM: creates prior (from dynamics) and posterior (from encoder)
    posts, priors = wm.rssm.observe(
        states=tssm_states,
        prev_actions=actions,
        is_firsts=is_firsts,
        prev_state=None,
        is_firsts_hidden=None
    )

    # Update Hidden States (for is_firsts_hidden flattening later)
    is_firsts_hidden_concat = is_firsts

    # Inject prior's per-level logits into posts so that detached_posts
    # contains the logits keys that imagine() expects.
    for level in range(num_levels):
        posts[f"logits_l{level}"] = priors[f"logits_l{level}"]

    # 3. Feature Extraction & Predictions

    # Get features: stoch_flat (1024) + deter (512) = 1536
    feats = wm.rssm.get_feat(posts)

    # Predict rewards
    model_rewards = wm.reward_network(feats)

    # Predict discounts
    discount_pred = wm.continue_network(feats)

    # 4. Contrastive Loss (pre-VQ features for embed)

    # Flatten B and L to ensure diff augment for each sample (B*L, 3, H, W)
    states_flatten = states.flatten(0, 1)

    # Augment
    states_aug = torch.stack([wm.config.contrastive_augments(states_flatten[b]) for b in range(states_flatten.shape[0])], dim=0).reshape(states.shape)

    # Forward encoder only on augmented (NO second TSSM observe)
    posts_con = wm.encoder_network(states_aug)

    # Use pre_vq_features for contrastive embed (continuous, not post-VQ)
    con_embed = posts_con["pre_vq_features"]  # (B, L, 1024)

    # Contrastive steps loop
    for t in range(wm.config.contrastive_steps):

        # Action condition (B, L-t, A*t)
        if t > 0:
            actions_cond = torch.cat([actions[:, 1+t_:min(actions.shape[1], actions.shape[1]+1+t_-t)] for t_ in range(t)], dim=-1)

        # Contrastive features (B, L-t, D)
        features_feats, features_embed = wm.contrastive_network[t](
            feats=wm.rssm.get_feat(priors) if t == 0 else torch.cat([wm.rssm.get_feat(priors)[:, :-t], actions_cond], dim=-1),
            embed=con_embed if t == 0 else con_embed[:, t:]
        )

        # Compute contrastive loss
        if features_feats.dtype != torch.float32:
            with torch.cuda.amp.autocast(enabled=False):
                info_nce_loss, acc_con = wm.compute_contrastive_loss(features_feats.type(torch.float32), features_embed.type(torch.float32))
                info_nce_loss = info_nce_loss.type(features_feats.dtype)
        else:
            info_nce_loss, acc_con = wm.compute_contrastive_loss(features_feats, features_embed)

        # Add Loss (exponential decay weighting, normalized)
        wm.add_loss(
            name="model_contrastive_{}".format(t),
            loss=-info_nce_loss.mean(),
            weight=wm.config.loss_contrastive_scale * (wm.config.contrastive_exp_lambda ** t) * (1.0 / sum([wm.config.contrastive_exp_lambda ** t_ for t_ in range(wm.config.contrastive_steps)]))
        )

        # Add Accuracy
        wm.add_metric("acc_con" if t == 0 else "acc_con_{}".format(t), acc_con)

    # 5. Cascade Reconstruction Loss (3 levels)

    level_weights = wm.config.echelon_level_loss_weights  # [1.0, 0.5, 0.1]
    for level in range(num_levels):
        recon_dist = spatial_cascade_decode(
            wm.decoder_network,
            hrvq_info["z_q_levels_spatial"],
            up_to_level=level,
            dim_cnn=wm.config.dim_cnn
        )
        recon_loss = -recon_dist.log_prob(states.detach()).mean()
        wm.add_loss(
            f"model_image_l{level}",
            recon_loss,
            weight=level_weights[level] * wm.config.loss_decoder_scale
        )

    # 6. Per-Level Cross-Entropy Prior Loss

    for level in range(num_levels):
        prior_logits = priors[f"logits_l{level}"]      # (B, L, 16, num_codes)
        target_indices = hrvq_info["indices"][level]    # (B, L, 16)

        ce_loss = F.cross_entropy(
            prior_logits.reshape(-1, prior_logits.shape[-1]),
            target_indices.reshape(-1)
        )
        wm.add_loss(
            f"ce_prior_l{level}",
            ce_loss,
            weight=level_weights[level] * wm.config.loss_kl_prior_scale
        )

    # 7. VQ Commitment Loss

    vq_loss = hrvq_info["vq_loss"]
    wm.add_loss("vq_commitment", vq_loss, weight=1.0)

    # 8. Reward Loss

    wm.add_loss("model_reward", -model_rewards.log_prob(rewards.unsqueeze(dim=-1).detach()).mean(), weight=wm.config.loss_reward_scale)

    # 9. Discount Loss

    wm.add_loss("model_discount", -discount_pred.log_prob((1.0 - dones).unsqueeze(dim=-1).detach()).mean(), wm.config.loss_discount_scale)

    # 10. Flatten and Detach Posts

    # K, V: (B, C+L, D) -> (B*L, C, D)
    hidden_flatten = [
        (
            # Key (B*L, C, D)
            torch.stack([
                torch.cat([
                    hidden_blk[0].new_zeros(hidden_blk[0].shape[0], max(0, wm.config.L + wm.config.att_context_left - 1 - t - hidden_blk[0].shape[1]), hidden_blk[0].shape[2]),
                    hidden_blk[0][:, max(0, hidden_blk[0].shape[1] - wm.config.L + t + 1 - wm.config.att_context_left):hidden_blk[0].shape[1] - wm.config.L + t + 1]
                ], dim=1)
            for t in range(0, wm.config.L)], dim=1).flatten(start_dim=0, end_dim=1).detach(),

            # Value (B*L, C, D)
            torch.stack([
                torch.cat([
                    hidden_blk[1].new_zeros(hidden_blk[1].shape[0], max(0, wm.config.L + wm.config.att_context_left - 1 - t - hidden_blk[1].shape[1]), hidden_blk[1].shape[2]),
                    hidden_blk[1][:, max(0, hidden_blk[1].shape[1] - wm.config.L + t + 1 - wm.config.att_context_left):hidden_blk[1].shape[1] - wm.config.L + t + 1]
                ], dim=1)
            for t in range(0, wm.config.L)], dim=1).flatten(start_dim=0, end_dim=1).detach(),
        )
    for hidden_blk in posts["hidden"]]

    # is_firsts flatten (B, L) -> (B*L, 1)
    wm.outer.detached_is_firsts = is_firsts.flatten(start_dim=0, end_dim=1).unsqueeze(dim=1).detach()

    # is_firsts hidden flatten (B, C+L) -> (B*L, C)
    wm.outer.detached_is_firsts_hidden = torch.stack([
        torch.cat([
            is_firsts_hidden_concat.new_zeros(is_firsts_hidden_concat.shape[0], max(0, wm.config.L + wm.config.att_context_left - 1 - t - is_firsts_hidden_concat.shape[1])),
            is_firsts_hidden_concat.new_ones(is_firsts_hidden_concat.shape[0], 1),
            is_firsts_hidden_concat[:, max(0, is_firsts_hidden_concat.shape[1] - wm.config.L + t + 1 - wm.config.att_context_left):is_firsts_hidden_concat.shape[1] - wm.config.L + t]
        ], dim=1)
    for t in range(0, wm.config.L)], dim=1).flatten(start_dim=0, end_dim=1).detach()

    # Flatten and detach post (B, L, D) -> (B*L, 1, D)
    wm.outer.detached_posts = {k: hidden_flatten if k == "hidden" else v.flatten(start_dim=0, end_dim=1).unsqueeze(dim=1).detach() for k, v in posts.items()}

    # 11. Logging metrics

    wm.add_info("vq_loss", vq_loss.item())
    for level in range(num_levels):
        wm.add_info(f"vq_perplexity_l{level}", hrvq_info["perplexities"][level].item())

    return outputs
