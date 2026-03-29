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

"""Spatial cascade decode: reconstruct images from partial HRVQ level sums.

Bypasses the decoder's self.proj Linear — spatial tokens at (16, 256) can
reshape directly to (256, 4, 4) which is exactly self.dim_proj_out.
"""

from nnet import distributions


def spatial_cascade_decode(decoder_network, z_q_levels_spatial, up_to_level, dim_cnn=32):
    """Decode spatial HRVQ tokens via the existing decoder's transposed CNN.

    Args:
        decoder_network: the existing DecoderNetwork instance
        z_q_levels_spatial: list of tensors, each (B, L, 16, 256) or (B, 16, 256)
        up_to_level: int 0..num_levels-1, decode using levels 0..up_to_level
        dim_cnn: CNN channel multiplier (32 for TWISTER default)
    Returns:
        obs_dist: MSEDist over (*, 3, 64, 64)
    """
    # 1. Sum levels 0..up_to_level: (*, 16, 256)
    z_q_partial = z_q_levels_spatial[0]
    for level in range(1, up_to_level + 1):
        z_q_partial = z_q_partial + z_q_levels_spatial[level]

    # 2. Reshape (*, 16, 256) -> (N, 16, 256) -> (N, 4, 4, 256) -> (N, 256, 4, 4)
    shape = z_q_partial.shape  # (*, 16, 256)
    batch_shape = shape[:-2]
    x = z_q_partial.reshape(-1, 16, 8 * dim_cnn)     # (N, 16, 256)
    x = x.reshape(-1, 4, 4, 8 * dim_cnn)              # (N, 4, 4, 256)
    x = x.permute(0, 3, 1, 2)                          # (N, 256, 4, 4)

    # 3. Pass through decoder's transposed CNN (skip self.proj)
    x = decoder_network.cnn(x)                          # (N, 3, 64, 64)

    # 4. Reshape back to batch dims
    x = x.reshape(batch_shape + x.shape[1:])            # (*, 3, 64, 64)

    # 5. Return MSEDist
    return distributions.MSEDist(x, reinterpreted_batch_ndims=3)
