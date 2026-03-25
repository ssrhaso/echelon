# Copyright 2025, Maxime Burchi.
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

# PyTorch
import torch
    
def return_padding_mask(seq_len, x_len, hidden_len=0, dtype=torch.float32, device="cpu"):

    # Init Mask (B, Th + T)
    mask = torch.zeros(x_len.shape[0], hidden_len + seq_len, dtype=dtype, device=device)

    # Batch Loop
    for b in range(x_len.size(0)):
        assert x_len[b] <= seq_len
        mask[b, :hidden_len + x_len[b]] = x_len.new_ones(hidden_len + x_len[b])

    # Padding Mask (B, 1, Th + T)
    return mask[:, None, :]

def return_padding_mask_hidden(seq_len, x_h_len, hidden_len=0, dtype=torch.float32, device="cpu"):

    # Init Mask (B, Th + T)
    mask = torch.zeros(x_h_len.shape[0], hidden_len + seq_len, dtype=dtype, device=device)

    # Batch Loop
    for b in range(x_h_len.size(0)):
        assert x_h_len[b] <= hidden_len
        # print(seq_len, x_h_len[b])
        mask[b, hidden_len - x_h_len[b]:] = x_h_len.new_ones(seq_len + x_h_len[b])

    # Padding Mask (B, 1, Th + T)
    return mask[:, None, :]
    
def return_mask(seq_len, x_len=None, hidden_len=0, x_h_len=None, left_context=None, right_context=None, mask_start=0, unsqueeze_head=True, dtype=torch.float32, device="cpu"):

    """ return attention Binary Mask (0 = masked, 1 = unmasked) of shape (1, T, Th + T) / (B, T, Th + T) or (1, 1, T, Th + T) / (B, 1, T, Th + T) if unsqueeze_head

    seq_len: length of element sequence T
    x_len: length of indivisual batch samples x_len <= seq_len, required to mask padding elements from right
    hidden_len: length of hidden state Th
    x_h_len: length of indivisual batch samples x_h_len <= hidden_len, required to mask padding elements from left
    left_context: attention left/past context, set to None for inf left context
    right_context: attention right/future context, set to None for inf right context
    mask_start: elements [:mask_start] will not be masked
    unsqueeze_head: unsqueeze mask for MHSA
    device: mask device
    dtype: mask dtype
    
    """

    # Right Context Mask (1, T, Th + T)
    right_context_mask = torch.ones(1, seq_len, hidden_len + seq_len, dtype=dtype, device=device)
    if right_context != None:

        assert isinstance(right_context, int) or isinstance(right_context, list)

        # same right_context for all batch elements (1, T, Th + T)
        if isinstance(right_context, int):
            right_context_mask = right_context_mask.tril(diagonal=hidden_len + right_context)

        # one context for each batch element (B, T, Th + T)
        else:

            # Repeat Right Context Mask
            right_context_mask = right_context_mask.repeat(len(right_context), 1, 1)

            # Tril
            for b, right_c in enumerate(right_context):
                right_context_mask[b] = right_context_mask[b].tril(diagonal=hidden_len + right_c)

    
    # Left Context Mask (1, T, Th + T)
    left_context_mask = torch.ones(1, seq_len, hidden_len + seq_len, dtype=dtype, device=device)
    if left_context != None:

        assert isinstance(left_context, int) or isinstance(left_context, list)

        # same left_context for all batch elements (1, T, Th + T)
        if isinstance(left_context, int):
            left_context_mask = left_context_mask.triu(diagonal=hidden_len-left_context)

        # one context for each batch element (B, T, Th + T)
        else:

            # Repeat Left Context Mask
            left_context_mask = left_context_mask.repeat(len(left_context), 1, 1)

            # Tril
            for b, left_c in enumerate(left_context):
                left_context_mask[b] = left_context_mask[b].tril(diagonal=hidden_len-left_c)

    
    # Full Context Mask (1 or B, T, Th + T)
    context_mask = right_context_mask.minimum(left_context_mask)

    # Mask Start
    context_mask[:, :mask_start, :hidden_len + mask_start] = 1

    # Padding Mask
    if x_len is not None:

        # Padding Mask (B, 1, Th + T)
        padding_mask = return_padding_mask(seq_len, x_len, hidden_len=hidden_len, dtype=dtype, device=device)

        # Context Mask Union Padding Mask (B, T, Th + T)
        context_mask = context_mask.minimum(padding_mask)

    # Hiddden Padding Mask
    if x_h_len is not None:

        # Padding Mask (B, 1, Th + T)
        padding_mask = return_padding_mask_hidden(seq_len, x_h_len, hidden_len=hidden_len, dtype=dtype, device=device)

        # Context Mask Union Padding Mask (B, T, Th + T)
        context_mask = context_mask.minimum(padding_mask)

    # Unsqueeze Head (B or 1, 1, T, Th + T)
    if unsqueeze_head:
        context_mask = context_mask[:, None, :, :]

    return context_mask

def return_is_firsts_mask(is_firsts, is_firsts_hidden=None, unsqueeze_head=True):

    """ return attention Binary Mask (0 = masked, 1 = unmasked) of shape (B, T, T) or (B, 1, T, T) if unsqueeze_head

    The sequence is supposed to be a concatenation of trajectories where the is_firsts tensor represent the positions of each first traj element
    This function returns a binary mask, masking elements of previous trajectories

    is_firsts: is_firsts tensor (B, L) 1.0==is_first 0.0==is_not_first 
    is_firsts_hidden: is_firsts tensor (B, Th) 1.0==is_first 0.0==is_not_first 
    
    """

    # Diagonal is_firsts (B, L, L)
    is_firts_mask = torch.diag_embed(is_firsts)

    # Concat is_firsts_hidden (B, L, L) -> (B, L, Th+L)
    if is_firsts_hidden is not None:
        is_firts_mask = torch.cat([is_firsts_hidden.unsqueeze(dim=1).repeat(1, is_firts_mask.shape[1], 1), is_firts_mask], dim=-1)

    # Roll 
    is_firts_mask = torch.roll(is_firts_mask, shifts=-1, dims=(-1,))
    is_firts_mask[:, :, -1] = 0

    # Inverse values
    is_firts_mask = 1.0 - is_firts_mask

    # Cum prods
    is_firts_mask = torch.flip(is_firts_mask, dims=(-1,))
    is_firts_mask = torch.cumprod(is_firts_mask, dim=-1)
    is_firts_mask = torch.flip(is_firts_mask, dims=(-1,))
    is_firts_mask = torch.cumprod(is_firts_mask, dim=-2)

    # Unsqueeze head (B, 1, L, Th+L)
    if unsqueeze_head:
        is_firts_mask = is_firts_mask.unsqueeze(dim=1)

    return is_firts_mask