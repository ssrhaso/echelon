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

from .multi_head_attention import MultiHeadAttention
from .rel_pos_multi_head_self_attention import RelPosMultiHeadSelfAttention
from .attention_mask import return_mask, return_padding_mask, return_is_firsts_mask

# Attentions Dictionary
att_dict = {
    "MultiHeadAttention": MultiHeadAttention,
    "RelPosMultiHeadSelfAttention": RelPosMultiHeadSelfAttention
}