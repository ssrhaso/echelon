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

"""Original flat-VQ tokenizer kept as the ablation baseline for ECHELON.

The active ECHELON training path uses nnet/modules/twister/hrvq/ instead.
This package is retained so flat-VQ comparisons can be re-run from source.
"""

from .vq import VectorQuantizerEMA
from .vq import HRVQ
from .encoder import EncoderNetwork
from .tssm import TSSM
