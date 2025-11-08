# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Biatron model configuration"""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class BiatronConfig(PretrainedConfig):
    
    r"""
    This is the configuration class to store the configuration of a [`Biatron`]. It is used to instantiate an Biatron
    model according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "Biatron"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=960,
        intermediate_size=3840,
        num_hidden_layers=32,
        num_attention_heads=15,
        head_dim=None,  # 960 // 15 = 64
        num_key_value_heads=5,
        hidden_act="gelu",
        max_position_embeddings=4096,
        initializer_range=0.0134,
        norm_eps=1e-5,
        use_cache=True,
        pad_token_id = 0,
        bos_token_id = 1,
        eos_token_id = 2,
        tie_word_embeddings=True,
        rope_theta=10_000.0,
        partial_rotary_factor=1,
        attention_bias=True,
        attention_dropout=0.0,
        mlp_bias=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        rope_config_validation(self)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_output_gate = False

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

__all__ = ["BiatronConfig"]