import torch
from Biatron.modeling_biatron import BiatronForCausalLM

def split_qkv_weights( provider_config,  qkv: torch.Tensor):
    """Split Megatron's interleaved QKV tensor into separate Q, K, V matrices.

    Args:
        provider_config (TransformerConfig): Model configuration provider_config.
        qkv (torch.Tensor): Interleaved QKV weights in Megatron format.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of (Q, K, V)
            weight matrices.
    """
    head_num = provider_config.num_attention_heads
    num_query_groups = provider_config.num_key_value_heads
    heads_per_group = head_num // num_query_groups
    head_size = provider_config.hidden_size // head_num
    if getattr(provider_config, "attention_output_gate", False):
        qkv_total_dim = 2 * head_num + 2 * num_query_groups
        total_heads_per_group = 2 * heads_per_group + 2
    else:
        qkv_total_dim = head_num + 2 * num_query_groups
        total_heads_per_group = heads_per_group + 2
    is_bias = qkv.ndim == 1

    if is_bias:
        hidden_size = 1
        qkv_reshaped = qkv.view(qkv_total_dim, head_size)
    else:
        hidden_size = qkv.shape[-1]
        qkv_reshaped = qkv.view(qkv_total_dim, head_size, hidden_size)


    # Extract Q, K, V from interleaved pattern
    q_slice = torch.cat(
        [
            torch.arange(total_heads_per_group * i, total_heads_per_group * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(total_heads_per_group - 2, qkv_total_dim, total_heads_per_group)
    v_slice = torch.arange(total_heads_per_group - 1, qkv_total_dim, total_heads_per_group)

    if getattr(provider_config, "attention_output_gate", False):
        z_slice = torch.cat(
            [
                torch.arange(
                    total_heads_per_group * i + heads_per_group,
                    total_heads_per_group * i + heads_per_group * 2,
                )
                for i in range(num_query_groups)
            ]
        )
        # In HF implementation, matrix Q and Z are mixed, so we need to concatenate them.
        q = torch.cat([qkv_reshaped[q_slice], qkv_reshaped[z_slice]], dim=1)
    else:
        q = qkv_reshaped[q_slice]
    k = qkv_reshaped[k_slice]
    v = qkv_reshaped[v_slice]

    assert q.numel() + k.numel() + v.numel() == qkv.numel(), (
        f"QKV weights are not correctly merged, {q.shape=}, {k.shape=}, {v.shape=}, {qkv.shape=}"
    )
    
    if is_bias:
        q = q.reshape(-1)
        k = k.reshape(-1)
        v = v.reshape(-1)
    else:
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)
    return q, k, v

def split_qkv_biases(config, qkv: torch.Tensor):
    """Split Megatron's interleaved QKV bias into separate Q, K, V biases.

    Args:
        config (TransformerConfig): Transformer configuration.
        qkv (torch.Tensor): Interleaved QKV biases in Megatron format (1D
            tensor).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of (Q, K, V) bias vectors.
    """
    head_num = config.num_attention_heads
    num_query_groups = config.num_key_value_heads
    heads_per_group = head_num // num_query_groups
    head_size = config.hidden_size // head_num
    
    if config.attention_output_gate:
        qkv_total_dim = 2 * head_num + 2 * num_query_groups
        total_heads_per_group = 2 * heads_per_group + 2
    else:
        qkv_total_dim = head_num + 2 * num_query_groups
        total_heads_per_group = heads_per_group + 2

    # Reshape to expose interleaved structure
    qkv = qkv.reshape(qkv_total_dim, head_size)

    # Extract Q, K, V from interleaved pattern
    q_slice = torch.cat(
        [
            torch.arange(total_heads_per_group * i, total_heads_per_group * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(total_heads_per_group - 2, qkv_total_dim, total_heads_per_group)
    v_slice = torch.arange(total_heads_per_group - 1, qkv_total_dim, total_heads_per_group)

    if config.attention_output_gate:
        z_slice = torch.cat(
            [
                torch.arange(
                    total_heads_per_group * i + heads_per_group,
                    total_heads_per_group * i + heads_per_group * 2,
                )
                for i in range(num_query_groups)
            ]
        )
        # In HF implementation, matrix Q and Z are mixed, so we need to concatenate them.
        q = torch.cat([qkv[q_slice], qkv[z_slice]], dim=1).flatten()
    else:
        q = qkv[q_slice].flatten()
    k = qkv[k_slice].flatten()
    v = qkv[v_slice].flatten()

    return q, k, v

def load_state_dict_biatron(model: BiatronForCausalLM, path_state_dict: str):
    state_dict = torch.load(path_state_dict, map_location="cpu")
    
    model.model.embed_tokens.weight.data = state_dict["embedding.word_embeddings.weight"]

    for lidx in range(model.config.num_hidden_layers):
        # input layernorm
        assert model.model.layers[lidx].input_layernorm.weight.data.shape == state_dict[f"decoder.layers.{lidx}.input_layernorm.weight"].shape
        model.model.layers[lidx].input_layernorm.weight.data = state_dict[f"decoder.layers.{lidx}.input_layernorm.weight"]
        assert model.model.layers[lidx].input_layernorm.bias.data.shape == state_dict[f"decoder.layers.{lidx}.input_layernorm.bias"].shape
        model.model.layers[lidx].input_layernorm.bias.data = state_dict[f"decoder.layers.{lidx}.input_layernorm.bias"]

        #O_proj
        assert model.model.layers[lidx].self_attn.o_proj.weight.data.shape == state_dict[f"decoder.layers.{lidx}.self_attention.linear_proj.weight"].shape
        model.model.layers[lidx].self_attn.o_proj.weight.data = state_dict[f"decoder.layers.{lidx}.self_attention.linear_proj.weight"]
        assert model.model.layers[lidx].self_attn.o_proj.bias.data.shape == state_dict[f"decoder.layers.{lidx}.self_attention.linear_proj.bias"].shape
        model.model.layers[lidx].self_attn.o_proj.bias.data = state_dict[f"decoder.layers.{lidx}.self_attention.linear_proj.bias"]

        ## KQV
        q,k,v = split_qkv_weights(model.config, state_dict[f"decoder.layers.{lidx}.self_attention.linear_qkv.weight"])

        assert model.model.layers[lidx].self_attn.q_proj.weight.data.shape == q.shape
        model.model.layers[lidx].self_attn.q_proj.weight.data = q
        assert model.model.layers[lidx].self_attn.k_proj.weight.data.shape == k.shape
        model.model.layers[lidx].self_attn.k_proj.weight.data = k
        assert model.model.layers[lidx].self_attn.v_proj.weight.data.shape == v.shape
        model.model.layers[lidx].self_attn.v_proj.weight.data = v

        qb,kb,vb = split_qkv_biases(model.config, state_dict[f"decoder.layers.{lidx}.self_attention.linear_qkv.bias"])
        assert model.model.layers[lidx].self_attn.q_proj.bias.data.shape == qb.shape
        model.model.layers[lidx].self_attn.q_proj.bias.data = qb
        assert model.model.layers[lidx].self_attn.k_proj.bias.data.shape == kb.shape
        model.model.layers[lidx].self_attn.k_proj.bias.data = kb
        assert model.model.layers[lidx].self_attn.v_proj.bias.data.shape == vb.shape
        model.model.layers[lidx].self_attn.v_proj.bias.data = vb

        #post attention layernorm
        assert model.model.layers[lidx].post_attention_layernorm.weight.data.shape == state_dict[f"decoder.layers.{lidx}.pre_mlp_layernorm.weight"].shape
        model.model.layers[lidx].post_attention_layernorm.weight.data = state_dict[f"decoder.layers.{lidx}.pre_mlp_layernorm.weight"]
        assert model.model.layers[lidx].post_attention_layernorm.bias.data.shape == state_dict[f"decoder.layers.{lidx}.pre_mlp_layernorm.bias"].shape
        model.model.layers[lidx].post_attention_layernorm.bias.data = state_dict[f"decoder.layers.{lidx}.pre_mlp_layernorm.bias"]

        # MLP
        assert model.model.layers[lidx].mlp.up_proj.weight.data.shape == state_dict[f"decoder.layers.{lidx}.mlp.linear_fc1.weight"].shape
        model.model.layers[lidx].mlp.up_proj.weight.data = state_dict[f"decoder.layers.{lidx}.mlp.linear_fc1.weight"]
        assert model.model.layers[lidx].mlp.up_proj.bias.data.shape == state_dict[f"decoder.layers.{lidx}.mlp.linear_fc1.bias"].shape
        model.model.layers[lidx].mlp.up_proj.bias.data = state_dict[f"decoder.layers.{lidx}.mlp.linear_fc1.bias"]

        assert model.model.layers[lidx].mlp.down_proj.weight.data.shape == state_dict[f"decoder.layers.{lidx}.mlp.linear_fc2.weight"].shape
        model.model.layers[lidx].mlp.down_proj.weight.data = state_dict[f"decoder.layers.{lidx}.mlp.linear_fc2.weight"]
        assert model.model.layers[lidx].mlp.down_proj.bias.data.shape == state_dict[f"decoder.layers.{lidx}.mlp.linear_fc2.bias"].shape
        model.model.layers[lidx].mlp.down_proj.bias.data = state_dict[f"decoder.layers.{lidx}.mlp.linear_fc2.bias"]

    # final layernorm
    assert model.model.norm.weight.data.shape == state_dict["decoder.final_layernorm.weight"].shape
    model.model.norm.weight.data = state_dict["decoder.final_layernorm.weight"]
    assert model.model.norm.bias.data.shape == state_dict["decoder.final_layernorm.bias"].shape
    model.model.norm.bias.data = state_dict["decoder.final_layernorm.bias"]

    model.lm_head.weight.data = model.model.embed_tokens.weight.data

    return model