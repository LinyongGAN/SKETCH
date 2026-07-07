from transformers import PretrainedConfig
from utils.earth_computation import deg_to_rad, rad_to_deg, calculate_initial_bearing, haversine_distance, UnitConversion

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            hidden_act: str = 'silu',
            hidden_size: int = 256,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 4,
            num_key_value_heads: int = 2,
            dim_size: int = 4,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            flash_attn: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.dim_size = dim_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(1).unsqueeze(0)) + (rotate_half(q) * sin.unsqueeze(1).unsqueeze(0))
    k_embed = (k * cos.unsqueeze(1).unsqueeze(0)) + (rotate_half(k) * sin.unsqueeze(1).unsqueeze(0))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.dim_size, self.num_hidden_layers = config.dim_size, config.num_hidden_layers
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.GELU(),
            nn.Linear(config.hidden_size*2, config.hidden_size)
        )

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                hidden_states: Optional[torch.Tensor] = None,
                seq_length: Optional[int] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        hidden_states = self.mlp(hidden_states) + hidden_states

        return hidden_states, presents


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def init_weights(self, module):
        """Initialize weights with Kaiming initialization for appropriate layers"""
        if isinstance(module, nn.Linear):
            
            gain = nn.init.calculate_gain('relu')
            
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), nonlinearity='relu')
            
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        
        elif isinstance(module, RMSNorm):
            pass
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def __init__(self, config: MiniMindConfig = None):
        
        self.config1 = config
        self.config2 = config
        self.config1.dim_size = 4; self.config1.num_hidden_layers = 3
        self.config2.dim_size = 6; self.config2.hidden_size = self.config1.hidden_size; self.config2.num_hidden_layers = 1
        super().__init__(config)

        self.encoder1 = nn.Sequential(
            nn.Linear(4, config.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        )
        self.dropout1 = nn.Dropout(config.dropout)

        self.encoder2 = nn.Sequential(
            nn.Linear(2, config.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        )
        self.dropout2 = nn.Dropout(config.dropout)
        
        self.model1 = MiniMindModel(self.config1)
        self.model2 = MiniMindModel(self.config2)

        self.dense = nn.Sequential(
            nn.Linear(self.config1.hidden_size*2, self.config1.hidden_size),
            nn.GELU(),
            nn.Linear(self.config1.hidden_size, self.config1.hidden_size)   
        )

        self.decoder1 = nn.Linear(self.config2.hidden_size, self.config1.dim_size, bias=False)
        self.apply(self.init_weights)
        # self.model.embed_tokens.weight = self.lm_head.weight

    def displacement_calculation(self, coord, vol_vec):
        """
        coord: [batch_size, seq_len, 2], (lat, lon) in rad
        vol_vec: [batch_size, seq_len, 2], (sog in kn, cog in rad)
        return: displacement: [batch_size, seq_len, 2], (dlat in rad, dlon in rad)
        """
        
        time_interval = 5.0/60.0 # h
        EARTH_RADIUS = 6371/1.852 # nmi
        dlat = vol_vec[:,:,0] * time_interval / EARTH_RADIUS
        tmp = (coord[:,:,0] + vol_vec[:,:,0]*time_interval/EARTH_RADIUS) # final latitude
        dlon = vol_vec[:,:,1]/(vol_vec[:,:,0]+1e-6) * torch.log(torch.abs((1/torch.cos(tmp) + torch.tan(tmp)) / (1/torch.cos(coord[:,:,0]) + torch.tan(coord[:,:,0]))))
        
        displacement = torch.stack((dlat, dlon), dim = -1)
        return displacement.clone()
    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                dest_coord: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                part = "train",
                **args):
        
        assert part in ["train", "inference", "sft"], f"part should be either train or inference, but {part} read"

        coord = input_ids[:,:,:2].clone()
        vol_vec = input_ids[:,:,2:4].clone()
        displacement = self.displacement_calculation(coord, vol_vec)  # [batch_size, seq_len, 2]
        std_displacement = displacement.clone()
        std_displacement[:,:,0] /= math.pi/3
        std_displacement[:,:,1] /= math.pi

        std_input_ids, std_dest_coord = UnitConversion(input_ids, dest_coord, fr="rad", to="std")

        hidden_states1, _ = self.model1(
            hidden_states=self.dropout1(self.encoder1(std_input_ids)),
            seq_length=std_input_ids.shape[1],
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            **args
        )

        if part == "sft": return hidden_states1
        assert dest_coord is not None, f"dest_coord should not be None when part is training or inference, now the type is {type(dest_coord)}"

        hidden_states2 = self.dropout2(self.encoder2(std_dest_coord))

        hidden_states3, _ = self.model2(
            hidden_states=self.dense(torch.concat((hidden_states1, hidden_states2), dim=-1)),
            seq_length=std_input_ids.shape[1],
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            **args
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.decoder1(hidden_states3[:, slice_indices, :])
        
        logits[:,:,:2] = (std_input_ids[:,:,:2] + std_displacement).clone()
        logits, _ = UnitConversion(logits, None, fr="std", to="rad")
        
        pred_coord = logits[:,:,:2].clone()
        pred_vol_vec = logits[:,:,2:4].clone()
        
        new_position = logits.clone()
        
        return new_position, pred_vol_vec, displacement, pred_coord
    