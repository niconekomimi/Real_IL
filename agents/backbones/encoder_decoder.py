import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import einops
import math
from typing import Optional
from torch.nn import functional as F
import logging

from agents.utils.time_embedding import BESO_TimeEmbedding, RF_TimeEmbedding

logger = logging.getLogger(__name__)


class LowRankAdapter(nn.Module):
    """
    Bottleneck-style low-rank adapter:
        y = x + W_up( act( W_down( LN(x) ) ) )
    """
    def __init__(
        self,
        d_model: int,
        r: int = 16,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.down = nn.Linear(d_model, r, bias=False)
        self.up = nn.Linear(r, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 初始化为近似 identity（不破坏原模型）
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.dropout(self.act(self.down(self.norm(x)))))

class SigmaAwareAdapter(nn.Module):
    def __init__(self, d_model, r=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, r)
        self.up = nn.Linear(r, d_model)
        self.sigma_proj = nn.Linear(d_model, r)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, sigma_emb):
        h = self.down(self.norm(x))
        s = self.sigma_proj(sigma_emb).unsqueeze(1)
        return x + self.up(h + s)

# Non-diffusion based encoder-decoder model
class EncDec(nn.Module):
    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            state_dim: int,
            goal_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            linear_output: bool = False,
            forward_type: str = 'cross_attn',  # cross_attn, context_token
            # ---- optional low-rank adapter (disabled by default) ----
            use_adapter: bool = False,
            adapter_rank: int = 16,
            adapter_dropout: float = 0.0,
            adapter_activation: str = "gelu",
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)
        self.decoder = hydra.utils.instantiate(decoder)

        self.device = device

        # mainly used for language condition or goal image condition
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0

        # the seq_size is the number of tokens in the input sequence
        self.seq_size = goal_seq_len + obs_seq_len + action_seq_len

        # linear embedding for the state
        self.tok_emb = nn.Linear(state_dim, embed_dim)

        # linear embedding for the goal
        self.goal_emb = nn.Linear(goal_dim, embed_dim)

        # position embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # get an action embedding
        self.query_embed = nn.Embedding(action_seq_len, embed_dim)

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        self.forward_type = forward_type

        if self.forward_type != 'cross_attn':
            self.context_embed = nn.Embedding(1, embed_dim)

        # ---- adapter: encoder_output -> (adapter) -> decoder ----
        self.adapter: Optional[nn.Module] = None
        if use_adapter:
            self.adapter = LowRankAdapter(
                d_model=embed_dim,
                r=adapter_rank,
                dropout=adapter_dropout,
                activation=adapter_activation,
                use_layernorm=True,
            )

        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def enc_only_forward(self, states, goals):
        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        b, t, dim = states.size()

        if self.goal_conditioned:
            goal_embed = self.goal_emb(goals)
            goal_x = self.drop(goal_embed + self.pos_emb[:, :self.goal_seq_len, :])

        state_embed = self.tok_emb(states)
        state_x = self.drop(state_embed + self.pos_emb[:, self.goal_seq_len:(self.goal_seq_len + t), :])

        if self.goal_conditioned:
            input_seq = torch.cat([goal_x, state_x], dim=1)
        else:
            input_seq = state_x

        encoder_output = self.encoder(input_seq)

        if self.adapter is not None:
            encoder_output = self.adapter(encoder_output)

        return encoder_output

    # decode the action sequence with cross attention over the encoder output
    def cross_attn_forward(self, states, goals):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        b, t, dim = states.size()

        if self.goal_conditioned:
            goal_embed = self.goal_emb(goals)
            goal_x = self.drop(goal_embed + self.pos_emb[:, :self.goal_seq_len, :])

        state_embed = self.tok_emb(states)
        state_x = self.drop(state_embed + self.pos_emb[:, self.goal_seq_len:(self.goal_seq_len + t), :])

        if self.goal_conditioned:
            input_seq = torch.cat([goal_x, state_x], dim=1)
        else:
            input_seq = state_x

        encoder_output = self.encoder(input_seq)

        if self.adapter is not None:
            encoder_output = self.adapter(encoder_output)

        # decode the action sequence with cross attention over the encoder output
        action_seq = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        decoder_output = self.decoder(action_seq, encoder_output)

        pred_actions = self.action_pred(decoder_output[:, -self.action_seq_len:, :])

        return pred_actions

    # add context token to the encoder, put the context token in the decoder's inputs
    def context_token_forward(self, states, goals):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        b, t, dim = states.size()

        if self.goal_conditioned:
            goal_embed = self.goal_emb(goals)
            goal_x = self.drop(goal_embed + self.pos_emb[:, :self.goal_seq_len, :])

        state_embed = self.tok_emb(states)
        state_x = self.drop(state_embed + self.pos_emb[:, self.goal_seq_len:(self.goal_seq_len + t), :])

        context_token = self.context_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        if self.goal_conditioned:
            input_seq = torch.cat([goal_x, state_x, context_token], dim=1)
        else:
            input_seq = torch.cat([state_x, context_token], dim=1)

        # only output the context token
        encoder_output = self.encoder(input_seq)[:, -1:, :]

        if self.adapter is not None:
            encoder_output = self.adapter(encoder_output)

        # decode the action sequence with cross attention over the encoder output
        action_seq = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        decoder_output = self.decoder(action_seq, encoder_output)

        pred_actions = self.action_pred(decoder_output[:, -self.action_seq_len:, :])

        return pred_actions

    def forward(
            self,
            states,
            goals=None,
    ):

        if self.forward_type == 'cross_attn':
            return self.cross_attn_forward(states, goals)
        elif self.forward_type == 'context_token':
            return self.context_token_forward(states, goals)
        else:
            raise ValueError(f"Invalid forward type: {self.forward_type}")


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding for ddpm/ddim."""
    def __init__(self, embed_dim: int, max_period: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor):
        # 支持 t 形状：(B,), (B,1) 或 (B,1,1)
        if t.dim() == 1:
            t = t.view(-1, 1, 1)
        elif t.dim() == 2:
            t = t.unsqueeze(1)
        device = t.device
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(0, half, device=device).float() / half)
        t_scalar = t[..., :1]  # 只取一个通道作为时间标量
        args = t_scalar * freqs.view(1, 1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.size(-1) < self.embed_dim:
            pad = torch.zeros(emb.size(0), emb.size(1), self.embed_dim - emb.size(-1), device=device)
            emb = torch.cat([emb, pad], dim=-1)
        return emb


# Diffusion based decoder-only model, we need time embedding and noisy antions inputs here
class Noise_EncDec(nn.Module):
    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            state_dim: int,
            goal_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            linear_output: bool = False,
            use_ada_conditioning: bool = False,
            diffusion_type: str = "beso",  # 支持: ddpm, ddim, beso, rf
            forward_type: str = 'cross_attn',  # cross_attn, context_token
            # ---- optional low-rank adapter (disabled by default) ----
            use_adapter: bool = False,
            adapter_rank: int = 16,
            adapter_dropout: float = 0.0,
            adapter_activation: str = "gelu",
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)
        self.decoder = hydra.utils.instantiate(decoder)

        self.device = device

        # mainly used for language condition or goal image condition
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0

        # the seq_size is the number of tokens in the input sequence
        self.seq_size = goal_seq_len + obs_seq_len + action_seq_len

        # linear embedding for the state / goal / action
        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.goal_emb = nn.Linear(goal_dim, embed_dim)
        self.action_emb = nn.Linear(action_dim, embed_dim)

        self.diffusion_type = diffusion_type.lower()
        if self.diffusion_type == "beso":
            self.sigma_emb = BESO_TimeEmbedding(embed_dim)
        elif self.diffusion_type == "rf":
            self.sigma_emb = RF_TimeEmbedding(embed_dim)
        elif self.diffusion_type in ("ddpm", "ddim"):
            self.sigma_emb = SinusoidalTimeEmbedding(embed_dim)
        else:
            raise ValueError(f"Diffusion type {diffusion_type} is not supported")

        # position embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        self.use_ada_conditioning = use_ada_conditioning
        self.forward_type = forward_type

        if self.forward_type != 'cross_attn':
            self.context_embed = nn.Embedding(1, embed_dim)

        # ---- adapter: encoder_output -> (adapter) -> decoder ----
        self.adapter: Optional[nn.Module] = None
        if use_adapter:
            self.adapter = LowRankAdapter(
                d_model=embed_dim,
                r=adapter_rank,
                dropout=adapter_dropout,
                activation=adapter_activation,
                use_layernorm=True,
            )

        # action pred head
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def enc_only_forward(self, states, goals, sigma):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        # t for the states does not mean the time, but the number of inputs tokens
        b, t, dim = states.size()

        if self.goal_conditioned:
            goal_embed = self.goal_emb(goals)
            goal_x = self.drop(goal_embed + self.pos_emb[:, :self.goal_seq_len, :])

        state_embed = self.tok_emb(states)
        state_x = self.drop(state_embed + self.pos_emb[:, self.goal_seq_len:(self.goal_seq_len + t), :])

        emb_t = self.sigma_emb(sigma)

        if self.goal_conditioned:
            input_seq = torch.cat([goal_x, state_x], dim=1)
        else:
            input_seq = state_x

        # adaLN conditioning
        if self.use_ada_conditioning:
            encoder_output = self.encoder(input_seq)
        else:
            input_seq = torch.cat([emb_t, input_seq], dim=1)
            encoder_output = self.encoder(input_seq)

        if self.adapter is not None:
            encoder_output = self.adapter(encoder_output)

        return encoder_output

    # decode the action sequence with cross attention over the encoder output
    def cross_attn_forward(
            self,
            states,
            actions,
            goals,
            sigma
    ):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        # t for the states does not mean the time, but the number of inputs tokens
        b, t, dim = states.size()
        _, t_a, _ = actions.size()

        if self.goal_conditioned:
            goal_embed = self.goal_emb(goals)
            goal_x = self.drop(goal_embed + self.pos_emb[:, :self.goal_seq_len, :])

        state_embed = self.tok_emb(states)
        state_x = self.drop(state_embed + self.pos_emb[:, self.goal_seq_len:(self.goal_seq_len + t), :])

        action_embed = self.action_emb(actions)
        action_x = self.drop(action_embed + self.pos_emb[:, (self.goal_seq_len + t):(self.goal_seq_len + t + t_a), :])

        emb_t = self.sigma_emb(sigma)

        if self.goal_conditioned:
            input_seq = torch.cat([goal_x, state_x], dim=1)
        else:
            input_seq = state_x

        # adaLN conditioning
        if self.use_ada_conditioning:
            encoder_output = self.encoder(input_seq)
            if self.adapter is not None:
                encoder_output = self.adapter(encoder_output)
            decoder_output = self.decoder(action_x, emb_t, encoder_output)
        else:
            input_seq = torch.cat([emb_t, input_seq], dim=1)
            encoder_output = self.encoder(input_seq)
            if self.adapter is not None:
                encoder_output = self.adapter(encoder_output)
            decoder_output = self.decoder(action_x, encoder_output)

        pred_actions = self.action_pred(decoder_output[:, -self.action_seq_len:, :])

        return pred_actions

    # add context token to the encoder, put the context token in the decoder's inputs
    def context_token_forward(
            self,
            states,
            actions,
            goals,
            sigma
    ):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        # t for the states does not mean the time, but the number of inputs tokens
        b, t, dim = states.size()
        _, t_a, _ = actions.size()

        if self.goal_conditioned:
            goal_embed = self.goal_emb(goals)
            goal_x = self.drop(goal_embed + self.pos_emb[:, :self.goal_seq_len, :])

        state_embed = self.tok_emb(states)
        state_x = self.drop(state_embed + self.pos_emb[:, self.goal_seq_len:(self.goal_seq_len + t), :])

        context_token = self.context_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        action_embed = self.action_emb(actions)
        action_x = self.drop(action_embed + self.pos_emb[:, (self.goal_seq_len + t):(self.goal_seq_len + t + t_a), :])

        emb_t = self.sigma_emb(sigma)

        if self.goal_conditioned:
            input_seq = torch.cat([goal_x, state_x, context_token], dim=1)
        else:
            input_seq = torch.cat([state_x, context_token], dim=1)

        # adaLN conditioning
        if self.use_ada_conditioning:
            encoder_output = self.encoder(input_seq)[:, -1:, :]
            if self.adapter is not None:
                encoder_output = self.adapter(encoder_output)
            emb_t = emb_t + encoder_output
            decoder_output = self.decoder(action_x, emb_t)
        else:
            input_seq = torch.cat([emb_t, input_seq], dim=1)
            encoder_output = self.encoder(input_seq)[:, -1:, :]
            if self.adapter is not None:
                encoder_output = self.adapter(encoder_output)
            decoder_output = self.decoder(action_x, encoder_output)

        pred_actions = self.action_pred(decoder_output[:, -self.action_seq_len:, :])

        return pred_actions

    def forward(
            self,
            states,
            actions,
            goals,
            sigma
    ):

        if self.forward_type == 'cross_attn':
            return self.cross_attn_forward(states, actions, goals, sigma)
        elif self.forward_type == 'context_token':
            return self.context_token_forward(states, actions, goals, sigma)
        else:
            raise ValueError(f"Invalid forward type: {self.forward_type}")
