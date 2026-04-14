# Copyright (c) 2024, Tri Dao, Albert Gu.
from typing import Optional

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError(
        "Please install mamba-ssm and causal-conv1d. "
        "e.g., `pip install mamba-ssm causal-conv1d`"
    ) from e


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero modulation for conditioning.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Initialize weights and biases to zero
        # nn.init.zeros_(self.modulation[1].weight)
        # nn.init.zeros_(self.modulation[1].bias)

    def forward(self, c):
        return self.modulation(c).chunk(6, dim=-1)


def modulate(x, shift, scale):
    return shift + (x * (scale))


class ConditionedBlock(Block):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):

        super().__init__(dim, mixer_cls, mlp_cls, norm_cls, fused_add_norm, residual_in_fp32)

        self.adaLN_zero = AdaLNZero(dim)

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, cond=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(cond)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

            hidden_states = modulate(hidden_states, shift_msa, scale_msa)

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )

        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        hidden_states = gate_msa * hidden_states

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))

                hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)

                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )

            hidden_states = self.mlp(hidden_states)
            hidden_states = gate_mlp * hidden_states

        return hidden_states, residual


class MambaFiLMDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        resid_pdrop: float = 0.1,
        adaLN_zero: bool = False,  # NEW: 默认关闭，不破坏原结构
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.adaLN_zero = adaLN_zero

        # 保持你原来的 norm / dropout
        self.in_norm = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(resid_pdrop)

        # 时间嵌入 -> FiLM 参数（仍保留，便于兼容 if_film_condition）
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim)
        )

        mixer_cls = lambda dim: Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        mlp_cls = nn.Identity  # Decoder 这里先不额外加 MLP（最小改动）

        if self.adaLN_zero:
            # 使用带 AdaLN-Zero 的 Block
            self.layers = nn.ModuleList([
                ConditionedBlock(
                    dim=embed_dim,
                    mixer_cls=mixer_cls,
                    mlp_cls=mlp_cls,
                    norm_cls=nn.LayerNorm,
                    fused_add_norm=fused_add_norm,
                    residual_in_fp32=residual_in_fp32,
                )
                for _ in range(n_layers)
            ])
        else:
            # 保持原实现：纯 Mamba 层堆叠
            self.layers = nn.ModuleList([
                Mamba(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
                for _ in range(n_layers)
            ])

    def _apply_film(self, x: torch.Tensor, temb: torch.Tensor | None):
        if temb is None:
            return x
        gamma, beta = self.film(temb).chunk(2, dim=-1)
        return x * (1 + gamma) + beta

    def forward(self, x: torch.Tensor, cond1: torch.Tensor | None = None, cond2: torch.Tensor | None = None):
        temb, memory = None, None
        if cond2 is not None:
            temb, memory = cond1, cond2
        else:
            if cond1 is not None:
                if cond1.dim() == 3 and cond1.size(-1) == self.embed_dim:
                    memory = cond1
                else:
                    temb = cond1

        if memory is not None:
            x = torch.cat([memory, x], dim=1)

        # ---------- AdaLN-Zero 路径 ----------
        if self.adaLN_zero:
            if temb is None:
                raise ValueError("adaLN_zero=True 需要传入 temb (B, 1, D) 作为 cond。")

            # 这里使用 ConditionedBlock 的 residual 语义：第一个 block residual=None
            hidden_states = x
            residual = None
            for blk in self.layers:
                hidden_states, residual = blk(hidden_states, residual=residual, cond=temb)
                hidden_states = self.drop(hidden_states)
            return self.out_norm(hidden_states)

        # ---------- 原 FiLM 路径（不变） ----------
        h = x
        for layer in self.layers:
            y = layer(self.in_norm(h))
            y = self._apply_film(y, temb)
            h = h + self.drop(y)
        return self.out_norm(h)