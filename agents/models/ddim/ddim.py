import torch
import torch.nn as nn
from typing import Optional, Union
from omegaconf import DictConfig
import hydra
from .utils import linear_beta_schedule, cosine_beta_schedule, vp_beta_schedule, extract, Losses

class Diffusion(nn.Module):
    """
    Diffusion 模型：
    - 训练使用 DDPM loss
    - 推理使用 DDIM 采样
    """

    def __init__(
        self,
        inner_model: Optional[Union[nn.Module, DictConfig]] = None,
        backbones: Optional[DictConfig] = None,  # 兼容旧配置
        state_dim: int = None,
        action_dim: int = None,
        beta_schedule: str = "cosine",
        n_timesteps: int = 8,
        loss_type: str = "l2",
        predict_epsilon: bool = True,
        device: str = "cuda",
        eta: float = 0.0,  # 新增：DDIM eta，0 表示确定性
        **kwargs,  # 吞掉其它无关键，避免 Hydra 合并残留导致报错
    ):
        super().__init__()
        # 统一 inner_model 来源：优先使用 inner_model，没有则用 backbones
        if inner_model is None and backbones is not None:
            inner_model = backbones

        if isinstance(inner_model, DictConfig):
            inner_model = hydra.utils.instantiate(inner_model)

        if inner_model is None:
            raise ValueError("Diffusion requires 'inner_model' (or config via 'backbones').")

        self.model = inner_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta_schedule = beta_schedule
        self.n_timesteps = n_timesteps
        self.loss_type = loss_type
        self.predict_epsilon = predict_epsilon
        self.device = device
        self.eta = float(eta)  # 新增

        # beta schedule
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(n_timesteps).to(device)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(n_timesteps).to(device)
        elif beta_schedule == 'vp':
            self.betas = vp_beta_schedule(n_timesteps).to(device)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        # alphas (与 DDPM 保持一致)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # 添加 DDPM 中的其他系数（用于 DDIM）
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0).to(device)

        # DDPM loss function
        self.loss_fn = Losses[loss_type]()

    def forward(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        if_train: bool = False,
        ddim_steps: Optional[int] = None
    ) -> Union[torch.Tensor, float]:
        if if_train:
            # 训练使用 DDPM loss
            return self.diffusion_loss(perceptual_emb, action, latent_goal)
        else:
            # 推理使用 DDIM
            return self.denoise_actions(perceptual_emb, latent_goal, ddim_steps=ddim_steps)

    # ------------------- DDPM Loss (与 DDPM 保持一致) -------------------
    def diffusion_loss(self, perceptual_emb, actions, latent_goal=None, weights=1.0):
        batch_size = len(actions)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=actions.device).long()
        return self._p_losses(actions, perceptual_emb, latent_goal, t, weights)

    def _p_losses(self, x_start, state, goal, t, weights=1.0):
        """与 DDPM 完全一致的损失计算"""
        noise = torch.randn_like(x_start)
        x_noisy = self._q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(state, x_noisy, goal, t)

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)
        return loss

    def _q_sample(self, x_start, t, noise=None):
        """与 DDPM 完全一致的前向扩散"""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # ------------------- DDIM Sampling (修改版) -------------------
    def denoise_actions(self, perceptual_emb, latent_goal, ddim_steps=None):
        """基于 DDPM 的 denoise_actions 修改"""
        batch_size = perceptual_emb.shape[0]
        
        # 使用与 DDPM 相同的形状推断逻辑
        if len(perceptual_emb.shape) == 3:
            # 如果 perceptual_emb 是 3D，假设有序列长度
            if hasattr(self.model, 'action_seq_len'):
                shape = (batch_size, self.model.action_seq_len, self.action_dim)
            else:
                # 回退到使用 perceptual_emb 的序列长度
                shape = (batch_size, perceptual_emb.shape[1], self.action_dim)
        else:
            # 如果是 2D，使用单步动作
            shape = (batch_size, self.action_dim)
        
        return self._ddim_sample_loop(perceptual_emb, latent_goal, shape, ddim_steps)

    @torch.no_grad()
    def _ddim_sample_loop(self, state, goal, shape, ddim_steps: Optional[int] = None):
        """DDIM 采样循环"""
        batch_size = shape[0]
        if ddim_steps is None:
            ddim_steps = self.n_timesteps  # 兜底
        steps = ddim_steps
        
        # 初始化噪声
        x = torch.randn(shape, device=self.device)
        
        # 创建时间步序列
        if steps == self.n_timesteps:
            # 使用所有时间步
            timesteps = list(range(self.n_timesteps))[::-1]
        else:
            # 均匀采样时间步
            timesteps = torch.linspace(0, self.n_timesteps - 1, steps).long().tolist()[::-1]

        for i in timesteps:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self._ddim_sample_step(x, t, state, goal)

        return x

    @torch.no_grad()
    def _ddim_sample_step(self, x_t, t, state, goal):
        """DDIM 单步采样"""
        # 预测噪声或 x0
        model_output = self.model(state, x_t, goal, t)
        
        # 根据预测类型处理
        if self.predict_epsilon:
            # 从噪声预测 x0
            x0_pred = (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                      extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * model_output)
            eps = model_output
        else:
            # 直接预测 x0
            x0_pred = model_output
            eps = (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0_pred) / \
                  extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        # DDIM 更新公式
        alpha_t = extract(self.alphas_cumprod, t, x_t.shape)
        alpha_prev = extract(self.alphas_cumprod_prev, t, x_t.shape)
        
        # DDIM 方差（eta=0 为确定性）
        if getattr(self, "eta", 0.0) == 0.0:
            sigma_t = torch.zeros_like(alpha_t)
        else:
            sigma_t = self.eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        
        # 预测均值
        pred_dir = torch.sqrt(1 - alpha_prev - sigma_t ** 2) * eps
        x_prev = torch.sqrt(alpha_prev) * x0_pred + pred_dir
        
        # 添加噪声（如果 eta > 0）
        if self.eta > 0:
            noise = torch.randn_like(x_t)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))
            x_prev = x_prev + nonzero_mask * sigma_t * noise
            
        return x_prev

