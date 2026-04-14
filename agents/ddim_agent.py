import logging
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class DDIM_Agent(BaseAgent):
    """
    Agent that trains with the usual DDIM forward-noising objective,
    but uses DDIM sampling at inference.
    """

    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        optimization: DictConfig,
        obs_seq_len: int,
        act_seq_len: int,
        cam_names: list[str],
        replan_every: int | None = None,
        overlap_blend_steps: int = 0,
        overlap_blend_new_weight_start: float = 0.2,
        overlap_blend_new_weight_end: float = 1.0,
        overlap_blend_mode: str = "linear",
        overlap_blend_power: float = 2.0,
        verify_eye_in_hand: bool = False,
        if_robot_states: bool = False,
        if_film_condition: bool = False,
        device: str = "cpu",
        state_dim: int = 7,
        latent_dim: int = 64,
        ddim_steps: int | None = None,
        eta: float = 0.0,
    ):
        super().__init__(
            model=model,
            obs_encoders=obs_encoders,
            language_encoders=language_encoders,
            device=device,
            state_dim=state_dim,
            latent_dim=latent_dim,
            obs_seq_len=obs_seq_len,
            act_seq_len=act_seq_len,
            cam_names=cam_names,
            replan_every=replan_every,
            overlap_blend_steps=overlap_blend_steps,
            overlap_blend_new_weight_start=overlap_blend_new_weight_start,
            overlap_blend_new_weight_end=overlap_blend_new_weight_end,
            overlap_blend_mode=overlap_blend_mode,
            overlap_blend_power=overlap_blend_power,
            verify_eye_in_hand=verify_eye_in_hand,
        )

        self.if_robot_states = if_robot_states
        self.if_film_condition = if_film_condition

        self.eval_model_name = "eval_best_bc.pth"
        self.last_model_name = "last_bc.pth"

        self.optimizer_config = optimization
        self.use_lr_scheduler = False

        # ---------------- DDIM 配置（收敛版） ----------------
        self.ddim_steps = ddim_steps
        self.eta = float(eta)

        # eta：Diffusion(self.eta) 这种实现通常会用到，保留注入
        if hasattr(self.model, "eta"):
            self.model.eta = self.eta
        else:
            log.warning("Model has no attribute `eta`; configured eta=%.4f may not take effect.", self.eta)

        # ddim_steps：以 forward 显式参数为准，这里不再尝试写入 model.ddim_steps，避免误导
        if hasattr(self.model, "ddim_steps"):
            log.warning(
                "Model has attribute `ddim_steps`, but DDIM_Agent will pass `ddim_steps` via forward() "
                "and will not set model.ddim_steps in __init__."
            )

    def set_scaler(self, scaler):
        self.scaler = scaler
        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device).float()
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device).float()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        return optimizer

    def forward(self, obs_dict, actions=None):
        perceptual_emb, latent_goal = self.compute_input_embeddings(obs_dict)

        if self.training:
            if actions is None:
                raise ValueError("Training mode requires `actions` for diffusion loss.")
            return self.model(perceptual_emb, latent_goal, action=actions, if_train=True)

        # 推理：始终显式传入 ddim_steps；None 则由模型内部回退到 n_timesteps
        return self.model(
            perceptual_emb,
            latent_goal,
            if_train=False,
            ddim_steps=self.ddim_steps,
        )
