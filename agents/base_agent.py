import abc
import logging
import os
import pickle
from collections import deque

import einops
import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig

from agents.utils.scaler import ActionScaler, MinMaxScaler, Scaler

# A logger for this file
log = logging.getLogger(__name__)


class BaseAgent(nn.Module, abc.ABC):

    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        device: str,
        state_dim: int,
        latent_dim: int,
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
    ):
        super().__init__()

        self.device = device
        self.working_dir = os.getcwd()
        self.scaler = None

        # Initialize model and encoder
        self.img_encoder = hydra.utils.instantiate(obs_encoders).to(device)
        self.language_encoder = hydra.utils.instantiate(language_encoders).to(device)
        self.model = hydra.utils.instantiate(model).to(device)
        self.state_emb = nn.Linear(state_dim, latent_dim)

        self.cam_names = cam_names

        # default flags (children typically override)
        self.if_robot_states = False
        self.if_film_condition = False

        # Optional runtime checks
        self.verify_eye_in_hand = bool(verify_eye_in_hand)
        self._logged_verify_eye_in_hand = False

        # for inference
        self.rollout_step_counter = 0
        self.act_seq_len = act_seq_len
        self.obs_seq_len = obs_seq_len

        # Closed-loop control: replan every N env steps (default keeps old behavior)
        if replan_every is None:
            self.replan_every = int(act_seq_len)
        else:
            self.replan_every = int(replan_every)
        if self.replan_every <= 0:
            raise ValueError(f"replan_every must be >= 1, got {self.replan_every}")

        # We only cache act_seq_len actions; force replanning no later than that.
        if self.replan_every > self.act_seq_len:
            log.warning(
                "replan_every (%d) > act_seq_len (%d); clamping replan_every to act_seq_len.",
                self.replan_every,
                self.act_seq_len,
            )
            self.replan_every = self.act_seq_len

        self.overlap_blend_steps = int(overlap_blend_steps)
        if self.overlap_blend_steps < 0:
            raise ValueError(
                f"overlap_blend_steps must be >= 0, got {self.overlap_blend_steps}"
            )
        self.overlap_blend_new_weight_start = float(overlap_blend_new_weight_start)
        self.overlap_blend_new_weight_end = float(overlap_blend_new_weight_end)
        self.overlap_blend_mode = str(overlap_blend_mode).strip().lower() or "linear"
        self.overlap_blend_power = float(overlap_blend_power)
        if not 0.0 <= self.overlap_blend_new_weight_start <= 1.0:
            raise ValueError(
                "overlap_blend_new_weight_start must be within [0, 1], "
                f"got {self.overlap_blend_new_weight_start}"
            )
        if not 0.0 <= self.overlap_blend_new_weight_end <= 1.0:
            raise ValueError(
                "overlap_blend_new_weight_end must be within [0, 1], "
                f"got {self.overlap_blend_new_weight_end}"
            )
        if self.overlap_blend_new_weight_end < self.overlap_blend_new_weight_start:
            raise ValueError(
                "overlap_blend_new_weight_end must be >= overlap_blend_new_weight_start, "
                f"got {self.overlap_blend_new_weight_end} < {self.overlap_blend_new_weight_start}"
            )
        if self.overlap_blend_mode not in {"linear", "cosine", "ease_out_power"}:
            raise ValueError(
                "overlap_blend_mode must be one of {'linear', 'cosine', 'ease_out_power'}, "
                f"got {self.overlap_blend_mode!r}"
            )
        if self.overlap_blend_power <= 0.0:
            raise ValueError(
                f"overlap_blend_power must be > 0, got {self.overlap_blend_power}"
            )
        if self.overlap_blend_steps > 0:
            if self.replan_every >= self.act_seq_len:
                log.warning(
                    "overlap_blend_steps=%d is configured, but replan_every (%d) >= act_seq_len (%d); "
                    "overlap blending will stay inactive.",
                    self.overlap_blend_steps,
                    self.replan_every,
                    self.act_seq_len,
                )
            else:
                log.info(
                    "Enabled overlap blending: steps=%d, replan_every=%d, act_seq_len=%d, "
                    "mode=%s, power=%.3f, new_weight_start=%.3f, new_weight_end=%.3f",
                    self.overlap_blend_steps,
                    self.replan_every,
                    self.act_seq_len,
                    self.overlap_blend_mode,
                    self.overlap_blend_power,
                    self.overlap_blend_new_weight_start,
                    self.overlap_blend_new_weight_end,
                )

        self._steps_since_plan = 0
        self.pred_action_seq: torch.Tensor | None = None

        self.obs_seq: dict[str, deque[torch.Tensor]] = {}

    def set_scaler(self, scaler):
        self.scaler = scaler

    # @abc.abstractmethod
    def compute_input_embeddings(self, obs_dict):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        #########################################
        # deal with language embedding
        #########################################
        if "lang" in obs_dict:
            obs_dict["lang_emb"] = self.language_encoder(obs_dict["lang"]).float()

        latent_goal = obs_dict["lang_emb"]

        if "point_cloud" in obs_dict and "robot0_agentview_left_image" in obs_dict:

            assert obs_dict["point_cloud"].shape[-1] in [3, 6], "Point cloud should have 3 or 6 channels"

            obs_dict["point_cloud"] = einops.rearrange(obs_dict["point_cloud"], "b t n d -> (b t) n d")

            B, T, C, H, W = obs_dict[f"{self.cam_names[0]}_image"].shape
            # B, T, C, H, W = obs_dict["robot0_agentview_center_image"].shape
            for camera in self.cam_names:
                obs_dict[f"{camera}_image"] = obs_dict[f"{camera}_image"].view(B * T, C, H, W)

            if self.if_film_condition:
                perceptual_emb = self.img_encoder(obs_dict, latent_goal)
            else:
                perceptual_emb = self.img_encoder(obs_dict)

            return perceptual_emb, latent_goal

        #########################################
        # using RGB images or point clouds
        #########################################
        if "point_cloud" in obs_dict:
            assert obs_dict["point_cloud"].shape[-1] in [3, 6], "Point cloud should have 3 or 6 channels"

            pc = einops.rearrange(obs_dict["point_cloud"], "b t n d -> (b t) n d")
            if self.if_film_condition:
                perceptual_emb = self.img_encoder(pc, latent_goal)
            else:
                perceptual_emb = self.img_encoder(pc)

        elif self.cam_names is not None:

            if self.verify_eye_in_hand and not self._logged_verify_eye_in_hand:
                # Validate that eye_in_hand is both present in inputs and expected by the encoder.
                expected_key = "eye_in_hand_image"
                if "eye_in_hand" in self.cam_names:
                    if expected_key not in obs_dict:
                        raise KeyError(
                            f"verify_eye_in_hand=True but `{expected_key}` is missing from obs_dict keys: {list(obs_dict.keys())}"
                        )
                    if hasattr(self.img_encoder, "rgb_keys") and expected_key not in getattr(self.img_encoder, "rgb_keys"):
                        log.warning(
                            "verify_eye_in_hand=True: `%s` present in obs_dict, but not listed in img_encoder.rgb_keys=%s. "
                            "This may indicate the encoder config (shape_meta) is not using eye-in-hand.",
                            expected_key,
                            getattr(self.img_encoder, "rgb_keys"),
                        )
                    else:
                        log.info(
                            "verify_eye_in_hand=True: using `%s` with shape=%s and encoder=%s",
                            expected_key,
                            tuple(obs_dict[expected_key].shape),
                            type(self.img_encoder).__name__,
                        )
                self._logged_verify_eye_in_hand = True

            B, T, C, H, W = obs_dict[f"{self.cam_names[0]}_image"].shape
            # B, T, C, H, W = obs_dict["robot0_agentview_center_image"].shape

            for camera in self.cam_names:
                obs_dict[f"{camera}_image"] = obs_dict[f"{camera}_image"].view(B * T, C, H, W)

            # for camera in obs_dict.keys():
            #     if "rgb" not in camera and "image" not in camera:
            #         continue
            #     # print(obs_dict[camera].shape)
            #     obs_dict[camera] = obs_dict[camera].view(B * T, C, H, W)

            # print(self.if_film_condition)
            if self.if_film_condition:
                perceptual_emb = self.img_encoder(obs_dict, latent_goal)
            else:
                # obs_dict is a dict with two images and one lang: images are [64,3,256,256]
                perceptual_emb = self.img_encoder(obs_dict)
        else:
            raise NotImplementedError("Either use point clouds or images as input.")

        #########################################
        # add robot states
        #########################################
        if self.if_robot_states and "robot_states" in obs_dict.keys():
            robot_states = obs_dict["robot_states"]
            robot_states = self.state_emb(robot_states)

            perceptual_emb = torch.cat([perceptual_emb, robot_states], dim=1)

        return perceptual_emb, latent_goal

    @abc.abstractmethod
    def forward(self, obs_dict: dict[str, torch.Tensor], actions=None) -> torch.Tensor:
        """
        Forward pass of the model
        """
        pass

    def reset(self):
        """Resets the context of the model."""
        self.rollout_step_counter = 0
        self._steps_since_plan = 0
        self.pred_action_seq = None
        self.obs_seq: dict[str, deque[torch.Tensor]] = {}

    def _blend_replanned_action_seq(
        self,
        previous_plan: torch.Tensor | None,
        new_plan: torch.Tensor,
    ) -> torch.Tensor:
        if previous_plan is None or self.overlap_blend_steps <= 0:
            return new_plan
        if self.replan_every >= self.act_seq_len:
            return new_plan

        previous_plan_len = int(previous_plan.shape[1]) if previous_plan.ndim >= 2 else 0
        new_plan_len = int(new_plan.shape[1]) if new_plan.ndim >= 2 else 0
        available_old_tail = max(0, previous_plan_len - self.replan_every)
        overlap_len = min(self.overlap_blend_steps, available_old_tail, new_plan_len)
        if overlap_len <= 0:
            return new_plan

        old_tail = previous_plan[
            :, self.replan_every:self.replan_every + overlap_len, :
        ]
        if old_tail.shape[1] != overlap_len:
            return new_plan

        blended_plan = new_plan.clone()
        twist_dims = min(6, blended_plan.shape[-1], old_tail.shape[-1])
        if twist_dims <= 0:
            return blended_plan

        curve_t = torch.linspace(
            0.0,
            1.0,
            overlap_len,
            device=blended_plan.device,
            dtype=blended_plan.dtype,
        )
        if self.overlap_blend_mode == "linear":
            curve = curve_t
        elif self.overlap_blend_mode == "cosine":
            curve = torch.sin(0.5 * torch.pi * curve_t)
        elif self.overlap_blend_mode == "ease_out_power":
            curve = 1.0 - torch.pow(1.0 - curve_t, self.overlap_blend_power)
        else:
            raise RuntimeError(f"Unsupported overlap_blend_mode: {self.overlap_blend_mode}")

        alpha = (
            self.overlap_blend_new_weight_start
            + (self.overlap_blend_new_weight_end - self.overlap_blend_new_weight_start) * curve
        ).view(1, overlap_len, 1)
        blended_plan[:, :overlap_len, :twist_dims] = (
            (1.0 - alpha) * old_tail[:, :, :twist_dims]
            + alpha * new_plan[:, :overlap_len, :twist_dims]
        )
        return blended_plan

    @torch.no_grad()
    def predict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # initialize self.obs_seq if empty
        if not self.obs_seq:
            for key in obs_dict.keys():
                self.obs_seq[key] = deque(maxlen=self.obs_seq_len)

        for key in obs_dict.keys():
            self.obs_seq[key].append(obs_dict[key])

            if key == "lang":
                continue
            obs_dict[key] = torch.concat(list(self.obs_seq[key]), dim=1)

            if obs_dict[key].shape[1] < self.obs_seq_len:
                # left pad repeated first element
                pad = einops.repeat(
                    obs_dict[key][:, 0],
                    "b ... -> b t ...",
                    t=self.obs_seq_len - obs_dict[key].shape[1],
                )
                obs_dict[key] = torch.cat([pad, obs_dict[key]], dim=1)
                
        # Receding-horizon control: replan every `replan_every` env steps.
        if self._steps_since_plan == 0:
            self.eval()
            previous_plan = self.pred_action_seq
            pred_action_seq = self(obs_dict)[:, : self.act_seq_len]
            pred_action_seq = self.scaler.inverse_scale_output(pred_action_seq)
            pred_action_seq = self._blend_replanned_action_seq(previous_plan, pred_action_seq)
            self.pred_action_seq = pred_action_seq

        # Execute the cached plan.
        plan_len = self.pred_action_seq.shape[1]
        plan_idx = min(self._steps_since_plan, plan_len - 1)
        current_action = self.pred_action_seq[0, plan_idx]

        self._steps_since_plan += 1
        if self._steps_since_plan >= self.replan_every:
            self._steps_since_plan = 0

        # Keep rollout_step_counter for backward-compat / logging (0..act_seq_len-1)
        self.rollout_step_counter = (self.rollout_step_counter + 1) % self.act_seq_len

        return current_action

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load pretrained weights for the entire agent
        """
        path = os.path.join(
            weights_path,
            "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth",
        )
        self.load_state_dict(torch.load(path, weights_only=True))
        log.info("Loaded pre-trained model")

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the weights of the entire agent
        """
        path = os.path.join(
            store_path, "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth"
        )
        torch.save(self.state_dict(), path)
        log.info(f"Model saved to: {store_path}")

    def store_model_scaler(self, store_path: str, sv_name=None) -> None:
        """
        Store the model scaler inside the store path as model_scaler.pkl
        """
        save_path = os.path.join(
            store_path, "model_scaler.pkl" if sv_name is None else sv_name
        )
        with open(save_path, "wb") as f:
            pickle.dump(self.scaler, f)
        log.info(f"Model scaler saved to: {save_path}")

    def load_model_scaler(self, weights_path: str, sv_name=None) -> None:
        """
        Load the model scaler from the weights path
        """
        if sv_name is None:
            sv_name = "model_scaler.pkl"

        with open(os.path.join(weights_path, sv_name), "rb") as f:
            self.scaler = pickle.load(f)
        log.info("Loaded model scaler")

    def get_params(self):

        total_params = sum(p.numel() for p in self.parameters())

        wandb.log({"model parameters": total_params})

        log.info("The model has a total amount of {} parameters".format(total_params))

    @property
    def get_model_state_dict(self) -> dict:
        return self.state_dict()

    @property
    def get_scaler(self) -> Scaler:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return self.scaler

    @property
    def get_model_state(self) -> tuple[dict, Scaler]:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return (self.state_dict(), self.get_scaler)

    def recover_model_state(self, model_state, scaler):
        self.load_state_dict(model_state)
        self.set_scaler(scaler)
