import argparse
import io
import logging
import os
import pickle
import sys
import types
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


log = logging.getLogger(__name__)


def _force_noninteractive_matplotlib() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass


def _patch_inference_compat(checkpoint_dir: str) -> None:
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.SimpleNamespace(log=lambda *args, **kwargs: None)

    from agents import base_agent
    from agents.encoders import clip_lang_encoder

    state_dict = torch.load(os.path.join(checkpoint_dir, "last_model.pth"), map_location="cpu", weights_only=False)
    clip_prefix = "language_encoder.clip_rn50."
    clip_state = {
        key[len(clip_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(clip_prefix)
    }

    def _load_clip_from_checkpoint(self, model_name: str) -> None:
        if clip_state:
            self.clip_rn50 = clip_lang_encoder.build_model(dict(clip_state)).to(self.device)
            return
        model, _ = clip_lang_encoder.load_clip(model_name, device=self.device)
        self.clip_rn50 = clip_lang_encoder.build_model(model.state_dict()).to(self.device)

    def _load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        path = os.path.join(
            weights_path,
            "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth",
        )
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        log.info("Loaded pre-trained model")

    class _DeviceAwareUnpickler(pickle.Unpickler):
        def __init__(self, file_obj, map_location):
            super().__init__(file_obj)
            self.map_location = map_location

        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda payload: torch.load(
                    io.BytesIO(payload),
                    map_location=self.map_location,
                    weights_only=False,
                )
            return super().find_class(module, name)

    def _load_model_scaler(self, weights_path: str, sv_name=None) -> None:
        if sv_name is None:
            sv_name = "model_scaler.pkl"

        with open(os.path.join(weights_path, sv_name), "rb") as f:
            self.scaler = _DeviceAwareUnpickler(f, self.device).load()

        if hasattr(self.scaler, "device"):
            self.scaler.device = self.device
        for attr_name, attr_value in vars(self.scaler).items():
            if torch.is_tensor(attr_value):
                setattr(self.scaler, attr_name, attr_value.to(self.device))
        log.info("Loaded model scaler")

    clip_lang_encoder.LangClip._load_clip = _load_clip_from_checkpoint
    base_agent.BaseAgent.load_pretrained_model = _load_pretrained_model
    base_agent.BaseAgent.load_model_scaler = _load_model_scaler


class RealRobotPolicy:
    def __init__(
        self,
        checkpoint_dir: str,
        config_path: str | None = None,
        device: str | None = None,
        task_description: str | None = None,
        task_name: str | None = None,
        task_embedding_path: str | None = None,
    ):
        if config_path is None:
            config_path = os.path.join(checkpoint_dir, ".hydra", "config.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.cfg = OmegaConf.load(config_path)
        chosen_device = device or (self.cfg.device if torch.cuda.is_available() else "cpu")
        OmegaConf.set_struct(self.cfg, False)
        self.cfg.device = chosen_device
        self.cfg.agents.device = chosen_device

        self.device = torch.device(chosen_device)
        _force_noninteractive_matplotlib()
        _patch_inference_compat(checkpoint_dir)
        self.agent = hydra.utils.instantiate(self.cfg.agents)
        self.agent.to(self.device)
        self.agent.eval()
        self.agent.load_pretrained_model(checkpoint_dir, sv_name="last_model")
        self.agent.load_model_scaler(checkpoint_dir)
        self.agent.reset()

        self.default_lang = None
        self.default_lang_emb = None
        self.default_task_name = None
        self.default_task_description = None
        self.default_task_embedding_path = None
        self.set_task(
            task_name=task_name,
            task_description=task_description,
            task_embedding_path=task_embedding_path,
            reset_context=False,
        )

    @staticmethod
    def _load_task_embedding(
        task_embedding_path: str,
        task_name: str | None,
        task_description: str | None,
    ) -> np.ndarray:
        with open(task_embedding_path, "rb") as handle:
            embedding_map = pickle.load(handle)

        for key in (task_name, task_description):
            if key and key in embedding_map:
                embedding = np.asarray(embedding_map[key], dtype=np.float32)
                if embedding.ndim == 1:
                    return embedding
                if embedding.ndim == 2 and embedding.shape[0] == 1:
                    return embedding[0]
                raise ValueError(f"Task embedding for {key} must have shape [D] or [1, D]")

        raise KeyError(
            f"None of {[task_name, task_description]} were found in task embedding file {task_embedding_path}"
        )

    def reset(self) -> None:
        self.agent.reset()

    def set_task(
        self,
        task_name: str | None = None,
        task_description: str | None = None,
        task_embedding_path: str | None = None,
        *,
        reset_context: bool = True,
    ) -> None:
        normalized_task_name = str(task_name).strip() or None if task_name is not None else None
        normalized_task_description = (
            str(task_description).strip() or None if task_description is not None else None
        )
        normalized_embedding_path = (
            str(Path(task_embedding_path).expanduser().resolve()) if task_embedding_path else None
        )

        self.default_task_name = normalized_task_name
        self.default_task_description = normalized_task_description
        self.default_task_embedding_path = normalized_embedding_path
        self.default_lang = None
        self.default_lang_emb = None

        if normalized_embedding_path is not None:
            self.default_lang_emb = self._load_task_embedding(
                task_embedding_path=normalized_embedding_path,
                task_name=normalized_task_name,
                task_description=normalized_task_description,
            )
        elif normalized_task_description is not None:
            self.default_lang = normalized_task_description
        elif normalized_task_name is not None:
            self.default_lang = normalized_task_name

        if reset_context:
            self.reset()

    def predict_action(
        self,
        agentview_image: np.ndarray,
        eye_in_hand_image: np.ndarray,
        robot_states: np.ndarray | None = None,
        lang: str | None = None,
        lang_emb: np.ndarray | None = None,
    ) -> np.ndarray:
        obs_dict: dict[str, torch.Tensor | list[str]] = {
            "agentview_image": self._to_image_tensor(agentview_image),
            "eye_in_hand_image": self._to_image_tensor(eye_in_hand_image),
        }

        if robot_states is not None:
            obs_dict["robot_states"] = self._to_state_tensor(robot_states)

        effective_lang_emb = lang_emb if lang_emb is not None else self.default_lang_emb
        effective_lang = lang if lang is not None else self.default_lang

        if effective_lang_emb is not None:
            obs_dict["lang_emb"] = self._to_goal_tensor(effective_lang_emb)
        elif effective_lang is not None:
            obs_dict["lang"] = [effective_lang]
        else:
            raise ValueError("A task goal is required. Provide lang, lang_emb, task_description, or task_embedding_path.")

        with torch.no_grad():
            action = self.agent.predict(obs_dict)
        return action.detach().cpu().numpy()

    def _to_image_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image with shape [H, W, C] or [C, H, W], got {image.shape}")
        if image.shape[-1] in (1, 3):
            image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def _to_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 1:
            raise ValueError(f"Expected robot state with shape [D], got {state.shape}")
        return torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(self.device)

    def _to_goal_tensor(self, goal: np.ndarray) -> torch.Tensor:
        goal = np.asarray(goal, dtype=np.float32)
        if goal.ndim == 1:
            goal = goal[None, None, :]
        elif goal.ndim == 2 and goal.shape[0] == 1:
            goal = goal[None, :, :]
        else:
            raise ValueError(f"Expected goal embedding with shape [D] or [1, D], got {goal.shape}")
        return torch.from_numpy(goal).to(self.device)


def _load_demo_observation(hdf5_path: str, demo_key: str, step: int) -> dict:
    with h5py.File(hdf5_path, "r") as dataset_file:
        demo = dataset_file["data"][demo_key]
        obs = demo["obs"]

        robot_states = None
        if "joint_states" in obs and "gripper_states" in obs:
            robot_states = np.concatenate(
                [
                    np.asarray(obs["joint_states"][step], dtype=np.float32),
                    np.asarray(obs["gripper_states"][step], dtype=np.float32),
                ],
                axis=-1,
            )
        elif "robot_states" in demo:
            robot_states = np.asarray(demo["robot_states"][step], dtype=np.float32)
        elif "states" in demo:
            robot_states = np.asarray(demo["states"][step], dtype=np.float32)

        return {
            "agentview_image": np.asarray(obs["agentview_rgb"][step]),
            "eye_in_hand_image": np.asarray(obs["eye_in_hand_rgb"][step]),
            "robot_states": robot_states,
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a trained real-robot policy and run one-step inference.")
    parser.add_argument("--ckpt-dir", required=True, help="Run directory containing last_model.pth and model_scaler.pkl")
    parser.add_argument("--config-path", default=None, help="Optional config path, defaults to ckpt_dir/.hydra/config.yaml")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu")
    parser.add_argument("--task-name", default=None, help="Task name or fallback text goal")
    parser.add_argument("--task-description", default=None, help="Natural language task description")
    parser.add_argument("--task-embedding-path", default=None, help="Path to task embedding pkl")
    parser.add_argument("--demo-hdf5", default=None, help="Optional HDF5 file for an offline smoke test")
    parser.add_argument("--demo-key", default="demo_0", help="Demo key under /data")
    parser.add_argument("--step", type=int, default=0, help="Time step inside the demo")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_argparser().parse_args()

    policy = RealRobotPolicy(
        checkpoint_dir=args.ckpt_dir,
        config_path=args.config_path,
        device=args.device,
        task_description=args.task_description,
        task_name=args.task_name,
        task_embedding_path=args.task_embedding_path,
    )
    log.info("Loaded policy from %s", args.ckpt_dir)

    if args.demo_hdf5 is None:
        log.info("Policy is ready. Import RealRobotPolicy from real_robot.infer for online control.")
        return

    obs = _load_demo_observation(args.demo_hdf5, args.demo_key, args.step)
    action = policy.predict_action(
        agentview_image=obs["agentview_image"],
        eye_in_hand_image=obs["eye_in_hand_image"],
        robot_states=obs["robot_states"],
    )
    print(action)


if __name__ == "__main__":
    main()
