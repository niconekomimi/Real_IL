# 真机推理说明

这里仅用于加载训练好的 checkpoint 并输出动作。

## 必需的 checkpoint 文件

- `last_model.pth`
- `model_scaler.pkl`
- `.hydra/config.yaml`

## 环境安装要求

先安装基础依赖：

```bash
pip install -r requirements.txt
```

### 安装 mamba1/2

```bash
pip install \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu11torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

pip install \
  https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu11torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

pip install "setuptools==79.0.1"
```

- 不要直接执行 `pip install mamba-ssm[causal-conv1d]`。如果没有命中预编译 wheel，它会退回源码构建，并在缺少 `nvcc` 时触发上游 `setup.py` 报错。
- 这组 wheel 已在本机验证通过，适用环境是 `Python 3.10`、`torch 2.7.1+cu118`、`cxx11abi=TRUE`、`linux_x86_64`。
- 如果环境版本不同，请去以下 release 页换成匹配标签的 wheel：
  - `https://github.com/Dao-AILab/causal-conv1d/releases`
  - `https://github.com/state-spaces/mamba/releases`

安装后自检：

```bash
python -c "import torch, mamba_ssm, causal_conv1d; from mamba_ssm.modules.mamba2 import Mamba2; print(torch.__version__, torch.cuda.is_available(), mamba_ssm.__version__, getattr(causal_conv1d, '__version__', 'unknown'), Mamba2)"
```

### 安装 xlstm

```bash
git clone https://github.com/NX-AI/xlstm.git
cd xlstm
pip install -e . --ignore-requires-python
```

- `pip install timm`
- `pip install transformers`
- `pip install torchsde torchdiffeq`

## 命令行推理

```bash
python real_robot/infer.py \
  --ckpt-dir /path/to/checkpoint_dir \
  --task-description "pick up the object and place it in the basket"
```

如果你已经有预计算好的任务 embedding：

```bash
python real_robot/infer.py \
  --ckpt-dir /path/to/checkpoint_dir \
  --task-name pick_up_the_object_and_place_it_in_the_basket \
  --task-embedding-path /path/to/task_embeddings.pkl
```

## 离线单步测试

```bash
python real_robot/infer.py \
  --ckpt-dir /path/to/checkpoint_dir \
  --task-description "pick up the object and place it in the basket" \
  --demo-hdf5 /path/to/demo.hdf5 \
  --demo-key demo_0 \
  --step 0
```

## Python 调用方式

```python
from real_robot.infer import RealRobotPolicy

policy = RealRobotPolicy(
    checkpoint_dir="/path/to/checkpoint_dir",
    task_description="pick up the object and place it in the basket",
)

action = policy.predict_action(
    agentview_image=agentview_rgb,
    eye_in_hand_image=eye_in_hand_rgb,
    robot_states=robot_state,
)
```
