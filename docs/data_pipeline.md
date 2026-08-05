# X-VLA 数据处理流程文档

## 1. 概述

本文档描述X-VLA从训练数据到推理输入的完整数据处理流程，包括数据格式定义、数据加载机制、格式转换、归一化处理等关键细节。

---

## 2. 数据格式定义

### 2.1 训练数据输出格式

`InfiniteDataReader` 输出的单个样本格式：

```python
{
    'domain_id': LongTensor[],           # domain标识 (e.g., 0-17)
    'language_instruction': str,         # 文本指令
    'image_input': FloatTensor[V,C,H,W], # V个视角，默认3，H=W=224
    'image_mask': BoolTensor[V],         # 有效视角掩码
    'proprio': FloatTensor[dim_proprio], # 本体感受 (单臂10维，双臂20维)
    'action': FloatTensor[T,dim_action]  # 未来动作序列 T=30, dim=20
}
```

### 2.2 支持的数据集与Domain ID

定义位置：`datasets/domain_config.py`

| Domain ID | 数据集 | 机器人类型 | 用途 |
|-----------|--------|-----------|------|
| 0 | Bridge, widowx-air, agiworld-* | WidowX/Agilex | Finetuning |
| 1 | RT1 | RT1机器人 | Finetuning |
| 2 | Calvin | Franka | Finetuning |
| 3 | libero | Franka | Finetuning |
| 4 | widowx-air | WidowX | Finetuning |
| 5 | AIR-AGILEX-HQ | Agilex | Finetuning |
| 6 | robotwin2_abs_ee | 双臂Agilex | Finetuning |
| 7 | robocasa-human | 软操控 | Finetuning |
| 8 | VLABench | Franka | Finetuning |
| 9 | AGIBOT-challenge | Agilex G1 | Finetuning |
| 10 | AIR-AGILEX | Agilex | Finetuning |
| 11 | robomind-franka | Franka | Pretraining |
| 12 | robomind-ur | UR机器人 | Pretraining |
| 13 | Droid-Left | Droid | Pretraining |
| 14 | Droid-Right | Droid | Pretraining |
| 15 | AGIBOT | Agilex | Pretraining |
| 16 | robomind-agilex | Agilex | Pretraining |
| 17 | robomind-franka-dual | 双臂Franka | Pretraining |

---

## 3. 训练数据处理流程

### 3.1 完整数据流图

```
train.py / peft_train.py
│
├── create_dataloader(metas_path, num_actions, action_mode, training=True)
│   │
│   ▼
├── InfiniteDataReader (IterableDataset)
│   │
│   ├── 读取JSON meta文件 (dataset.py:54-74)
│   │   ├── General Style: {"dataset_name": "...", "datalist": [...]}
│   │   └── LeRobot v2.1 Style: {"codebase_version": "v2.1", "root_path": "..."}
│   │
│   ├── for each dataset:
│   │   └── get_handler_cls(robot_type) → Handler类
│   │
│   ├── Handler.iter_episode() 数据加载
│   │   ├── 读取H5/parquet/video原始数据
│   │   ├── 频率插值到统一时间轴 (scipy.interpolate.interp1d)
│   │   ├── 旋转格式转换 (euler/quat → 6D rotation)
│   │   ├── 生成action chunk (当前帧 + 未来30个动作点)
│   │   └── 图像增强 (Resize + ColorJitter + Normalize)
│   │
│   ├── action_slice() 分割proprio/action (utils.py:90-108)
│   │   ├── proprio = abs_trajectory[0]      # 当前帧
│   │   └── action = abs_trajectory[1:]      # 未来30帧
│   │
│   ├── 加权采样 (training=True时) (dataset.py:118-122)
│   │   └── random.choices(names, weights=DATA_WEIGHTS)
│   │
│   └── yield sample
│
├── DataLoader batch化 (batch_size=16, num_workers=4)
│
├── processor.encode_language() Tokenize文本
│
└── model.forward() → loss计算
```

### 3.2 Meta文件格式

**General Style**：
```json
{
    "dataset_name": "Calvin",
    "robot_type": "Calvin",
    "datalist": ["/path/to/traj1.h5", "/path/to/traj2.h5", ...],
    "observation_key": ["image_front", "image_wrist"],
    "language_instruction_key": "instruction",
    "lang_aug_map": {"原始指令": ["变体1", "变体2"]}  // 可选
}
```

**LeRobot v2.1 Style**：
```json
{
    "codebase_version": "v2.1",
    "robot_type": "agibot",
    "root_path": "/path/to/dataset",
    "total_episodes": 1000
}
```
（配合同目录的 `episodes.jsonl` 文件使用）

### 3.3 图像处理

位置：`datasets/dataset.py:76-83`

```python
image_aug = [
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.),  # 仅训练时
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet归一化
]
```

| 处理步骤 | 说明 |
|---------|------|
| Resize | 双三次插值到224×224 |
| ColorJitter | 训练时启用，增强泛化能力 |
| Normalize | ImageNet mean/std |

### 3.4 频率对齐与插值

位置：`datasets/domain_handler/base.py:142-154`

不同机器人数据频率不同，通过时间插值对齐：

| 数据集 | 原始频率 | 未来窗口(qdur) |
|-------|---------|--------------|
| Calvin | 30Hz | 1.0s |
| RT1 | 3Hz | 10.0s |
| Bridge | 5Hz | 5.0s |
| LIBERO | 30Hz | 1.0s |
| WidowX | 5Hz | 5.0s |

```python
# 构建时间参考和插值器
lt = np.arange(left.shape[0], dtype=np.float64) / float(freq)
L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))

# 查询未来窗口: 当前时间 → 当前+qdur，生成num_actions+1个点
q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1)
lseq = torch.tensor(L(q))  # 插值到统一时间点
```

### 3.5 旋转格式转换

位置：`datasets/utils.py:54-58`

统一转换为 **6D Rotation** 表示（Zhou et al.）：

```python
def quat_to_rotate6d(q: np.ndarray) -> np.ndarray:
    """Quaternion → 6D (wxyz格式)"""
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    """Euler → 6D"""
    return R.from_euler(pattern, q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))
```

各数据集原始旋转格式：

| 数据集 | 原始格式 | 转换函数 |
|-------|---------|---------|
| Calvin | euler_xyz | `euler_to_rotate6d(proprio[:, 3:6], "xyz")` |
| RT1 | quat (wxyz) | `quat_to_rotate6d(eefq[:, 3:])` |
| Bridge | euler_xyz | `euler_to_rotate6d(proprio[:, 3:6], "xyz")` |
| LIBERO | 已有6D | 直接使用 |
| RobotWin2 | quat | `quat_to_rotate6d(l[:, 3:])` |
| AGIBOT | quat | `quat_to_rotate6d(orientation)` |

### 3.6 EE6D动作格式

位置：`models/action_hub.py:119-122`

```
EE6D 20维 = [左臂(10维)] + [右臂(10维)]

索引分配：
[0,1,2]         - 左臂位置 xyz           (3)
[3,4,5,6,7,8]   - 左臂旋转6D             (6)
[9]             - 左臂gripper (0=open, 1=close) (1)
[10,11,12]      - 右臂位置 xyz           (3)
[13-18]         - 右臂旋转6D             (6)
[19]            - 右臂gripper            (1)
```

**Proprio与Action格式相同**，都使用EE6D表示。

### 3.7 Gripper离散化

Gripper状态已经是二值的（0=open, 1=close），代码中直接转换：

```python
# Bridge: 1=open → 1=close
1 - action[:, -1:]

# CALVIN: <0 → True (closed)
proprio[:, -1:] < 0.

# LIBERO: >0 → True (closed)
(a[:, 9:] > 0.0)
```

Loss计算使用 **BCEWithLogitsLoss**。

---

## 4. 数据权重与采样

### 4.1 DATA_WEIGHTS定义

位置：`datasets/domain_config.py:19-39`

```python
DATA_WEIGHTS = {
    # Pretraining
    "AGIBOT": 0.4,
    "Droid-Left": 0.15,
    "Droid-Right": 0.15,
    "robomind-franka": 0.1,
    "robomind-ur": 0.1,
    "robomind-agilex": 0.07,
    "robomind-franka-dual": 0.03,

    # AgiWorld Challenge (finetuning)
    "agiworld-on-site-pack": 0.8,
    "agiworld-on-site-conveyor": 0.8,
    "agiworld-on-site-restock": 1.0,
    "agiworld-on-site-pour": 1.0,
    "agiworld-on-site-microwave": 1.2,
    "agiworld-on-site-cloth": 1.2,
    "agiworld-on-site-cloth-2": 0.1,
}
```

### 4.2 加权采样实现

位置：`datasets/dataset.py:118-122`

```python
def __iter__(self):
    names = list(self.metas.keys())
    if not self.training:
        # 验证/推理：顺序遍历
        for n in names: yield from self._iter_one_dataset(n)
    else:
        # 训练：加权随机采样
        gens = [iter(self._iter_one_dataset(n)) for n in names]
        ws = [DATA_WEIGHTS.get(n, 1.0) for n in names]
        ws = [w / sum(ws) for w in ws]  # 归一化
        while True:
            i = random.choices(range(len(names)), weights=ws, k=1)[0]
            yield next(gens[i])
```

**注意**：权重影响的是"从哪个数据集采样"的概率，而非数据内部的采样点。

---

## 5. Finetuning数据处理

### 5.1 与Pretraining的差异

**数据处理代码完全相同**，区别仅在于：

| 方面 | Pretraining | Finetuning |
|-----|-------------|------------|
| 数据来源 | 多域大规模数据混合 | 特定任务/机器人数据 |
| Domain ID | 11-17 | 0-10 |
| 采样权重 | DATA_WEIGHTS | 通常uniform |
| 训练目标 | 学习通用VLA表示 | 适应特定任务 |

### 5.2 Finetuning模型与数据集对应

| 模型 | 评估场景 | Finetuning数据 |
|-----|---------|---------------|
| X-VLA-Calvin-ABC_D | CALVIN ABC→D | Calvin数据集 |
| X-VLA-Libero | LIBERO | LIBERO benchmark |
| X-VLA-VLABench | VLABench | VLABench数据集 |
| X-VLA-Google-Robot | Google Robot | Google Robot大数据集 |
| X-VLA-WidowX | WidowX | BridgeDataV2 |
| X-VLA-RoboTwin2 | RoboTwin2.0 | RoboTwin2数据集 |
| X-VLA-SoftFold | 布料折叠 | SoftFold数据集 |
| X-VLA-AgiWorld-Challenge | IROS 2025 | agiworld-on-site-* |

---

## 6. 推理/评估数据处理

### 6.1 Server-API输入格式

位置：`models/modeling_xvla.py:232-280`

**POST http://{host}:{port}/act**

```json
{
    "language_instruction": "Pick up the object",      // 必需: 任务指令
    "proprio": "json_numpy.dumps(array)",              // 必需: 本体感受 (10/20维)
    "image0": "json_numpy.dumps(image)",               // 必需: 主视角图像 (H,W,3)
    "image1": "json_numpy.dumps(image)",               // 可选: 第二视角
    "image2": "json_numpy.dumps(image)",               // 可选: 第三视角
    "domain_id": 2,                                    // 必需: domain标识
    "steps": 10                                        // 可选: 去噪步数，默认10
}
```

### 6.2 客户端数据格式化

评估客户端负责将各机器人的原始obs转换为统一格式：

```python
# calvin_client.py - 转换为EE6D格式
proprio = np.concatenate([
    obs["robot_obs"][:3],                    # xyz位置 (3)
    euler_xyz_to_rotate6D(obs["robot_obs"][3:6]),  # euler → 6D (6)
    obs['robot_obs'][-1:] > 0.               # gripper二值化 (1)
])
# 单臂补零到20维
proprio = np.concatenate([proprio, np.zeros_like(proprio)])
```

Proprio维度要求：

| 机器人类型 | Dimension | 说明 |
|-----------|-----------|------|
| 单臂 (Franka/WidowX/UR/Agilex等) | 10 | 左臂 + 右臂补零 |
| 双臂 (RobotWin2) | 20 | 左臂(10) + 右臂(10) |

### 6.3 推理输出格式

```json
{
    "action": [[pos3, rot6d_1, grip1, pos3, rot6d_2, grip2], ...]
}
```

- Shape: `(30, 20)` - 30个时间步，每步20维EE6D
- 位置范围：取决于训练数据的归一化范围
- Gripper：已经是sigmoid后的概率值 (0-1)

---

## 7. 归一化处理总结

| 数据类型 | 归一化 | 方法 |
|---------|-------|------|
| image_input | ✅ 是 | ImageNet mean/std |
| proprio | ❌ 否 | 原始数据直接使用 |
| action | ❌ 否 | 原始数据直接使用，但loss通过权重平衡量纲 |

### Loss中的隐式归一化

位置：`models/action_hub.py:115-117`

```python
# EE6DActionSpace loss权重
XYZ_SCALE = 500.0   # 位置：放大500倍
ROT_SCALE = 10.0    # 旋转：放大10倍
GRIPPER_SCALE = 1.0 # 夹爪：不变
```

这实际上是一种**量纲平衡策略**，而非真正的归一化。

---

## 8. 关键代码位置索引

| 功能 | 文件位置 | 行号 |
|-----|---------|------|
| Domain ID定义 | datasets/domain_config.py | 41-83 |
| DataWeights定义 | datasets/domain_config.py | 19-39 |
| 创建dataloader | datasets/__init__.py | 27-40 |
| InfiniteDataReader | datasets/dataset.py | 28-122 |
| action_slice | datasets/utils.py | 90-108 |
| 旋转格式转换 | datasets/utils.py | 54-87 |
| BaseHDF5Handler | datasets/domain_handler/base.py | 68-172 |
| 各数据集Handler | datasets/domain_handler/*.py | - |
| ActionSpace定义 | models/action_hub.py | 51-362 |
| EE6D格式定义 | models/action_hub.py | 119-122 |
| Server API | models/modeling_xvla.py | 232-284 |
| Client评估示例 | evaluation/calvin/calvin_client.py | 69-109 |

---

## 9. 扩展新机器人数据集

要支持新的机器人（如arx_x5），需要：

1. **在domain_config.py添加domain_id**：
```python
DATA_DOMAIN_ID = {
    "arx_x5": 20,  # 新增
    ...
}
```

2. **在registry.py注册handler**：
```python
from .my_dataset import ArxX5Handler
_REGISTRY = {
    ...
    "arx_x5": ArxX5Handler,
}
```

3. **创建新的Handler类**，实现：
```python
class ArxX5Handler(DomainHandler):
    def build_left_right(self, f) -> (left, right, lt, rt, freq, qdur):
        # 返回格式: [T, 10] 的EE6D动作
        pass

    def index_candidates(self, T_left, training) -> Iterable[int]:
        pass
```

4. **创建meta.json**：
```json
{
    "dataset_name": "arx_x5",
    "robot_type": "arx_x5",
    "datalist": ["/path/to/traj1.h5", ...]
}
```

---

## 10. 注意事项

1. **Proprio维度必须匹配**：单臂10维，双臂20维，否则模型会报错
2. **图像格式**：需要是RGB uint8 (H,W,3)，json_numpy序列化
3. **domain_id必须正确**：错误的domain_id会导致使用错误的soft_prompt
4. **推理时steps参数**：影响去噪精度，默认10步足够
5. **Action输出需要后处理**：6D旋转需要转换为quat/euler才能控制真实机器人
