# X-VLA 模型架构文档

## 1. 模块结构总览

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                      XVLA                                           │
│                              (models/modeling_xvla.py)                               │
│                                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │  1. Florence2Encoder                                                          │  │
│  │     位置: models/modeling_florence2.py                                        │  │
│  │     功能: 图像+文本联合编码 → vlm_features + aux_visual_inputs                │  │
│  ├───────────────────────────────────────────────────────────────────────────────┤  │
│  │  2. SoftPromptedTransformer                                                   │  │
│  │     位置: models/transformer.py                                               │  │
│  │     ├── action_encoder (DomainAwareLinear)              [Domain相关]          │  │
│  │     ├── vlm_proj / aux_visual_proj (Linear)                                  │  │
│  │     ├── pos_emb (可学习位置编码)                                              │  │
│  │     ├── soft_prompt_hub (Embedding)                    [Domain相关]          │  │
│  │     ├── TransformerBlocks × 24                          [冻结]               │  │
│  │     └── action_decoder (DomainAwareLinear)              [Domain相关]          │  │
│  ├───────────────────────────────────────────────────────────────────────────────┤  │
│  │  3. ActionSpace                                                               │  │
│  │     位置: models/action_hub.py                                                │  │
│  │     功能: 动作格式定义、loss计算、后处理                                       │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心模块详解

### 2.1 Florence2Encoder

**位置**: `models/modeling_florence2.py`

```
                    ┌─────────────────────────────────────┐
                    │      Florence2Encoder               │
                    │                                     │
image_input ───────▶│  DaVIT                              │
[B,V,C,H,W]         │  Visual Encoder                     │
                    └────────────────┬────────────────────┘
                                     │
                                     ▼
                            image_features
                            [B, V, N=256, D=1024]
                                     │
                    ┌────────────────┴────────────────────┐
                    │                                     │
input_ids ─────────▶│  Text Embeddings                   │
[B,L]               │                                     │
                    └────────────────┬────────────────────┘
                                     │
                                     ▼
                            inputs_embeds
                            [B, L, D=1024]
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │  _merge_input_ids_with_image_features │
                    │  融合文本 + 主视角图像                │
                    └────────────────┬────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │  BART Encoder                       │
                    │  (models/modeling_florence2.py:...) │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────┴────────────────────────────────────┐
                    ▼                                                         ▼
          vlm_features                                        aux_visual_inputs
          [B, T_vlm, 1024]                                    [B, T_aux, 1024]
          (主视角+文本融合)                                     (辅视角扁平化)
```

---

### 2.2 SoftPromptedTransformer

**位置**: `models/transformer.py:286-403`

```
                    ┌─────────────────────────────────────────────────────┐
                    │           SoftPromptedTransformer                  │
                    │                                                     │
                    │  ┌───────────────────────────────────────────────┐  │
                    │  │ Step 1: 构建Action Tokens                     │  │
                    │  │                                               │  │
 action_with_noise ─┼─▶│  concat([action, proprio, time_tokens])     │  │
 [B,T,20]          │  │  [B,T,20] + [B,T,20] + [B,T,32] = [B,T,72]   │  │
                    │  └───────────────────────┬───────────────────────┘  │
                    │                          │                          │
                    │                          ▼                          │
                    │  ┌───────────────────────────────────────────────┐  │
                    │  │ Step 2: action_encoder (DomainAwareLinear)    │  │
                    │  │  DomainAwareLinear(72 → 1024, num_domains=30) │  │
                    │  │                                               │  │
                    │  │  输入: [B, T, 72]                            │  │
                    │  │  输出: [B, T, 1024]                          │  │
                    │  └───────────────────────┬───────────────────────┘  │
                    │                          │                          │
                    │  ┌───────────────────────┴───────────────────────┐  │
                    │  │ Step 3: 视觉特征投影                          │  │
                    │  │                                               │  │
vlm_features ───────┼─▶│  vlm_proj: Linear(1024 → 1024)               │  │
 [B,T_vlm,1024]     │  │  aux_visual_proj: Linear(1024 → 1024)        │  │
                    │  └───────────────────────┬───────────────────────┘  │
                    │                          │                          │
                    │                          ▼                          │
                    │  ┌───────────────────────────────────────────────┐  │
                    │  │ Step 4: concat所有token                       │  │
                    │  │                                               │  │
                    │  │  [action_emb, vlm_proj, aux_proj]            │  │
                    │  │  [B, T + T_vlm + T_aux, 1024]                │  │
                    │  └───────────────────────┬───────────────────────┘  │
                    │                          │                          │
                    │                          ▼                          │
                    │  ┌───────────────────────────────────────────────┐  │
                    │  │ Step 5: + pos_emb + soft_prompts              │  │
                    │  │                                               │  │
                    │  │  pos_emb: [1, 512, 1024] (可学习)             │  │
domain_id ──────────┼─▶│  soft_prompts: soft_prompt_hub(domain_id)     │  │
 [B]                │  │  → [B, 32, 1024] (每domain独立)               │  │
                    │  └───────────────────────┬───────────────────────┘  │
                    │                          │                          │
                    │                          ▼                          │
                    │  ┌───────────────────────────────────────────────┐  │
                    │  │ Step 6: TransformerBlocks × 24               │  │
                    │  │  (标准Pre-LN Transformer, 无causal mask)     │  │
                    │  │                                               │  │
                    │  │  for block in self.blocks:                   │  │
                    │  │      x = block(x)                            │  │
                    │  └───────────────────────┬───────────────────────┘  │
                    │                          │                          │
                    │                          ▼                          │
                    │  ┌───────────────────────────────────────────────┐  │
                    │  │ Step 7: action_decoder (DomainAwareLinear)    │  │
                    │  │  DomainAwareLinear(1024 → 20, num_domains=30) │  │
                    │  │                                               │  │
                    │  │  输入: [B, T, 1024]                          │  │
                    │  │  输出: [B, T, 20]                            │  │
                    │  └───────────────────────────────────────────────┘  │
                    │                                                     │
                    └─────────────────────────────────────────────────────┘
                                     │
                                     ▼
                              pred_action
                              [B, T, 20]
```

---

### 2.3 DomainAwareLinear

**位置**: `models/transformer.py:210-250`

```
          ┌──────────────────────────────────────────────────────┐
          │              DomainAwareLinear                       │
          │                                                      │
          │  输入: x [B, ..., input_size]                        │
          │       domain_id [B]                                 │
          │                                                      │
          │  W = fc(domain_id)  → [B, output_size, input_size]  │
          │  b = bias(domain_id) → [B, output_size]             │
          │                                                      │
          │  y = x @ W^T + b                                    │
          │                                                      │
          │  输出: y [B, ..., output_size]                      │
          └──────────────────────────────────────────────────────┘

  关键: 不同domain_id使用不同的W和b权重
        即便输入x相同，不同domain输出也不同
```

---

### 2.4 ActionSpace (EE6DActionSpace)

**位置**: `models/action_hub.py:109-168`

```
                    ┌─────────────────────────────────────────────┐
                    │           EE6DActionSpace                   │
                    │                                             │
                    │  dim_action = 20                            │
                    │  gripper_idx = (9, 19)                      │
                    │                                             │
                    │  20维结构:                                   │
                    │  [0:3]   左臂位置 xyz                       │
                    │  [3:9]   左臂旋转6D                         │
                    │  [9]     左臂gripper (0=open, 1=close)     │
                    │  [10:13] 右臂位置 xyz                       │
                    │  [13:19] 右臂旋转6D                         │
                    │  [19]    右臂gripper                        │
                    │                                             │
                    ├─────────────────────────────────────────────┤
                    │  preprocess(proprio, action)                │
                    │  → gripper通道置零                          │
                    ├─────────────────────────────────────────────┤
                    │  compute_loss(pred, target)                 │
                    │  → position_loss * 500                      │
                    │  → rotate6D_loss * 10                       │
                    │  → gripper_loss * 1.0 (BCE)                 │
                    ├─────────────────────────────────────────────┤
                    │  postprocess(action)                        │
                    │  → sigmoid(gripper)                         │
                    └─────────────────────────────────────────────┘
```

---

## 3. 完整数据流

### 3.1 训练数据流

```
                    ┌──────────────────────────────────────────────────────────────┐
                    │                      训练数据流                                │
                    │                                                              │
language ──────────▶│                                                              │
"PICK UP THE CUP"   │                                                              │
                    │                                                              │
                    ▼                                                              │
          ┌─────────────────────────────────────────────────────────────────┐       │
          │ XVLAProcessor                                                       │       │
          │                                                                     │       │
          │  encode_language() → input_ids [B, L]                              │       │
          │  encode_image() → image_input [B, V, C, H, W]                      │       │
          └─────────────────────────────────────────────────────────────────┘       │
                    │                                                              │
                    ▼                                                              │
          ┌─────────────────────────────────────────────────────────────────┐       │
          │ Florence2Encoder                                                    │       │
          │                                                                     │       │
          │  image_input → DaVIT → image_features [B,V,N,D]                   │       │
          │  input_ids → TextEmbed → inputs_embeds [B,L,D]                    │       │
          │  merge + BART Encoder → vlm_features [B,T_vlm,D]                  │       │
          │                      → aux_visual_inputs [B,T_aux,D]               │       │
          │                                                                     │       │
          │  位置: modeling_xvla.py:104-145                                    │       │
          └─────────────────────────────────────────────────────────────────┘       │
                    │                                                              │
                    ▼                                                              │
          ┌─────────────────────────────────────────────────────────────────┐       │
          │ Flow Matching (modeling_xvla.py:165-168)                          │       │
          │                                                                     │       │
          │  t = random()                                                      │       │
          │  action_noisy = t * noise + (1-t) * action                        │       │
          │                                                                     │       │
          └─────────────────────────────────────────────────────────────────┘       │
                    │                                                              │
                    ▼                                                              │
          ┌─────────────────────────────────────────────────────────────────┐       │
          │ SoftPromptedTransformer                                           │       │
          │                                                                     │       │
          │  concat(action, proprio, time) → action_encoder                  │       │
          │  vlm_proj(vlm_features) + aux_visual_proj(aux_visual_inputs)     │       │
          │  concat → +pos_emb → +soft_prompts                               │       │
          │  TransformerBlocks × 24                                           │       │
          │  action_decoder → pred_action [B,T,20]                            │       │
          │                                                                     │       │
          │  位置: transformer.py:341-403                                     │       │
          └─────────────────────────────────────────────────────────────────┘       │
                    │                                                              │
                    ▼                                                              │
          ┌─────────────────────────────────────────────────────────────────┐       │
          │ ActionSpace.compute_loss()                                        │       │
          │                                                                     │       │
          │  position_loss = MSE(pred[:,:,:3], target[:,:,:3]) * 500         │       │
          │  rotate6D_loss = MSE(pred[:,:,3:9], target[:,:,3:9]) * 10        │       │
          │  gripper_loss = BCE(pred[:,:,9], target[:,:,9])                   │       │
          │                                                                     │       │
          │  位置: action_hub.py:129-154                                      │       │
          └─────────────────────────────────────────────────────────────────┘       │
                    │                                                              │
                    ▼                                                              │
                 loss                                                              │
                 backward                                                          │
                    │                                                              │
                    ▼                                                              │
          ┌─────────────────────────────────────────────────────────────────┐       │
          │                    参数更新策略                                    │       │
          │                                                                     │       │
          │  step < freeze_steps:                                              │       │
          │    VLM:              lr=0      (冻结)                              │       │
          │    TransformerBlocks: lr=0    (冻结)                               │       │
          │    soft_prompts:     lr=base  (可学习)                             │       │
          │    action_heads:     lr=base  (可学习)                             │       │
          │                                                                     │       │
          │  step >= freeze_steps:                                             │       │
          │    全部解开训练                                                     │       │
          │                                                                     │       │
          │  位置: train.py:154-172                                            │       │
          └─────────────────────────────────────────────────────────────────┘       │
                    │                                                              │
                    └──────────────────────────────────────────────────────────────┘
```

### 3.2 推理数据流

```
                    ┌──────────────────────────────────────────────────────────────┐
                    │                      推理数据流                                │
                    │                                                              │
language ──────────▶│                                                              │
                    │                                                              │
                    ▼                                                              │
          ┌─────────────────────────────────────────────────────────────────┐       │
          │ XVLAProcessor                                                       │       │
          │  encode_language() + encode_image()                               │       │
          └─────────────────────────────────────────────────────────────────┘       │
                    │                                                              │
                    ▼                                                              │
          ┌─────────────────────────────────────────────────────────────────┐       │
          │ Florence2Encoder → vlm_features + aux_visual_inputs              │       │
          │  位置: modeling_xvla.py:104-145                                  │       │
          └─────────────────────────────────────────────────────────────────┘       │
                    │                                                              │
                    ▼                                                              │
          ┌─────────────────────────────────────────────────────────────────┐       │
          │ Flow Matching 去噪 (modeling_xvla.py:181-216)                     │       │
          │                                                                     │       │
          │  x1 = randn(B, T, D)     # 纯噪声                                  │       │
          │  action = zeros(B, T, D)  # 初始x_0                                │       │
          │                                                                     │       │
          │  for i in [steps, steps-1, ..., 1]:                               │       │
          │      t = i / steps                                                │       │
          │      x_t = t * x1 + (1-t) * action                                │       │
          │      action = transformer(domain_id, x_t, t, proprio, ...)        │       │
          │                                                                     │       │
          └─────────────────────────────────────────────────────────────────┘       │
                    │                                                              │
                    ▼                                                              │
          ┌─────────────────────────────────────────────────────────────────┐       │
          │ ActionSpace.postprocess()                                         │       │
          │  sigmoid(gripper)                                                 │       │
          │  位置: action_hub.py:164-168                                      │       │
          └─────────────────────────────────────────────────────────────────┘       │
                    │                                                              │
                    ▼                                                              │
                 action                                                           │
                 [B, T=30, D=20]                                                   │
                    │                                                              │
                    ▼                                                              │
          ┌─────────────────────────────────────────────────────────────────┐       │
          │ 6D → Quaternion 转换 (client.py:72-84)                           │       │
          │  rotate6D_to_quat()                                               │       │
          └─────────────────────────────────────────────────────────────────┘       │
                    │                                                              │
                    ▼                                                              │
               robot控制命令                                                       │
               [B, 16] = [xyz, quat, grip] × 2 arms
```

---

## 4. Pretraining vs Finetuning

### 4.1 参数分组

```
model.parameters() 全部参数
    │
    ├── VLM (Florence2)
    │       位置: modeling_florence2.py
    │       包含: DaVIT, TextEmbeddings, BART Encoder
    │
    ├── Transformer Core (冻结)
    │       位置: transformer.py
    │       包含: pos_emb, norm, blocks × 24, vlm_proj, aux_visual_proj
    │
    ├── Domain-Aware (可学习)
    │       位置: transformer.py
    │       包含: soft_prompt_hub, action_encoder, action_decoder
    │
    └── ActionSpace (结构定义)
            位置: action_hub.py
```

### 4.2 学习率调度

```
学习率
  │
  │  soft_prompts / action_heads (始终可学习)
  │  ┌────────────────────────────────────────
  │  │                                    ╱
  │  │                                 ╱
  │  │                              ╱
  │  │                           ╱
  │  │________________________╱_______________ step
  │  │           freeze_steps    freeze_steps+warmup_steps
  │  │              (1000)           (3000)
  │  │
  │  │  vlm / transformer_core (freeze_steps后才开始)
  │  │  ___________________________________
  │  │                                 |  ╲
  │  │                                 |   ╲
  │  │_________________________________|____╲________
  │  │                                1000  3000
  │  │
  └─────────────────────────────────────────────────────
```

### 4.3 对比表

| 参数组 | 冻结/可学习 | Pretraining | Finetune 第一阶段 | Finetune 第二阶段 |
|--------|------------|-------------|-------------------|-------------------|
| **VLM** | 冻结 | ✅ 更新 | ❌ 冻结 | ✅ 更新 |
| **Transformer Blocks** | 冻结 | ✅ 更新 | ❌ 冻结 | ✅ 更新 |
| **vlm_proj/aux_visual_proj** | 冻结 | ✅ 更新 | ❌ 冻结 | ✅ 更新 |
| **soft_prompt_hub** | 可学习 | ✅ 更新 | ✅ 更新 | ✅ 更新 |
| **action_encoder** | 可学习 | ✅ 更新 | ✅ 更新 | ✅ 更新 |
| **action_decoder** | 可学习 | ✅ 更新 | ✅ 更新 | ✅ 更新 |

**默认参数**:
- `freeze_steps = 1000`
- `warmup_steps = 2000`
- `learning_rate = 1e-4`
- `learning_coef = 1.0`

---

## 5. RoboTwin-2.0 Finetuning 完整流程

### 5.1 Domain 相关配置

```
┌─────────────────────────────────────────────────────────────────┐
│                    Domain 配置                                   │
│                                                                 │
│  DATA_DOMAIN_ID (datasets/domain_config.py:49-50)               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  "robotwin2_abs_ee": 6,                                   │  │
│  │  "robotwin2_clean": 6,   ← 同一Handler                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Handler注册 (datasets/domain_handler/registry.py:58-59)        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  "robotwin2_abs_ee": RobotWin2Handler,                    │  │
│  │  "robotwin2_clean": RobotWin2Handler,                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 RobotWin2Handler 数据格式

```
┌─────────────────────────────────────────────────────────────────┐
│              RobotWin2Handler (双臂AgileX)                       │
│              位置: domain_handler/simulations.py:150-174         │
│                                                                 │
│  输入HDF5:                                                       │
│    /endpose/left_endpose   [T,7]  xyz(3)+quat(4)               │
│    /endpose/right_endpose  [T,7]                                │
│    /endpose/left_gripper   [T]                                  │
│    /endpose/right_gripper  [T]                                  │
│                                                                 │
│  处理流程:                                                       │
│    left_grip = 1 - left_gripper   # 1=open→1=closed            │
│    right_grip = 1 - right_gripper                                │
│    left  = concat([xyz, quat→6D, left_grip])   → [T,10]        │
│    right = concat([xyz, quat→6D, right_grip])  → [T,10]        │
│                                                                 │
│  输出:                                                           │
│    left [T,10] + right [T,10] = [T,20] (EE6D格式)               │
│    freq=30Hz, qdur=1s → 30个动作步                               │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Meta 文件格式

```json
// robotwin2_meta.json
{
    "dataset_name": "robotwin2_abs_ee",
    "robot_type": "robotwin2_abs_ee",
    "datalist": ["/path/to/traj1.h5", "/path/to/traj2.h5", ...],
    "observation_key": ["image_left", "image_right", "image_front"],
    "language_instruction_key": "instruction"
}
```

### 5.4 训练流程

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           RoboTwin-2.0 训练流程                                      │
│                                                                                      │
│  1. 准备阶段                                                                          │
│     ├── meta.json (包含datalist、observation_key等)                                  │
│     ├── H5数据文件 (包含endpose/left_endpose等)                                      │
│     └── 训练命令:                                                                    │
│         python train.py \                                                            │
│             --models /path/to/pretrained-xvla \                                      │
│             --train_metas_path /path/to/robotwin2_meta.json \                        │
│             --output_dir /path/to/finetuned-robotwin2 \                              │
│             --batch_size 16 \                                                        │
│             --learning_rate 1e-4 \                                                   │
│             --freeze_steps 1000 \                                                    │
│             --warmup_steps 2000                                                      │
│                                                                                      │
│  2. 数据加载                                                                          │
│     meta.json → InfiniteDataReader → RobotWin2Handler                               │
│                    │                                                                 │
│                    ├── build_left_right(): left[T,10] + right[T,10]                 │
│                    ├── 频率插值: 30Hz → 1s窗口                                       │
│                    ├── 生成action chunk: 30步                                        │
│                    └── action_slice(): proprio[20] + action[30,20]                  │
│                                                                                      │
│  3. 训练循环                                                                          │
│     for batch in dataloader:                                                        │
│         lang = processor.encode_language()                                           │
│         inputs = {**batch, **lang}                                                  │
│                                                                                      │
│         # forward                                                                   │
│         enc = model.forward_vlm(inputs)                                             │
│         action_noisy = t * noise + (1-t) * action  (Flow Matching)                  │
│         pred = transformer(domain_id=6, action_noisy, t, proprio, **enc)            │
│         loss = action_space.compute_loss(pred, action)                              │
│                                                                                      │
│         # backward                                                                  │
│         step < freeze_steps:                                                        │
│             只更新: soft_prompt_hub(6), action_encoder(6), action_decoder(6)        │
│             冻结:   VLM, Transformer Blocks                                         │
│         step >= freeze_steps:                                                       │
│             全部解开训练                                                             │
│                                                                                      │
│  4. 保存模型                                                                         │
│     model.save_pretrained(save_dir)                                                 │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.5 推理/评估流程

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           RoboTwin-2.0 推理流程                                      │
│                                                                                      │
│  1. 启动Server                                                                       │
│     python -m models.modeling_xvla --model_path /path/to/finetuned-robotwin2        │
│                                                                                      │
│  2. Client准备obs                                                                   │
│     ┌─────────────────────────────────────────────────────────────────────────┐      │
│     │  ClientModel.step(obs):                                                 │      │
│     │                                                                           │      │
│     │  # 提取并转换EE6D格式                                                     │      │
│     │  left_ee  = obs["endpose"]["left_endpose"]   # [T,7] xyz+quat           │      │
│     │  right_ee = obs["endpose"]["right_endpose"]                             │      │
│     │  abs_eef = concat([                                                      │      │
│     │      left_ee[:, :3], quat_to_rotate6D(left_ee[:, 3:]), left_grip,       │      │
│     │      right_ee[:, :3], quat_to_rotate6D(right_ee[:, 3:]), right_grip     │      │
│     │  ])  # [T, 20]                                                          │      │
│     │                                                                           │      │
│     │  # 发送HTTP请求                                                           │      │
│     │  query = {                                                               │      │
│     │      "domain_id": 6,                                                     │      │
│     │      "proprio": json_numpy.dumps(abs_eef),                               │      │
│     │      "language_instruction": instruction,                                │      │
│     │      "image0": head_view,                                                │      │
│     │      "image1": left_view,                                                │      │
│     │      "image2": right_view                                                │      │
│     │  }                                                                       │      │
│     │  response = requests.post(url, json=query)                               │      │
│     │  return response.json()["action"]  # [30, 20]                           │      │
│     └─────────────────────────────────────────────────────────────────────────┘      │
│                                                                                      │
│  3. Server处理请求 (FastAPI /act)                                                   │
│     modeling_xvla.py:232-284                                                        │
│     ┌─────────────────────────────────────────────────────────────────────────┐      │
│     │  act(payload):                                                           │      │
│     │      proprio = json_numpy.loads(payload["proprio"])  # [20]             │      │
│     │      domain_id = payload["domain_id"]  # 6                              │      │
│     │                                                                           │      │
│     │      # VLM编码                                                            │      │
│     │      inputs = processor(images, text)                                    │      │
│     │      enc = model.forward_vlm(**inputs)                                   │      │
│     │                                                                           │      │
│     │      # Flow Matching去噪 (10步)                                          │      │
│     │      x1 = randn(B, 30, 20)                                               │      │
│     │      action = zeros(B, 30, 20)                                           │      │
│     │      for i in range(steps, 0, -1):                                       │      │
│     │          t = i / steps                                                   │      │
│     │          x_t = t * x1 + (1-t) * action                                   │      │
│     │          action = transformer(domain_id=6, x_t, t, proprio, **enc)       │      │
│     │                                                                           │      │
│     │      return action_space.postprocess(action)                             │      │
│     └─────────────────────────────────────────────────────────────────────────┘      │
│                                                                                      │
│  4. Client后处理动作                                                                 │
│     ┌─────────────────────────────────────────────────────────────────────────┐      │
│     │  # 6D → Quaternion                                                       │      │
│     │  left_quat  = rotate6D_to_quat(action[:, 3:9])                          │      │
│     │  right_quat = rotate6D_to_quat(action[:, 13:19])                        │      │
│     │                                                                           │      │
│     │  # gripper处理: >0.7 → close, 否则open                                  │      │
│     │  left_grip  = 1 - 2 * (action[:, 9:10] > 0.7)                           │      │
│     │  right_grip = 1 - 2 * (action[:, 19:20] > 0.7)                          │      │
│     │                                                                           │      │
│     │  # 最终控制命令: [xyz, quat, grip] × 2 = 16维                            │      │
│     │  rollout_action = concat([left_xyz, left_quat, left_grip,               │      │
│     │                          right_xyz, right_quat, right_grip])             │      │
│     └─────────────────────────────────────────────────────────────────────────┘      │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 关键代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| XVLA主类定义 | models/modeling_xvla.py | 39-100 |
| forward_vlm | models/modeling_xvla.py | 104-145 |
| forward (训练) | models/modeling_xvla.py | 148-178 |
| generate_actions (推理) | models/modeling_xvla.py | 181-216 |
| FastAPI /act | models/modeling_xvla.py | 232-284 |
| SoftPromptedTransformer | models/transformer.py | 286-403 |
| DomainAwareLinear | models/transformer.py | 210-250 |
| TransformerBlock | models/transformer.py | 253-281 |
| timestep_embedding | models/transformer.py | 177-205 |
| EE6DActionSpace | models/action_hub.py | 109-168 |
| build_optimizer | train.py | 116-129 |
| update_group_lrs | train.py | 154-172 |
| InfiniteDataReader | datasets/dataset.py | 28-122 |
| RobotWin2Handler | datasets/domain_handler/simulations.py | 150-174 |
| DATA_DOMAIN_ID | datasets/domain_config.py | 41-83 |
| create_dataloader | datasets/__init__.py | 27-40 |
