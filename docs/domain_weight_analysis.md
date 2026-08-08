# X-VLA per-domain 权重分析文档

**数据来源**（~Downloads 下的 4 个分析文件 + 仓库代码）：
- 预训练基线：`soft_prompt_hub_per_dim_stats.json`、`action_weights_per_dim_stats.json`（X-VLA-Pt）
- 官方：`RoboDojo-sim-arx_x5-ee-0-ckpt-100000_per_dim_stats.json`（domain=0，100k 步）
- 自训练：`xvla-ee6d-020000_per_dim_stats.json`（domain=6，20k 步）
- 模型 key：`X-VLA-Pt_keys.txt`

**术语**：每个组件的权重按 domain 分成 30 行（`nn.Embedding(num_domains, …)`）。行内 std/l2 相对"未训练行"基线的偏离程度，表示该 domain 是否被训练过。

---

## 第一部分：预训练模型中哪些 domain 做过训练

**判据**：每个组件按 domain 分成 30 行（`nn.Embedding(num_domains, …)`），未训练行 std = 初始化分布的样本标准差——`soft_prompt_hub.weight` ≈0.016（init `normal_(0.02)`）、`action_decoder.fc` ≈0.008 / `action_encoder.fc` ≈0.0043（init `xavier_uniform_`）、两个 bias =0（init `zeros_`）。某行 std 显著高于该基线即为**训练过**。

| 组件 | domain | 未训练基线 std | 预训练基线 std | 结论 |
|---|---|---|---|---|
| soft_prompt_hub.weight | 0-5, 7-9, 18-29 | 0.016 | 0.0163~0.0165 | 未训练 |
| | 6 | 0.016 | **0.0300** | 训练过 |
| | 10-17 | 0.016 | **0.0367~0.0412** | 训练过 |
| action_decoder.fc.weight | 0-5, 7-9, 18-29 | 0.008 | 0.0080~0.0081 | 未训练 |
| | 6 | 0.008 | **0.1431** | 训练过 |
| | 10-17 | 0.008 | **0.0813~0.1801** | 训练过 |
| action_encoder.fc.weight | 0-5, 7-9, 18-29 | 0.0043 | 0.0042~0.0043 | 未训练 |
| | 6 | 0.0043 | **0.0239** | 训练过 |
| | 10-17 | 0.0043 | **0.0210~0.0433** | 训练过 |
| action_decoder.bias.weight | 0-5, 7-9, 18-29 | 0 | 0 | 未训练 |
| | 6 | 0 | **0.0271** | 训练过 |
| | 10-17 | 0 | **0.0165~0.0499** | 训练过 |
| action_encoder.bias.weight | 0-5, 7-9, 18-29 | 0 | 0 | 未训练 |
| | 6 | 0 | **0.0040** | 训练过 |
| | 10-17 | 0 | **0.0045~0.0067** | 训练过 |

**读法**：横向看同一行——预训练 std 显著高于未训练基线即为训练过；纵向看同一 domain 块（6 / 10-17）在 5 个组件上全部偏离，未训练块全部保持基线。

**推论**：预训练基线中 domain 6（robotwin）与 domain 10-17（robomind/droid/AGIBOT 预训练块）参与了预训练；domain 0-5、7-9、18-29 未训练（std 精确等于初始值，bias 从 0 长出亦佐证训练过）。

---

## 第二部分：官方(domain=0) 与自训练(domain=6) 的对比及结论

### 2.1 权重对比表（std / l2）

单元格格式：`std (l2_norm)`。**预训练基线列**为论证"base 是否相同"与"是否重写权重"所加的参照，非题目要求的最简列。

| 组件 | domain | 预训练基线 | 官方 10万步 (domain=0) | 自训练 2万步 (domain=6) |
|---|---|---|---|---|
| soft_prompt_hub.weight | 0 | 0.016 (2.98) | 0.017 (3.02) | 0.016 (2.98) |
| | 6 | **0.030 (5.42)** | 0.016 (2.94) | 0.016 (2.95) |
| | 11-17 | 0.037~0.041 (6.65~7.46) | 0.037~0.041 (6.65~7.46) | 0.037~0.041 (6.65~7.46) |
| action_decoder.fc.weight | 0 | 0.008 (1.15) | **0.109 (15.54)** | 0.008 (1.15) |
| | 6 | **0.143 (20.48)** | 0.008 (1.16) | 0.021 (3.05) |
| | 11-17 | 0.081~0.180 (11.63~25.78) | 0.081~0.180 (11.63~25.78) | 0.081~0.180 (11.63~25.78) |
| action_encoder.fc.weight | 0 | 0.004 (1.16) | **0.028 (7.47)** | 0.004 (1.16) |
| | 6 | **0.024 (6.50)** | 0.004 (1.15) | 0.016 (4.39) |
| | 11-17 | 0.021~0.043 (5.69~11.75) | 0.021~0.043 (5.69~11.75) | 0.021~0.043 (5.69~11.75) |
| action_decoder.bias.weight | 0 | 0.00 (0.00) | **0.040 (0.19)** | 0.00 (0.00) |
| | 6 | **0.027 (0.12)** | 0.00 (0.00) | 0.014 (0.06) |
| | 11-17 | 0.016~0.050 (0.08~0.23) | 0.016~0.050 (0.08~0.23) | 0.016~0.050 (0.08~0.23) |
| action_encoder.bias.weight | 0 | 0.00 (0.00) | **0.007 (0.22)** | 0.00 (0.00) |
| | 6 | **0.004 (0.13)** | 0.00 (0.00) | 0.003 (0.10) |
| | 11-17 | 0.005~0.007 (0.14~0.21) | 0.005~0.007 (0.14~0.21) | 0.005~0.007 (0.14~0.21) |

**表格读法**：
- **11-17 共享块**：三个 checkpoint 逐位相同（精确到 4 位小数）——从未被任何微调改动。
- **官方**：只在 domain 0 上长出强权重（decoder.fc 0.008→0.109），domain 6 精确等于初始值（0.0081）。
- **自训练**：domain 0 未动；domain 6 被大幅改写（decoder.fc 0.143→0.021，soft_prompt 5.42→2.95）。

### 2.2 代码证据：微调不会重置任何权重

1. **初始化只在模型创建时发生**。`DomainAwareLinear.fc`=xavier_uniform、`bias`=zeros、`soft_prompt_hub`=normal(0.02)——但这是 `__init__` 的默认值，`from_pretrained` 加载时会被覆盖（[transformer.py:221-224](models/transformer.py#L221-L224)、[transformer.py:336-337](models/transformer.py#L336-L337)）。
2. **`from_pretrained` 用 checkpoint 权重覆盖全部同名参数**。`XVLA` 继承 `PreTrainedModel`（[modeling_xvla.py:39](models/modeling_xvla.py#L39)），标准 HF 加载逻辑，已存在的 key 一律从权重文件载入，不会重新初始化。
3. **action_mode 覆盖不触发重初始化**。训练从 checkpoint 的 `ee6d` 覆盖为 `arx_ee6d`，两者 `dim_action` 都是 20（[action_hub.py:113](models/action_hub.py#L113)、[action_hub.py:267-272](models/action_hub.py#L267-L272)），无形状不匹配 → 无 reinit。
4. **训练只更新目标行，非目标行梯度为 0**。forward 中 `action_encoder`/`action_decoder`/`soft_prompt_hub` 按 `domain_id` 只 gather 一行（[transformer.py:240-246](models/transformer.py#L240-L246)、[transformer.py:374-403](models/transformer.py#L374-L403)），其它行的梯度恒为 0，权重不变。
5. **weight_decay=0**（[train.py:134](train.py#L134)）——权重收缩纯由数据梯度驱动，不是正则化。
6. **经验证据**：dims 1-5、7-9、18-29 与 11-17 在三个 checkpoint 中 std/l2 逐位相同，说明微调确实只动了目标行。

### 2.3 结论 A：官方与自训练的 base 模型不同

- 官方只训练 domain 0（ee-0）；按 2.2，非目标行（含 domain 6）梯度为 0、权重不变。
- 因此官方 checkpoint 的 domain 6 = 官方 **base** 的 domain 6。观测到官方 domain 6 = 0.0081，**精确等于初始值**（与 1-5/7-9 未训练行同值）。
- 而自训练的 base（预训练基线）domain 6 = 0.143（已训练）。
- **推论**：官方 base 中 domain 6 本来就是初始值 → **官方 base ≠ 我的 base**。两者共享 10-17 预训练核心块（逐位相同），是同源但不同 release 的模型。

### 2.4 结论 B：自训练在重写权重，没有利用 robotwin 的先验

- 自训练目标行 = domain 6，从预训练基线 0.143（decoder.fc）出发，20k 步后掉到 0.021；soft_prompt 5.42→2.95（≈初始值）、bias 0.12→0.06。
- weight_decay=0 → 这是 arx 数据梯度**主动改写**，不是正则化收缩。
- 对照官方：domain 0 从初始值 0.008 **涨**到 0.109（建立强权重）；而自训练 domain 6 是从预训练值**跌**向初始值。
- **推论**：自训练把 robotwin 预训练权重稀释/覆盖，没有继承其优势；robotwin 先验在当前 arx 训练目标（loss/数据）下没有提供帮助。domain 6 最终 std=0.021 仍高于初始值 0.008（2.6 倍），说明该行确实被训练（非重置），但幅度远低于预训练（0.143）与官方平衡点（0.109）——训练方向偏离了 robotwin 权重，且步数不足。

---

## 第三部分：robotwin 数据分析——同类型机械臂，但 loss 不同

### 3.1 robotwin 与 arx_x5 数据规格对比

| 项 | robotwin-2.0 | arx_x5（GOAI 2026） |
|---|---|---|
| 任务类型 | **双臂桌面**操作（RoboTwin-2.0） | **双臂** end-effector |
| 原始动作 | 每臂 endpose `[T,7]`=xyz+quat，gripper `[T]` | Lerobot v3，20 维（16 维自动转 20） |
| 转换后 action | **20 维** `[l_xyz(3),l_rot6d(6),l_g(1), r_xyz(3),r_rot6d(6),r_g(1)]` | **20 维** 同布局 |
| proprio | 14 维（左右各 7） | 20 维（左右各 10） |
| 相机 | **4 路**（head/left/right/front） | **3 路**（cam_high/left_wrist/right_wrist） |
| 数据格式 | HDF5 | LeRobot v3（parquet+mp4） |
| 频率 | 30Hz | 动作时间轴解耦，~25fps |

证据：[simulations.py:152-166](xvla_datasets/domain_handler/simulations.py#L152-L166)、[client.py:112-134](evaluation/robotwin-2.0/client.py#L112-L134)、[lerobot_v3_robodojo.py:37-39](xvla_datasets/domain_handler/lerobot_v3_robodojo.py#L37-L39)。

### 3.2 loss（action mode）对比

| | ee6d（robotwin/预训练用，默认） | arx_ee6d（自训练用） |
|---|---|---|
| gripper loss | **BCE 二分类** | **MSE 连续值** |
| gripper postprocess | sigmoid | 无（连续值即目标） |
| XYZ_SCALE | 500 | 100 |
| GRIPPER_SCALE | 1.0 | 10.0 |
| ROT_SCALE | 10 | 10 |

证据：[action_hub.py:109-139](models/action_hub.py#L109-L139)、[action_hub.py:267-300](models/action_hub.py#L267-L300)。

### 3.3 推论

1. **robotwin 与 arx 是同一类机械臂/动作空间**：都是双臂、20 维 ee 控制、桌面操作，动作布局逐位相同。domain 6 学的不是"错误的表示"，而是双臂 ee 控制本身。
2. **但训练目标（loss）不同**：预训练用 ee6d（gripper 二分类 BCE、位置 loss 权重 XYZ=500），自训练用 arx_ee6d（gripper 连续 MSE、XYZ=100）。目标函数切换（尤其 gripper 从二分类变连续值、位置权重从 500→100）本身就会要求重写 domain-6 权重。
3. **可能机制**：预训练 domain 6 权重是 ee6d loss 下的最优解；切到 arx_ee6d loss 后不再是该目标的最优，梯度把权重向新目标改写——这与"robotwin 与 arx 本体不同"是两个相互独立的因素。因此第二部分观察到的 0.143→0.021 塌缩，不能全部归因于"机器人不同"，**loss 不匹配是一个需要单独验证的候选原因**。
4. 图像 4 路 vs 3 路影响的是共享 VLM（Florence2），不落在 domain 行上，与 domain-6 先验迁移无关。

---

## 附：建议的验证实验

1. **验证 loss 影响**：用 `ee6d`（与预训练同款 loss）在 domain 6 微调一版，对比是否还出现大幅塌缩——可区分"loss 不匹配"与"robotwin 先验不适用"两个因素。
2. **验证 base 差异**：获取官方 base（domain 6 行=初始值的 release）直接对比权重，确认官方与自训练 base 是否同源。
3. **domain 选择**：若放弃 domain 6，推荐 domain 0（未预训练 + 官方在 sim-arx 上已证明可从零训出强权重 0.109 + 官方 eval 工具链按 domain 0 运行）；需修改 `DATA_DOMAIN_ID` 中 `arx_x5_ee` 的映射或训练入口增加 domain_id 覆盖。
