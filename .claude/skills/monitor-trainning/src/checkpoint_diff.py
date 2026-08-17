#!/usr/bin/env python3
"""
Compare two safetensors checkpoints with pluggable key mapping.

Architecture
------------
  CheckpointData        – one-shot load, holds all tensors in memory
  KeyMapper (Protocol)  – build_mapping + transform_tensor
  IdentityMapper        – default 1:1 mapping, no transform
  CheckpointComparator  – generic key/weight diff + prefix stats
  XVLAKeyMapper         – X-VLA specific remapping + transpose
  XVLASpecialChecker    – shared.weight / image_projection / param_count
  XVLACheckpointAnalyzer– full report wiring

Usage
-----
================================================================================
通用调用示例（任意 safetensors 模型）
================================================================================

from checkpoint_diff import CheckpointData, CheckpointComparator

# 1. 一次性加载两个 checkpoint 到内存
orig = CheckpointData.from_path("model_a.safetensors")
ft   = CheckpointData.from_path("model_b.safetensors")

# 2. 创建比较器（不注入 mapper，默认 IdentityMapper，同名 key 1:1 映射）
comp = CheckpointComparator(orig, ft)

# 3. Key 差异分析
kd = comp.key_diff()
print(f"Identity 映射: {kd.identity_count}, 缺失: {len(kd.missing)}, 新增: {kd.unmapped_ft_count}")

# 4. 权重差异分析（核心：判断哪些参数被真正训练更新）
wd = comp.weight_diff(threshold=3.0, top_n=5)
print(f"实质性更新: {len(wd.updated)} ({wd.update_ratio:.1f}%)")
for mod, cnt in wd.prefix_stats.most_common(5):
    print(f"  {mod}: {cnt}")

# 5. 对指定 key 输出详细 diff 表格
print(comp.sample_report(["model.layers.0.attn.q_proj.weight"], threshold=3.0))

# 6. 完整文本报告
print(comp.full_text_report(threshold=3.0, sample_keys=[...], target_prefixes=["model.layers.0"]))
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence

import torch
from safetensors import safe_open


def parse_slice_spec(spec: str) -> tuple[str, int, int]:
    """解析 --slice 规格：`key@dim:idx`，返回 (key, dim, idx)。

    对任意 key 沿任意维度取任意索引做分析，不预设 domain 语义。示例：
        --slice transformer.action_decoder.fc.weight@0:0   # 取 dim0 第 0 行
        --slice transformer.soft_prompt_hub.weight@0:0
    若某个 key 是 [60, n]，想分析 60 里的第一维就用 `key@0:0`。
    """
    key, _, dim_idx = spec.rpartition("@")
    dim, _, idx = dim_idx.partition(":")
    return key, int(dim), int(idx)


# ============================================================================
# 1. Data layer – one-shot load
# ============================================================================


@dataclass
class CheckpointData:
    """一次性加载 checkpoint 中所有 tensor 到内存，避免后续反复打开文件。

    Attributes
    ----------
    keys : set[str]
        所有参数 key 的集合。
    tensors : dict[str, torch.Tensor]
        key -> float32 Tensor 的字典。所有 tensor 统一转为 float32，
        方便后续做数值比较（不受原始 dtype 影响）。
    param_count : int
        总参数量（所有 tensor numel 之和）。
    file_size : int
        文件字节大小。
    bytes_per_param : float
        文件大小 / 参数量，用于推断保存精度（>3 近似 float32，否则 bfloat16）。
    """

    keys: set[str]
    tensors: dict[str, torch.Tensor]
    param_count: int
    file_size: int
    bytes_per_param: float

    @classmethod
    def from_path(cls, path: str, dtype: torch.dtype = torch.float32) -> CheckpointData:
        """从 safetensors 文件路径加载，一次性读取所有 tensor。

        Parameters
        ----------
        path : str
            .safetensors 文件路径。
        dtype : torch.dtype, default=torch.float32
            加载后统一转换的目标 dtype。建议保持 float32，避免 bf16/fp16
            精度损失干扰后续 diff 计算。
        """
        file_size = Path(path).stat().st_size
        tensors: dict[str, torch.Tensor] = {}
        param_count = 0

        # safe_open 只打开一次，遍历所有 key 读取 tensor
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
            for k in keys:
                t = f.get_tensor(k).to(dtype)
                tensors[k] = t
                param_count += t.numel()

        bytes_per_param = file_size / max(param_count, 1)
        return cls(keys, tensors, param_count, file_size, bytes_per_param)


# ============================================================================
# 2. Mapping layer – pluggable, identity by default
# ============================================================================


class KeyMapper(Protocol):
    """Key 映射协议：用于处理两个 checkpoint 之间 key 命名不一致的情况。

    典型场景：
    - 原始模型使用 vendored 代码，训练后改为 transformers 原生实现，
      导致同一层参数 key 名称不同。
    - 原始模型某一层的 weight 矩阵被转置，对比前需要先做 transform。

    实现类需要完成两件事：
    1. build_mapping: 建立 ft_key -> orig_key 的映射字典。
    2. transform_tensor: 对原始 tensor 做必要的变换（如 transpose），
       使其 shape 和语义与训练后 tensor 对齐。
    """

    def build_mapping(
        self, ft_keys: set[str], orig_keys: set[str]
    ) -> dict[str, str]: ...

    """返回映射字典：{ft_key: orig_key}。

    如果某个 ft_key 在 orig_keys 中找不到对应，则不应出现在返回字典中。
    """

    def transform_tensor(
        self, ft_key: str, orig_tensor: torch.Tensor
    ) -> torch.Tensor: ...

    """根据 ft_key 判断是否需要对 orig_tensor 做变换（如 transpose）。

    默认实现直接返回原 tensor，不做任何修改。
    """


class IdentityMapper:
    """默认映射器：不做任何 key 重映射，同名 key 直接 1:1 对应。

    这是通用场景下的默认行为。如果两个 checkpoint 来自同一套代码、
    同一模型结构，直接用这个 mapper 即可，无需任何定制。
    """

    def build_mapping(self, ft_keys: set[str], orig_keys: set[str]) -> dict[str, str]:
        # 只保留两个 checkpoint 中名称完全相同的 key
        return {k: k for k in ft_keys if k in orig_keys}

    def transform_tensor(self, ft_key: str, orig_tensor: torch.Tensor) -> torch.Tensor:
        return orig_tensor


# ============================================================================
# 3. Generic comparator – model-agnostic
# ============================================================================


@dataclass
class KeyDiffResult:
    """Key 差异分析结果。

    Attributes
    ----------
    missing : list[str]
        原始 checkpoint 有、但训练后 checkpoint 中找不到对应映射的 key。
        即：这些参数在训练后"丢失"了。
    added : list[str]
        训练后 checkpoint 有、但原始 checkpoint 中找不到对应映射的 key。
        即：这些参数是训练后"新增"的。
    renamed_mapping : dict[str, str]
        所有成功建立的映射关系：ft_key -> orig_key。
        包含 identity 映射和 custom 映射。
    identity_count : int
        ft_key 和 orig_key 名称完全相同的映射数量。
    custom_mapped_count : int
        ft_key 和 orig_key 名称不同、通过 custom 规则映射的数量。
    unmapped_ft_count : int
        训练后 checkpoint 中未能建立映射的 key 数量（即 len(added)）。
    """

    missing: list[str]
    added: list[str]
    renamed_mapping: dict[str, str]
    identity_count: int
    custom_mapped_count: int
    unmapped_ft_count: int

    def summary(self) -> str:
        total_mapped = self.identity_count + self.custom_mapped_count
        return (
            f"Key mapping: identity={self.identity_count}, custom={self.custom_mapped_count}, "
            f"total_mapped={total_mapped}, missing={len(self.missing)}, added={self.unmapped_ft_count}"
        )


@dataclass
class WeightDiffResult:
    """权重差异分析结果。

    Attributes
    ----------
    updated : list[str]
        判定为"有实质性更新"的 key 列表。
        判定标准：diff > roundtrip * threshold，即差异显著大于 bf16 精度噪声。
    precision_only : list[str]
        判定为"仅精度差异"的 key 列表。
        判定标准：diff <= roundtrip * threshold，差异可归因于 bf16 保存精度转换。
    new_keys : list[str]
        训练后 checkpoint 中新增、原始中没有对应映射的 key。
    unmatched : list[str]
        映射存在但 shape 不匹配的 key（理论上不应出现，用于兜底）。
    details : dict[str, dict]
        每个 key 的详细 diff 数据：
        {ft_key: {"diff": float, "roundtrip": float, "ratio": float,
                  "verdict": str, "orig_key": str, "is_identity": bool}}
    prefix_stats : Counter
        仅统计 updated keys，按模块前缀（前3段）聚合计数。
        用于快速定位哪些大模块被训练修改得最多。
    threshold : float
        判定实质性更新的 ratio 阈值。
    """

    updated: list[str] = field(default_factory=list)
    precision_only: list[str] = field(default_factory=list)
    new_keys: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    details: dict[str, dict] = field(default_factory=dict)
    prefix_stats: Counter = field(default_factory=Counter)
    threshold: float = 3.0

    @property
    def total_processed(self) -> int:
        """总共处理了多少个 key（所有分类之和）。"""
        return (
            len(self.updated)
            + len(self.precision_only)
            + len(self.new_keys)
            + len(self.unmatched)
        )

    @property
    def update_ratio(self) -> float:
        """实质性更新的 key 占比（百分比）。"""
        return len(self.updated) / max(self.total_processed, 1) * 100


def count_by_prefix(prefix_stats: Counter, prefix: str) -> int:
    """统计前缀下的更新 key 数，兼容两种 checkpoint 保存格式：
      - HF save_pretrained 格式：key 带 `model.` 前缀（如 model.transformer.*）；
      - 裸 state_dict 格式：无前缀（train.py 直接保存的 transformer.* / vlm.*）。
    先按原样匹配；无结果时自动去掉/补上 `model.` 前缀再试，避免用户按文档里的
    `model.` 前缀传入却匹配为 0。
    """
    cnt = sum(c for m, c in prefix_stats.items() if m.startswith(prefix))
    if cnt == 0:
        for variant in (prefix.removeprefix("model."), "model." + prefix):
            if variant != prefix:
                cnt = sum(
                    c for m, c in prefix_stats.items() if m.startswith(variant)
                )
                if cnt:
                    break
    return cnt


class CheckpointComparator:
    """通用 checkpoint 比较器，与具体模型无关。

    通过注入不同的 KeyMapper，可以适配任意模型结构。
    如果不注入 mapper，默认使用 IdentityMapper（1:1 映射）。

    核心设计：所有 tensor 已在 CheckpointData 中加载到内存，
    后续所有分析直接操作内存中的 dict，无需再打开文件。
    """

    def __init__(
        self,
        orig: CheckpointData,
        ft: CheckpointData,
        mapper: Optional[KeyMapper] = None,
    ):
        self.orig = orig
        self.ft = ft
        self.mapper = mapper or IdentityMapper()
        self._mapping: Optional[dict[str, str]] = None

    @property
    def mapping(self) -> dict[str, str]:
        """懒加载 ft_key -> orig_key 映射字典。

        第一次调用时通过 mapper.build_mapping 计算，后续直接复用。
        """
        if self._mapping is None:
            self._mapping = self.mapper.build_mapping(self.ft.keys, self.orig.keys)
        return self._mapping

    # ------------------------------------------------------------------
    # 3.1 Key diff – 分析 key 的缺失/新增/映射关系
    # ------------------------------------------------------------------
    def key_diff(self) -> KeyDiffResult:
        """比较两个 checkpoint 的 key 集合差异。

        返回 KeyDiffResult，包含：
        - 缺失的 key（orig 有，ft 无映射）
        - 新增的 key（ft 有，orig 无映射）
        - identity / custom 映射数量统计
        """
        mapping = self.mapping
        # 所有被映射覆盖的原始 key
        mapped_orig = set(mapping.values())

        # missing: 原始中有，但没有被任何 ft_key 映射到
        missing = sorted(self.orig.keys - mapped_orig)
        # added: ft 中有，但不在 mapping 的 keys 里（即没有对应的 orig_key）
        added = sorted(self.ft.keys - set(mapping.keys()))

        identity_count = sum(1 for fk, ok in mapping.items() if fk == ok)
        custom_mapped_count = len(mapping) - identity_count

        return KeyDiffResult(
            missing=missing,
            added=added,
            renamed_mapping=mapping,
            identity_count=identity_count,
            custom_mapped_count=custom_mapped_count,
            unmapped_ft_count=len(added),
        )

    # ------------------------------------------------------------------
    # 3.2 Weight diff – 核心分析：判断每个参数是否被真正训练更新
    # ------------------------------------------------------------------
    def _key_metrics(
        self,
        orig_t: torch.Tensor,
        ft_t: torch.Tensor,
        threshold: float,
        slice_dim: Optional[int],
        slice_idx: Optional[int],
    ) -> dict:
        """对单个 tensor 对计算差异指标与判定（可选切片）。

        指标：
          diff      = mean(abs(orig - ft))，元素平均绝对差异（真实差异值）
          max_delta = max(abs(orig - ft))，元素最大绝对差异
          rel_delta = diff / mean(abs(orig))，相对原权重量级的相对变化
          roundtrip = mean(abs(orig - orig.bfloat16().float()))，bf16 精度噪声地板
          ratio     = diff / roundtrip；roundtrip==0（基线为 bf16 精确常数/全零）时为 None
          verdict   = updated / precision

        判定（三分法，全零/常数张量不再产生 infx 误报）：
          - roundtrip>0：ratio > threshold → updated，否则 precision（仅精度噪声）；
          - roundtrip==0 且 diff>0：基线为常数（如 LayerNorm weight=1/bias=0），
            无噪声地板可归一化，但确实被训练移动 → updated（ratio=N/A）；
          - roundtrip==0 且 diff==0：两档相同（如全零 final_logits_bias）→ precision，
            不进 updated、不纳入结论。
        """
        if slice_dim is not None and orig_t.ndim > slice_dim:
            idx = torch.tensor([slice_idx], dtype=torch.long)
            orig_t = torch.index_select(orig_t, slice_dim, idx)
            ft_t = torch.index_select(ft_t, slice_dim, idx)

        delta = (orig_t - ft_t).abs()
        diff = float(delta.mean())
        max_delta = float(delta.max())
        roundtrip = float((orig_t.bfloat16().float() - orig_t).abs().mean())
        base_abs = float(orig_t.abs().mean())
        rel_delta = diff / base_abs if base_abs > 0 else float("nan")

        if roundtrip > 0:
            ratio = diff / roundtrip
            verdict = "updated" if ratio > threshold else "precision"
        else:
            ratio = None
            verdict = "updated" if diff > 0 else "precision"
        return {
            "diff": diff,
            "max_delta": max_delta,
            "rel_delta": rel_delta,
            "roundtrip": roundtrip,
            "ratio": ratio,
            "verdict": verdict,
        }

    def weight_diff(
        self,
        threshold: float = 3.0,
        target_prefixes: Optional[Sequence[str]] = None,
        top_n: int = 5,
        slices: Optional[dict[str, tuple[int, int]]] = None,
    ) -> WeightDiffResult:
        """逐参数比较权重值，区分"真正训练更新"和"仅 bf16 精度噪声"。

        核心思路（原代码精华）：
        ------------------------
        1. diff = mean(abs(orig - ft))
           原始权重与训练后权重的平均绝对差异。
        2. roundtrip = mean(abs(orig - orig.bfloat16().float()))
           把原始权重从 fp32 -> bf16 -> fp32 转一圈，由于 bf16 尾数只有 7 bit，
           必然产生精度损失。这个 roundtrip 就是"纯保存格式导致的噪声地板"。
        3. ratio = diff / roundtrip
           - ratio ≈ 1.0：差异全部由 bf16 精度转换解释，权重未被训练触及。
           - ratio > threshold（默认 3.0）：差异显著大于噪声，判定为真正更新。
           - roundtrip==0：ratio=N/A，由 diff 直接判定（见 _key_metrics）。

        Parameters
        ----------
        threshold : float, default=3.0
            ratio 判定阈值。diff > roundtrip * threshold 才认为有实质性更新。
        target_prefixes : list[str] | None
            指定前缀列表，用于定向统计。如果为 None，则输出 top_n 最多的前缀。
        top_n : int, default=5
            前缀统计时取前 N 个。
        slices : dict[str, tuple[int, int]] | None
            通用 per-key 切片：{key: (dim, idx)}，对指定 key 只沿 dim 取 idx 行做 diff。
            不预设 domain 语义；未列出的 key 整张分析。

        Returns
        -------
        WeightDiffResult
        """
        slices = slices or {}
        mapping = self.mapping
        result = WeightDiffResult(threshold=threshold)

        for ft_key in self.ft.keys:
            # 查找 ft_key 对应的原始 key
            orig_key = mapping.get(ft_key)
            if orig_key is None or orig_key not in self.orig.keys:
                # 训练后新增 key，原始中没有对应
                result.new_keys.append(ft_key)
                continue

            # 通过 mapper 对原始 tensor 做必要的变换（如 transpose）
            orig_t = self.mapper.transform_tensor(ft_key, self.orig.tensors[orig_key])
            ft_t = self.ft.tensors[ft_key]

            if orig_t.shape != ft_t.shape:
                # 映射存在但 shape 不匹配（理论上不应出现，用于兜底）
                result.unmatched.append(ft_key)
                continue

            sd, si = slices.get(ft_key, (None, None))
            m = self._key_metrics(orig_t, ft_t, threshold, sd, si)
            result.details[ft_key] = {
                **m,
                "orig_key": orig_key,
                "is_identity": ft_key == orig_key,
            }

            if m["verdict"] == "updated":
                result.updated.append(ft_key)
                # 按前3段前缀聚合，快速定位哪些模块被修改最多
                prefix = ".".join(ft_key.split(".")[:3])
                result.prefix_stats[prefix] += 1
            else:
                result.precision_only.append(ft_key)

        return result

    # ------------------------------------------------------------------
    # 3.3 Sample report – 对指定 key 输出详细对比表格
    # ------------------------------------------------------------------
    def sample_report(
        self,
        sample_keys: Sequence[str],
        threshold: float = 3.0,
        slices: Optional[dict[str, tuple[int, int]]] = None,
    ) -> str:
        """对一组指定的 key 输出详细的 meanΔ / maxΔ / relΔ / ratio / verdict 表格。

        与 weight_diff() 共用同一套内存中的 tensor，无需重新读取文件。
        原代码中 _compare_sample_keys 与 weight_diff 主循环逻辑完全重复，
        重构后统一为单一流，sample_report 只是对已有 tensors 的筛选输出。
        """
        slices = slices or {}
        mapping = self.mapping
        lines: list[str] = []
        header = (
            f"{'key':<52} {'meanΔ':>10} {'maxΔ':>10} {'relΔ':>7} "
            f"{'ratio':>7} {'verdict':>9} {'map':>9}"
        )
        lines.append(header)
        lines.append("-" * 106)

        for ft_key in sample_keys:
            if ft_key not in self.ft.tensors:
                lines.append(f"  {ft_key}: 训练后不存在")
                continue

            orig_key = mapping.get(ft_key, ft_key)
            if orig_key not in self.orig.tensors:
                lines.append(f"  {ft_key}: 原始中无对应 ({orig_key})")
                continue

            orig_t = self.mapper.transform_tensor(ft_key, self.orig.tensors[orig_key])
            ft_t = self.ft.tensors[ft_key]

            if orig_t.shape != ft_t.shape:
                lines.append(f"  {ft_key}: 形状不匹配 {orig_t.shape} vs {ft_t.shape}")
                continue

            sd, si = slices.get(ft_key, (None, None))
            m = self._key_metrics(orig_t, ft_t, threshold, sd, si)
            ratio_s = "N/A" if m["ratio"] is None else f"{m['ratio']:.1f}x"
            rel_s = "nan" if m["rel_delta"] != m["rel_delta"] else f"{m['rel_delta']:.3f}"
            map_type = "identity" if ft_key == orig_key else "remapped"

            # 取最后3段作为短名称，避免表格过宽；切片 key 标注取的行
            short = ".".join(ft_key.split(".")[-3:])
            if si is not None:
                short = f"{short}@{si}"
            lines.append(
                f"  {short:<52} {m['diff']:>10.2e} {m['max_delta']:>10.2e} "
                f"{rel_s:>7} {ratio_s:>7} {m['verdict']:>9} {map_type:>9}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 3.4 Full text report – 整合所有分析输出为文本
    # ------------------------------------------------------------------
    def full_text_report(
        self,
        threshold: float = 3.0,
        sample_keys: Optional[Sequence[str]] = None,
        target_prefixes: Optional[Sequence[str]] = None,
        top_n: int = 5,
        top_ratio: int = 0,
        slices: Optional[dict[str, tuple[int, int]]] = None,
    ) -> str:
        """生成完整的文本报告，整合 key diff + weight diff + sample report。

        这是通用报告，不包含任何模型特定的专项检查（如 X-VLA 的 shared.weight）。
        模型特定的检查由外部 XVLASpecialChecker 完成，通过 XVLACheckpointAnalyzer 拼接。
        """
        slices = slices or {}
        lines: list[str] = []
        lines.append("=" * 100)
        lines.append("Checkpoint Diff 完整报告")
        lines.append("=" * 100)
        lines.append(
            f"原始:   {self.orig.file_size / 1e9:.2f}G  "
            f"{self.orig.param_count / 1e6:.1f}M params  {len(self.orig.keys)} keys"
        )
        lines.append(
            f"训练后: {self.ft.file_size / 1e9:.2f}G  "
            f"{self.ft.param_count / 1e6:.1f}M params  {len(self.ft.keys)} keys"
        )
        lines.append("")

        # ---- Key diff section ----
        kd = self.key_diff()
        lines.append("-" * 50)
        lines.append("Key Diff 统计")
        lines.append("-" * 50)
        lines.append(f"  Identity 映射:     {kd.identity_count}")
        lines.append(f"  Custom 映射:       {kd.custom_mapped_count}")
        lines.append(
            f"  总映射数:          {kd.identity_count + kd.custom_mapped_count}"
        )
        lines.append(f"  缺失 (orig 有 ft 无): {len(kd.missing)}")
        lines.append(f"  新增 (ft 有 orig 无): {kd.unmapped_ft_count}")
        lines.append("")

        # 缺失 key 按模块前缀统计（Top N）
        if kd.missing:
            lines.append(f"=== 缺失 key 按模块统计 (Top {top_n}) ===")
            missing_counter = Counter()
            for k in kd.missing:
                prefix = ".".join(k.split(".")[:3])
                missing_counter[prefix] += 1
            for mod, cnt in missing_counter.most_common(top_n):
                lines.append(f"  {mod}: {cnt}")
            lines.append("")

        # 新增 key 按模块前缀统计（Top N）
        if kd.added:
            lines.append(f"=== 新增 key 按模块统计 (Top {top_n}) ===")
            added_counter = Counter()
            for k in kd.added:
                prefix = ".".join(k.split(".")[:3])
                added_counter[prefix] += 1
            for mod, cnt in added_counter.most_common(top_n):
                lines.append(f"  {mod}: {cnt}")
            lines.append("")

        # ---- Weight diff section ----
        wd = self.weight_diff(
            threshold=threshold,
            target_prefixes=target_prefixes,
            top_n=top_n,
            slices=slices,
        )
        lines.append("-" * 50)
        lines.append("Weight Diff 统计")
        lines.append("-" * 50)
        lines.append(f"  总 key 数:            {wd.total_processed}")
        lines.append(f"  仅精度差异:           {len(wd.precision_only)}")
        lines.append(f"  有实质性更新:         {len(wd.updated)}")
        lines.append(f"  新增 key:             {len(wd.new_keys)}")
        lines.append(f"  形状不匹配:           {len(wd.unmatched)}")
        lines.append(f"  更新比例:             {wd.update_ratio:.1f}%")
        if slices:
            spec_str = ", ".join(
                f"{k}@{d}:{i}" for k, (d, i) in sorted(slices.items())
            )
            lines.append(f"  切片分析:            {spec_str}")
        lines.append("")

        # 按模块前缀统计更新（支持指定前缀或 Top N）
        if target_prefixes:
            lines.append("=== 指定前缀更新统计 ===")
            for prefix in target_prefixes:
                lines.append(f"  {prefix}: {count_by_prefix(wd.prefix_stats, prefix)}")
        else:
            lines.append(f"=== 按模块更新统计 (Top {top_n}) ===")
            for mod, cnt in wd.prefix_stats.most_common(top_n):
                lines.append(f"  {mod}: {cnt}")
        lines.append("")

        # ---- Top N 更新权重（按 meanΔ 降序；meanΔ 为真实差异值，ratio 仅辅助）----
        if top_ratio > 0:
            ranked = sorted(
                ((k, d) for k, d in wd.details.items() if d["verdict"] == "updated"),
                key=lambda kv: kv[1]["diff"],
                reverse=True,
            )[:top_ratio]
            if sample_keys is None:
                sample_keys = [k for k, _ in ranked]
            lines.append("=" * 100)
            lines.append(f"更新幅度最大的 Top {len(ranked)} 权重（按 meanΔ 降序）")
            lines.append("=" * 100)
            lines.append(
                f"{'key':<66} {'meanΔ':>10} {'maxΔ':>10} {'relΔ':>8} {'ratio':>8} {'verdict':>9}"
            )
            lines.append("-" * 100)
            for k, d in ranked:
                ratio_s = "N/A" if d["ratio"] is None else f"{d['ratio']:.1f}x"
                rel_s = "nan" if d["rel_delta"] != d["rel_delta"] else f"{d['rel_delta']:.3f}"
                label = k
                sd, si = slices.get(k, (None, None))
                if si is not None:
                    label = f"{k}@{si}"
                lines.append(
                    f"{label:<66} {d['diff']:>10.2e} {d['max_delta']:>10.2e} "
                    f"{rel_s:>8} {ratio_s:>8} {d['verdict']:>9}"
                )
            lines.append("")

        # ---- Sample keys ----
        if sample_keys:
            lines.append("=" * 100)
            lines.append("采样 key 对比")
            lines.append("=" * 100)
            lines.append(self.sample_report(sample_keys, threshold, slices))
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# 4. X-VLA custom mapper – 隔离模型特定的 key 重映射逻辑
# ============================================================================


class XVLAKeyMapper:
    """X-VLA 场景专用的 KeyMapper。

    原始 xvla-base 使用 vendored Florence-2 代码，训练后改为 transformers 原生实现，
    导致大量 key 名称变更。本类集中处理这些映射规则。

    映射规则（按优先级）：
    1. Identity 映射：先尝试同名 key 直接对应。
    2. image_projection：原始 key 为 model.vlm.image_projection，
       训练后为 model.vlm.multi_modal_projector.image_projection.weight，
       且 weight 矩阵需要 transpose(0, 1)。
    3. language_model：去掉 .model. 层级（vendored -> native）。
    4. multi_modal_projector 子模块：image_position_embed、image_proj_norm、
       visual_temporal_embed 的命名变更。
    5. vision_tower：DaViT 的 channel_attn / window_attn 中去掉 .fn. 层级。
    6. conv：.conv. -> .proj.
    7. ffn：fc1/fc2 -> fn.net.0/3
    8. norms：norm1/norm2 的层级调整。
    """

    def build_mapping(self, ft_keys: set[str], orig_keys: set[str]) -> dict[str, str]:
        # 第一步：先建立 identity 映射（同名 key 直接对应）
        mapping: dict[str, str] = {k: k for k in ft_keys if k in orig_keys}

        # 第二步：对未映射的 ft_key，尝试 custom 规则
        unmapped = ft_keys - set(mapping.keys())
        for ft_key in unmapped:
            candidate = self._find_candidate(ft_key, orig_keys)
            if candidate:
                mapping[ft_key] = candidate
        return mapping

    def _find_candidate(self, ft_key: str, orig_keys: set[str]) -> Optional[str]:
        """对单个 ft_key 尝试所有 custom 映射规则，返回第一个匹配的 orig_key。"""

        # 1. image_projection（特殊：需要 transpose，见 transform_tensor）
        if ft_key == "model.vlm.multi_modal_projector.image_projection.weight":
            c = "model.vlm.image_projection"
            return c if c in orig_keys else None

        # 2. language_model: 去掉 .model. 层级
        #    vendored: model.vlm.language_model.model.encoder.xxx
        #    native:   model.vlm.language_model.encoder.xxx
        if "model.vlm.language_model.encoder." in ft_key:
            c = ft_key.replace(
                "model.vlm.language_model.encoder.",
                "model.vlm.language_model.model.encoder.",
            )
            return c if c in orig_keys else None

        # 3. multi_modal_projector 子模块命名变更
        if "multi_modal_projector.image_position_embed" in ft_key:
            c = ft_key.replace(
                "multi_modal_projector.image_position_embed", "image_pos_embed"
            )
            return c if c in orig_keys else None
        if "multi_modal_projector.image_proj_norm" in ft_key:
            c = ft_key.replace("multi_modal_projector.", "")
            return c if c in orig_keys else None
        if "multi_modal_projector.visual_temporal_embed" in ft_key:
            c = ft_key.replace("multi_modal_projector.", "")
            return c if c in orig_keys else None

        # 4. vision_tower: DaViT 的 channel_attn / window_attn 去掉 .fn.
        if "channel_attn." in ft_key or "window_attn." in ft_key:
            for suffix in (
                ".qkv.",
                ".proj.",
                ".q_proj.",
                ".k_proj.",
                ".v_proj.",
                ".out_proj.",
            ):
                if suffix in ft_key:
                    c = ft_key.replace(suffix, ".fn" + suffix)
                    if c in orig_keys:
                        return c

        # 5. conv: .conv. -> .proj.
        if "convs." in ft_key and ".conv." in ft_key:
            c = ft_key.replace(".conv.", ".proj.")
            return c if c in orig_keys else None

        # 6. ffn: fc1/fc2 -> fn.net.0/3
        if ".ffn.fc1." in ft_key:
            c = ft_key.replace(".ffn.fc1.", ".ffn.fn.net.0.")
            return c if c in orig_keys else None
        if ".ffn.fc2." in ft_key:
            c = ft_key.replace(".ffn.fc2.", ".ffn.fn.net.3.")
            return c if c in orig_keys else None

        # 7. norms: norm1/norm2 的层级调整
        if ".norm1." in ft_key:
            for attn in ("channel_attn", "window_attn"):
                c = ft_key.replace(".norm1.", f".{attn}.norm.")
                if c in orig_keys:
                    return c
        if ".norm2." in ft_key:
            c = ft_key.replace(".norm2.", ".ffn.norm.")
            return c if c in orig_keys else None

        return None

    def transform_tensor(self, ft_key: str, orig_tensor: torch.Tensor) -> torch.Tensor:
        """对原始 tensor 做必要的变换，使其与训练后 tensor 对齐。

        目前只有 image_projection 需要 transpose(0, 1)，因为 vendored 版本和 native 版本
        对该 weight 矩阵的维度顺序定义相反。
        """
        if ft_key == "model.vlm.multi_modal_projector.image_projection.weight":
            return orig_tensor.transpose(0, 1).contiguous()
        return orig_tensor


# ============================================================================
# 5. X-VLA special checks – 模型特定的专项检查
# ============================================================================


class XVLASpecialChecker:
    """X-VLA 场景特有的额外检查，与通用分析器解耦。

    包含三项检查：
    1. shared.weight 修复验证：检查 vendored -> native 转换后，
       shared.weight 和 embed_tokens.weight 的数值一致性。
    2. image_projection 转置验证：确认 transpose 操作后数值是否对齐。
    3. 参数量对比：统计两个 checkpoint 的参数量差异和推断保存精度。
    """

    def __init__(self, orig: CheckpointData, ft: CheckpointData):
        self.orig = orig
        self.ft = ft

    def check_shared_weight(self) -> list[str]:
        """验证 shared.weight 在 vendored -> native 转换后的修复是否安全。

        vendored 版本使用 shared.weight 作为 tie-weight，native 版本拆分为
        embed_tokens.weight。需要确认两者数值一致，避免修复引入偏差。
        """
        lines = ["\n--- shared.weight 修复验证 ---"]
        orig_has = "model.vlm.language_model.shared.weight" in self.orig.keys
        ft_has = "model.vlm.language_model.shared.weight" in self.ft.keys
        ft_has_embed = (
            "model.vlm.language_model.encoder.embed_tokens.weight" in self.ft.keys
        )

        lines.append(f"  原始有 shared.weight: {orig_has}")
        lines.append(f"  训练后有 shared.weight: {ft_has}")
        lines.append(f"  训练后有 embed_tokens.weight: {ft_has_embed}")

        if not ft_has and ft_has_embed:
            # 训练后没有 shared.weight，但有 embed_tokens.weight
            # 需要验证 embed_tokens 和原始的 shared.weight 是否一致
            ft_embed = self.ft.tensors[
                "model.vlm.language_model.encoder.embed_tokens.weight"
            ]
            orig_key = None
            for cand in (
                "model.vlm.language_model.encoder.embed_tokens.weight",
                "model.vlm.language_model.model.encoder.embed_tokens.weight",
            ):
                if cand in self.orig.keys:
                    orig_key = cand
                    break

            if orig_key:
                orig_embed = self.orig.tensors[orig_key]
                diff = (orig_embed - ft_embed).abs().mean().item()
                rt = (orig_embed.bfloat16().float() - orig_embed).abs().mean().item()
                ratio = diff / rt if rt > 0 else 0
                ok = ratio < 3.0
                lines.append(
                    f"  修复后一致性: diff={diff:.8f} rt={rt:.8f} ratio={ratio:.1f}x "
                    f"-> {'OK' if ok else '*** 有差异 ***'}"
                )
            else:
                lines.append("  原始中也找不到 embed_tokens，无法对比")
        elif ft_has:
            lines.append("  训练后 shared.weight 存在，无需修复")
        else:
            lines.append("  *** 训练后既无 shared 也无 embed_tokens，可能有问题 ***")
        return lines

    def check_image_projection(self) -> list[str]:
        """验证 image_projection 的 transpose 操作是否正确。

        原始 vendored 版本的 shape 和 native 版本相反，
        transpose(0, 1) 后应与训练后版本对齐。
        """
        lines = ["\n--- image_projection 转置验证 ---"]
        if "model.vlm.image_projection" not in self.orig.keys:
            lines.append("  原始模型无 image_projection（可能已是 native 格式）")
            return lines

        orig_proj = self.orig.tensors["model.vlm.image_projection"]
        ft_key = "model.vlm.multi_modal_projector.image_projection.weight"
        if ft_key not in self.ft.keys:
            lines.append(f"  训练后无 {ft_key}")
            return lines

        ft_proj = self.ft.tensors[ft_key]
        orig_t = orig_proj.transpose(0, 1).contiguous()

        lines.append(f"  原始 shape: {orig_proj.shape} -> 转置后: {orig_t.shape}")
        lines.append(f"  训练后 shape: {ft_proj.shape}")

        diff = (orig_t - ft_proj).abs().mean().item()
        rt = (orig_t.bfloat16().float() - orig_t).abs().mean().item()
        ratio = diff / rt if rt > 0 else 0
        ok = ratio < 3.0
        lines.append(
            f"  diff={diff:.8f} rt={rt:.8f} ratio={ratio:.1f}x "
            f"-> {'OK (仅精度差异)' if ok else '有实质性更新'}"
        )
        return lines

    def check_param_count(self) -> list[str]:
        """对比两个 checkpoint 的参数量和推断保存精度。"""
        lines = ["\n--- 参数量对比 ---"]
        lines.append(
            f"  原始:   {self.orig.param_count:,} ({self.orig.param_count / 1e6:.1f}M)  "
            f"keys: {len(self.orig.keys)}"
        )
        lines.append(
            f"  训练后: {self.ft.param_count:,} ({self.ft.param_count / 1e6:.1f}M)  "
            f"keys: {len(self.ft.keys)}"
        )
        diff = self.ft.param_count - self.orig.param_count
        lines.append(f"  差异: {diff:,} ({diff / 1e6:.1f}M)")

        # 根据 bytes_per_param 推断保存精度
        o_dtype = "float32" if self.orig.bytes_per_param > 3 else "bfloat16"
        f_dtype = "float32" if self.ft.bytes_per_param > 3 else "bfloat16"
        lines.append(f"  原始每参数字节:   {self.orig.bytes_per_param:.1f} ({o_dtype})")
        lines.append(f"  训练后每参数字节: {self.ft.bytes_per_param:.1f} ({f_dtype})")
        return lines


# ============================================================================
# 6. X-VLA analyzer – 组装通用分析 + 专项检查
# ============================================================================


class XVLACheckpointAnalyzer:
    """X-VLA 场景的一站式分析器。

    将通用 CheckpointComparator（注入 XVLAKeyMapper）与
    XVLASpecialChecker 拼接，输出完整报告。
    """

    def __init__(self, orig_path: str, ft_path: str):
        self.orig = CheckpointData.from_path(orig_path)
        self.ft = CheckpointData.from_path(ft_path)
        self.comparator = CheckpointComparator(self.orig, self.ft, XVLAKeyMapper())
        self.checker = XVLASpecialChecker(self.orig, self.ft)

    def full_report(
        self,
        threshold: float = 3.0,
        sample_keys: Optional[Sequence[str]] = None,
        target_prefixes: Optional[Sequence[str]] = None,
        top_n: int = 5,
        top_ratio: int = 0,
        slices: Optional[dict[str, tuple[int, int]]] = None,
    ) -> str:
        """生成完整报告：通用分析 + X-VLA 专项检查。"""
        lines: list[str] = []
        lines.append(
            self.comparator.full_text_report(
                threshold=threshold,
                sample_keys=sample_keys,
                target_prefixes=target_prefixes,
                top_n=top_n,
                top_ratio=top_ratio,
                slices=slices,
            )
        )
        lines.extend(self.checker.check_shared_weight())
        lines.extend(self.checker.check_image_projection())
        lines.extend(self.checker.check_param_count())
        return "\n".join(lines)


# ============================================================================
# 7. CLI
# ============================================================================


def _add_slice_arg(parser: argparse.ArgumentParser) -> None:
    """为子命令添加通用 per-key 切片参数 --slice key@dim:idx（可重复）。"""
    parser.add_argument(
        "--slice",
        action="append",
        default=None,
        metavar="KEY@DIM:IDX",
        help=("通用 per-key 切片：只对指定 key 沿 dim 取 idx 行做 diff/统计，"
              "如 --slice transformer.action_decoder.fc.weight@0:0。"
              "可重复；未列出的 key 整张分析。"),
    )


def _slices_from(specs: Optional[list[str]]) -> dict[str, tuple[int, int]]:
    """把 --slice 参数列表解析成 {key: (dim, idx)}。"""
    out: dict[str, tuple[int, int]] = {}
    for spec in specs or []:
        key, dim, idx = parse_slice_spec(spec)
        out[key] = (dim, idx)
    return out


def _default_samples(ft_keys: set[str]) -> list[str]:
    """X-VLA 默认采样 key，用于快速查看关键层的更新情况。

    兼容两种 checkpoint 保存格式：
      - HF save_pretrained 格式：key 带 `model.` 前缀（XVLAKeyMapper 的映射目标）；
      - 裸 state_dict 格式：无前缀（train.py 直接保存的 transformer.* / vlm.*）。
    对每个规范 key 依次尝试：原名 → 去掉 `model.` 前缀 → 按最后两段后缀匹配实际
    存在的 key，保证两种格式都能采到样，而不是全部显示"训练后不存在"。
    """
    canonical = [
        "model.vlm.language_model.encoder.layers.0.self_attn.q_proj.weight",
        "model.vlm.vision_tower.blocks.0.0.channel_block.channel_attn.qkv.weight",
        "model.vlm.multi_modal_projector.image_projection.weight",
        "model.vlm.language_model.encoder.embed_tokens.weight",
        "model.transformer.action_decoder.fc.weight",
        "model.transformer.soft_prompt_hub.weight",
    ]
    resolved: list[str] = []
    for key in canonical:
        if key in ft_keys:
            resolved.append(key)
            continue
        stripped = key[len("model."):] if key.startswith("model.") else key
        if stripped in ft_keys:
            resolved.append(stripped)
            continue
        suffix = ".".join(key.split(".")[-2:])
        match = next((k for k in ft_keys if k.endswith("." + suffix)), None)
        resolved.append(match if match is not None else stripped)
    return resolved


def main() -> None:
    """
    Lerobot  X-VLA 调用示例：
     ----------------
     # Python API 方式
     from checkpoint_diff import XVLACheckpointAnalyzer

     analyzer = XVLACheckpointAnalyzer(
         "xvla-base.safetensors",
         "ft-30k.safetensors"
     )
     report = analyzer.full_report(
         threshold=3.0,
         sample_keys=["model.transformer.action_decoder.fc.weight"],
         target_prefixes=["model.transformer"],
         top_n=5,
     )
     print(report)

     # CLI 方式
     python checkpoint_diff.py full xvla-base.safetensors ft-30k.safetensors --threshold 3.0
     python checkpoint_diff.py full xvla-base.safetensors ft-30k.safetensors --prefixes model.transformer model.vlm.vision_tower
    """

    parser = argparse.ArgumentParser(
        description="Compare two safetensors checkpoints with pluggable key mapping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- key-diff ----
    p_key = sub.add_parser("key-diff", help="仅对比 key 差异")
    p_key.add_argument("orig", help="Original model.safetensors")
    p_key.add_argument("ft", help="Fine-tuned model.safetensors")

    # ---- weight-diff ----
    p_wt = sub.add_parser("weight-diff", help="对比权重差异，输出统计")
    p_wt.add_argument("orig", help="Original model.safetensors")
    p_wt.add_argument("ft", help="Fine-tuned model.safetensors")
    p_wt.add_argument(
        "--threshold", type=float, default=3.0, help="Ratio threshold (default: 3.0)"
    )
    p_wt.add_argument(
        "--prefixes", nargs="+", default=None, help="Target prefixes to report"
    )
    _add_slice_arg(p_wt)

    # ---- full ----
    p_full = sub.add_parser("full", help="X-VLA 完整报告（key + weight + 专项检查）")
    p_full.add_argument("orig", help="Original model.safetensors")
    p_full.add_argument("ft", help="Fine-tuned model.safetensors")
    p_full.add_argument(
        "--threshold", type=float, default=3.0, help="Ratio threshold (default: 3.0)"
    )
    p_full.add_argument(
        "--prefixes", nargs="+", default=None, help="Target prefixes to report"
    )
    p_full.add_argument(
        "--top-ratio",
        type=int,
        default=40,
        help="输出更新幅度最大的 Top N 权重表（按 meanΔ 降序），并把采样段改为这些 key；0 关闭",
    )
    _add_slice_arg(p_full)

    args = parser.parse_args()

    if args.command == "key-diff":
        comp = CheckpointComparator(
            CheckpointData.from_path(args.orig),
            CheckpointData.from_path(args.ft),
            XVLAKeyMapper(),
        )
        kd = comp.key_diff()
        print(f"原始 key 数: {len(comp.orig.keys)}")
        print(f"训练后 key 数: {len(comp.ft.keys)}")
        print(f"\n=== Identity 映射: {kd.identity_count} ===")
        print(f"=== Custom 映射:   {kd.custom_mapped_count} ===")
        print(f"=== 缺失 ({len(kd.missing)} 个) ===")
        for k in kd.missing[:20]:
            print(f"  - {k}")
        print(f"\n=== 新增 ({kd.unmapped_ft_count} 个) ===")
        for k in kd.added[:20]:
            print(f"  + {k}")

    elif args.command == "weight-diff":
        comp = CheckpointComparator(
            CheckpointData.from_path(args.orig),
            CheckpointData.from_path(args.ft),
            XVLAKeyMapper(),
        )
        wd = comp.weight_diff(
            threshold=args.threshold,
            target_prefixes=args.prefixes,
            slices=_slices_from(args.slice),
        )
        print(f"总 key: {wd.total_processed}")
        print(f"精度差异: {len(wd.precision_only)}")
        print(f"实质性更新: {len(wd.updated)}")
        print(f"新增: {len(wd.new_keys)}")
        print(f"不匹配: {len(wd.unmatched)}")
        print(f"更新比例: {wd.update_ratio:.1f}%")
        if args.slice:
            print(f"切片分析: {', '.join(args.slice)}")
        print("\n=== 按模块更新统计 (Top 5) ===")
        for mod, cnt in wd.prefix_stats.most_common(5):
            print(f"  {mod}: {cnt}")

    elif args.command == "full":
        analyzer = XVLACheckpointAnalyzer(args.orig, args.ft)
        print(
            analyzer.full_report(
                threshold=args.threshold,
                sample_keys=None if args.top_ratio else _default_samples(analyzer.ft.keys),
                target_prefixes=args.prefixes,
                top_ratio=args.top_ratio,
                slices=_slices_from(args.slice),
            )
        )


if __name__ == "__main__":
    main()
