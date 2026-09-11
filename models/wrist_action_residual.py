"""Independent wrist-conditioned action residual extension for X-VLA.

The base policy keeps its original single-camera token sequence.  Wrist views
are encoded by the frozen Florence vision encoder and routed only to this
module.  R0 predicts a direct SE(3) action residual; R1 additionally applies a
sample-level, per-arm dynamic gate.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .modeling_xvla import XVLA
from .transformer import timestep_embedding


class WristActionResidual(nn.Module):
    def __init__(
        self,
        *,
        visual_dim: int,
        action_dim: int,
        proprio_dim: int,
        time_dim: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        dropout: float,
        use_arm_gate: bool,
        gate_init_logit: float,
        se3_indices: tuple[int, ...],
        left_se3_indices: tuple[int, ...],
        right_se3_indices: tuple[int, ...],
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("wrist_hidden_size must be divisible by wrist_num_heads")
        self.action_dim = action_dim
        self.time_dim = time_dim
        self.use_arm_gate = use_arm_gate
        self.left_se3_indices = left_se3_indices
        self.right_se3_indices = right_se3_indices

        # Separate adapters preserve camera identity before cross-view attention.
        self.left_adapter = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_size), nn.GELU()
        )
        self.right_adapter = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_size), nn.GELU()
        )
        self.main_context_proj = nn.Linear(visual_dim, hidden_size)
        self.query_proj = nn.Linear(
            action_dim * 2 + proprio_dim + time_dim, hidden_size
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, action_dim)

        if use_arm_gate:
            self.arm_gate = nn.Sequential(
                nn.LayerNorm(hidden_size * 3 + proprio_dim + time_dim),
                nn.Linear(hidden_size * 3 + proprio_dim + time_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 2),
            )
            nn.init.zeros_(self.arm_gate[-1].weight)
            nn.init.constant_(self.arm_gate[-1].bias, gate_init_logit)
        else:
            self.arm_gate = None

        mask = torch.zeros(action_dim)
        mask[list(se3_indices)] = 1.0
        self.register_buffer("se3_mask", mask, persistent=True)

        # Exact function equivalence at step zero.
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        self.last_stats: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        *,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
        main_context: torch.Tensor,
        action_noisy: torch.Tensor,
        action_base: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        disable_residual: bool = False,
    ) -> torch.Tensor:
        batch, horizon = action_noisy.shape[:2]
        time = timestep_embedding(t, self.time_dim)
        time_tokens = time[:, None].expand(batch, horizon, self.time_dim)
        proprio_tokens = proprio[:, None].expand(batch, horizon, proprio.shape[-1])
        query = self.query_proj(
            torch.cat(
                [action_noisy, action_base.detach(), proprio_tokens, time_tokens],
                dim=-1,
            )
        )
        main = self.main_context_proj(main_context).unsqueeze(1)
        left = self.left_adapter(left_features)
        right = self.right_adapter(right_features)
        memory = torch.cat([main, left, right], dim=1)
        hidden = self.decoder(query, memory)
        raw = self.output_head(self.output_norm(hidden)) * self.se3_mask.to(
            device=hidden.device, dtype=hidden.dtype
        )

        if self.arm_gate is None:
            gates = raw.new_ones((batch, 2))
        else:
            wrist_context = torch.cat([left.mean(1), right.mean(1)], dim=-1)
            gate_input = torch.cat(
                [wrist_context, self.main_context_proj(main_context), proprio, time],
                dim=-1,
            )
            gates = torch.sigmoid(self.arm_gate(gate_input))

        gate_by_dim = raw.new_ones((batch, self.action_dim))
        gate_by_dim[:, self.left_se3_indices] = gates[:, 0, None]
        gate_by_dim[:, self.right_se3_indices] = gates[:, 1, None]
        effective = raw * gate_by_dim[:, None, :]
        if disable_residual:
            effective = torch.zeros_like(effective)

        with torch.no_grad():
            raw_abs = raw.detach().float().abs()
            effective_abs = effective.detach().float().abs()
            self.last_stats = {
                "residual_raw_mean_abs": raw_abs.mean(),
                "residual_raw_p50_abs": torch.quantile(raw_abs, 0.50),
                "residual_raw_p95_abs": torch.quantile(raw_abs, 0.95),
                "residual_raw_max_abs": raw_abs.max(),
                "residual_effective_mean_abs": effective_abs.mean(),
                "residual_effective_p50_abs": torch.quantile(effective_abs, 0.50),
                "residual_effective_p95_abs": torch.quantile(effective_abs, 0.95),
                "residual_effective_max_abs": effective_abs.max(),
                "residual_effective_left_mean_abs": effective_abs[
                    ..., self.left_se3_indices
                ].mean(),
                "residual_effective_right_mean_abs": effective_abs[
                    ..., self.right_se3_indices
                ].mean(),
            }
            if self.arm_gate is not None:
                self.last_stats.update(
                    {
                        "arm_gate_left": gates[:, 0].mean(),
                        "arm_gate_right": gates[:, 1].mean(),
                        "arm_gate_left_std": gates[:, 0].std(unbiased=False),
                        "arm_gate_right_std": gates[:, 1].std(unbiased=False),
                        "arm_gate_left_p10": torch.quantile(gates[:, 0], 0.10),
                        "arm_gate_right_p10": torch.quantile(gates[:, 1], 0.10),
                        "arm_gate_left_p50": torch.quantile(gates[:, 0], 0.50),
                        "arm_gate_right_p50": torch.quantile(gates[:, 1], 0.50),
                        "arm_gate_left_p90": torch.quantile(gates[:, 0], 0.90),
                        "arm_gate_right_p90": torch.quantile(gates[:, 1], 0.90),
                    }
                )
        return effective


class WristActionResidualXVLA(XVLA):
    """X-VLA whose frozen single-camera base is corrected by wrist residuals."""

    def _se3_index_groups(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """左右臂各自允许残差修正的 action 维度下标。"""
        action_space = self.action_space
        required = ("POS_IDX_1", "POS_IDX_2", "ROT_IDX_1", "ROT_IDX_2")
        if not all(hasattr(action_space, name) for name in required):
            raise ValueError(
                f"R0/R1 requires a dual-arm SE(3) action space; got {self.action_mode!r}"
            )
        left = tuple(action_space.POS_IDX_1) + tuple(action_space.ROT_IDX_1)
        right = tuple(action_space.POS_IDX_2) + tuple(action_space.ROT_IDX_2)
        return left, right

    def _make_wrist_residual(self) -> WristActionResidual:
        """按 config 构造腕部残差分支；output_head 由 WristActionResidual 零初始化。"""
        config = self.config
        action_space = self.action_space
        left, right = self._se3_index_groups()
        return WristActionResidual(
            visual_dim=int(self.vlm.config.projection_dim),
            action_dim=action_space.dim_action,
            proprio_dim=getattr(action_space, "dim_proprio", action_space.dim_action),
            time_dim=config.dim_time,
            hidden_size=int(getattr(config, "wrist_hidden_size", 384)),
            depth=int(getattr(config, "wrist_depth", 3)),
            num_heads=int(getattr(config, "wrist_num_heads", 6)),
            dropout=float(getattr(config, "wrist_dropout", 0.0)),
            use_arm_gate=getattr(config, "wrist_residual_mode", None) == "r1",
            gate_init_logit=float(getattr(config, "wrist_gate_init_logit", -2.0)),
            se3_indices=left + right,
            left_se3_indices=left,
            right_se3_indices=right,
        )

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        mode = getattr(config, "wrist_residual_mode", None)
        if mode not in {"r0", "r1"}:
            raise ValueError(f"wrist_residual_mode must be 'r0' or 'r1', got {mode!r}")
        self._se3_index_groups()  # 提前校验 action space 是双臂 SE(3)
        self.wrist_residual = self._make_wrist_residual()
        self._freeze_base = True
        self._verify_step0 = bool(getattr(config, "wrist_verify_step0", False))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """加载权重后保证腕部残差分支处于正确的初始状态。

        源 checkpoint 不含 `wrist_residual.*`（官方 base 就是如此）时，这些参数不在
        state dict 里，低内存加载路径用 `to_empty` 把它们留成**未初始化内存**，而 XVLA
        没有定义 `_init_weights` 兜底——实测 output_head 会变成 NaN / ~5e20，把
        `WristActionResidual.__init__` 的零初始化覆盖掉，进而破坏契约要求的
        「第 0 步与单主相机 base 数值等价」。

        这里用 HF 自己的 missing_keys 判断，而不是猜路径：源含腕部权重（resume R0/R1
        checkpoint）时一个键都不缺，分支不会被重建，已训权重安全。
        """
        wants_loading_info = bool(kwargs.get("output_loading_info", False))
        kwargs["output_loading_info"] = True
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        missing_wrist = [
            key
            for key in (loading_info or {}).get("missing_keys", ())
            if key.startswith("wrist_residual.")
        ]
        if missing_wrist:
            model.wrist_residual.load_state_dict(
                model._make_wrist_residual().state_dict()
            )
        if wants_loading_info:
            return model, loading_info
        return model

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_freeze_base", False):
            self.vlm.eval()
            self.transformer.eval()
            self.wrist_residual.train(mode)
        return self

    def _encode_main_and_wrists(self, input_ids, image_input, image_mask):
        batch, views = image_input.shape[:2]
        if views != 3 or image_mask.shape != (batch, 3):
            raise ValueError(
                "R0/R1 requires exactly three ordered views [main,left_wrist,right_wrist]; "
                f"got image_input={tuple(image_input.shape)}, mask={tuple(image_mask.shape)}"
            )
        if not image_mask.to(torch.bool).all():
            raise ValueError("R0/R1 requires all three camera views to be valid")
        flat = image_input.flatten(0, 1)
        features = self.vlm._encode_image(flat).view(
            batch, 3, -1, self.vlm.config.projection_dim
        )
        main, left, right = features.unbind(dim=1)
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged, attention_mask = self.vlm._merge_input_ids_with_image_features(
            main, inputs_embeds
        )
        main_encoded = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask, inputs_embeds=merged
        )[0]
        # Empty aux sequence reproduces the original one-camera base token layout.
        empty_aux = features.new_empty((batch, 0, features.shape[-1]))
        return main_encoded, empty_aux, left, right

    def _predict_from_encoded(
        self, main, empty_aux, left, right, domain_id, proprio, x_t, t
    ):
        with torch.no_grad():
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action_base = self.transformer(
                domain_id=domain_id,
                vlm_features=main,
                aux_visual_inputs=empty_aux,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
            )
        residual = self.wrist_residual(
            left_features=left.detach(),
            right_features=right.detach(),
            main_context=main.detach().mean(dim=1),
            action_noisy=x_t_m.detach(),
            action_base=action_base.detach(),
            proprio=proprio_m.detach(),
            t=t,
        )
        if self._verify_step0 and self.training:
            if torch.count_nonzero(residual.detach()).item() != 0:
                raise RuntimeError(
                    "R0/R1 step-0 equivalence failed: zero-initialized residual is nonzero"
                )
            print(
                "[wrist-residual] step-0 equivalence passed: action_final == action_base"
            )
            self._verify_step0 = False
        return action_base + residual

    def _predict(self, input_ids, image_input, image_mask, domain_id, proprio, x_t, t):
        with torch.no_grad():
            encoded = self._encode_main_and_wrists(input_ids, image_input, image_mask)
        return self._predict_from_encoded(*encoded, domain_id, proprio, x_t, t)

    def forward(
        self,
        input_ids,
        image_input,
        image_mask,
        domain_id,
        proprio,
        action,
        frame_weight_loss=None,
    ):
        batch = input_ids.shape[0]
        t = (
            torch.rand(1, device=input_ids.device)
            + torch.arange(batch, device=input_ids.device) / batch
        ) % (1 - 1e-5)
        x_t = torch.randn_like(action) * t[:, None, None] + action * (
            1 - t[:, None, None]
        )
        pred = self._predict(
            input_ids, image_input, image_mask, domain_id, proprio, x_t, t
        )
        return self.action_space.compute_loss(
            pred, action, frame_weight_loss=frame_weight_loss
        )

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids,
        image_input,
        image_mask,
        domain_id,
        proprio,
        steps: int = 10,
    ):
        self.eval()
        batch = input_ids.shape[0]
        action_dim = self.action_space.dim_action
        encoded = self._encode_main_and_wrists(input_ids, image_input, image_mask)
        x1 = torch.randn(
            batch,
            self.num_actions,
            action_dim,
            device=proprio.device,
            dtype=proprio.dtype,
        )
        action = torch.zeros_like(x1)
        for index in range(max(1, int(steps)), 0, -1):
            t = torch.full(
                (batch,),
                index / max(1, int(steps)),
                device=proprio.device,
                dtype=proprio.dtype,
            )
            x_t = x1 * t[:, None, None] + action * (1 - t[:, None, None])
            action = self._predict_from_encoded(*encoded, domain_id, proprio, x_t, t)
        return self.action_space.postprocess(action)
