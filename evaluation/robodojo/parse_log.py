# ------------------------------------------------------------------------------
# RoboDojo policy-server 日志解析：从策略服务器 [xvla_2][io] 日志还原每预测
# (state16/state20/完整 30x16 动作 chunk)；可选合并仿真端 [xvla_2][sim] 日志
# （sim_step_log 产物）成每步 (state16, action16)。CLI 导出 CSV / parquet。
#
# 权威日志是策略服务器 [xvla_2][io]（每 30 步 1 次预测）；仿真端日志去向不可控
# （落入仿真 stdout），仅作可选的逐帧补充。
# ------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

LOG_PREFIX = "[xvla_2][io]"
SIM_PREFIX = "[xvla_2][sim]"


def _iter_lines(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or not line.strip():
                continue
            yield line


def _split_prefix(line: str, prefix: str) -> dict[str, Any] | None:
    idx = line.find(prefix)
    if idx < 0:
        return None
    try:
        return json.loads(line[idx + len(prefix):].strip())
    except json.JSONDecodeError:
        return None


def parse_policy_log(path: str | Path) -> dict[str, Any]:
    """解析策略服务器 [xvla_2][io] 日志，按 (env_idx, request) 聚合出每预测。

    返回：
      {
        "envs": {"<env_idx>": {"predictions": [per-predict dict, request 升序],
                               "resets": int}},
        "events": [原始 init / reset / prepare_case / trial_end 事件],
      }
    每个 per-predict dict 含：
      request, env_idx, instruction, state16, state20,
      action16_chunk (完整 [T,16] 动作列表), images
    """
    envs: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    pending_observation: dict[str, dict[str, Any]] = {}  # (env_idx) -> obs 事件

    for line in _iter_lines(path):
        event = _split_prefix(line, LOG_PREFIX)
        if event is None:
            continue
        kind = event.get("event")
        env_idx = str(event.get("env_idx", 0))
        if kind in ("init", "reset", "prepare_case", "trial_end"):
            events.append(event)
            env = envs.setdefault(env_idx, {"predictions": [], "resets": 0})
            if kind == "reset":
                env["resets"] += 1
            continue
        if kind == "client_observation":
            pending_observation[env_idx] = event
            continue
        if kind == "server_actions":
            obs = pending_observation.pop(env_idx, {})
            env = envs.setdefault(env_idx, {"predictions": [], "resets": 0})
            env["predictions"].append({
                "request": int(event.get("request", obs.get("request", 0))),
                "env_idx": env_idx,
                "instruction": obs.get("instruction", ""),
                "state16": obs.get("state16"),
                "state20": obs.get("state20"),
                "images": obs.get("images"),
                "action16_chunk": event.get("action16"),  # 完整 [T,16]
                "num_actions": event.get("num_actions"),
            })
            continue

    return {"envs": envs, "events": events}


def parse_sim_log(path: str | Path) -> list[dict[str, Any]]:
    """解析仿真端 [xvla_2][sim] 日志（sim_step_log 产物），按出现顺序返回 step 事件。"""
    steps: list[dict[str, Any]] = []
    for line in _iter_lines(path):
        event = _split_prefix(line, SIM_PREFIX)
        if event is not None:
            steps.append(event)
    return steps


def merge_sim_steps(
    policy: dict[str, Any], sim_steps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """把每步 (state16) 仿真日志与每预测 action chunk 合并成每步 (state16, action16)。

    全 chunk 模式：第 i 次预测返回 T 个动作，依次对应该 env 后续 T 个
    step_observation（每步取一个动作作为 action16）。按 env 分别对齐。
    """
    if not sim_steps:
        return []
    by_env = {str(k): v["predictions"] for k, v in policy["envs"].items()}
    per_env_events: dict[str, list[dict[str, Any]]] = {}
    for event in sim_steps:
        per_env_events.setdefault(str(event.get("env_idx", 0)), []).append(event)
    result: list[dict[str, Any]] = []
    for env_idx, preds in by_env.items():
        env_events = per_env_events.get(env_idx, [])
        pos = 0
        for p in preds:
            chunk = p.get("action16_chunk") or []
            for action in chunk:
                if pos >= len(env_events):
                    break
                ev = env_events[pos]
                result.append({
                    "step": int(ev.get("step", 0)),
                    "env_idx": env_idx,
                    "state16": ev.get("state16"),
                    "images": ev.get("images"),
                    "action16": action,
                })
                pos += 1
    result.sort(key=lambda r: (r["env_idx"], r["step"]))
    return result


def to_rows(policy: dict[str, Any]) -> list[dict[str, Any]]:
    """把解析结果展平成可导出 CSV/parquet 的行（每预测一行）。"""
    rows: list[dict[str, Any]] = []
    for env_idx, env in policy["envs"].items():
        for p in env["predictions"]:
            rows.append({
                "env_idx": env_idx,
                "request": p["request"],
                "instruction": p.get("instruction", ""),
                "state16": p.get("state16"),
                "state20": p.get("state20"),
                "action16_chunk": p.get("action16_chunk"),
            })
    rows.sort(key=lambda r: (r["env_idx"], r["request"]))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="解析 [xvla_2][io] 策略服务器日志，可选合并 [xvla_2][sim] 仿真日志。"
    )
    parser.add_argument("log", help="策略服务器日志文件路径")
    parser.add_argument("--sim-log", help="可选：仿真端 [xvla_2][sim] 日志路径")
    parser.add_argument(
        "--merge", action="store_true",
        help="合并 sim-log 成每步 (state16, action16)（需同时给 --sim-log）",
    )
    parser.add_argument("--out", help="输出 CSV/parquet（按扩展名），默认 stdout JSON")
    args = parser.parse_args(argv)

    policy = parse_policy_log(args.log)
    if args.merge:
        if not args.sim_log:
            parser.error("--merge 需要同时提供 --sim-log")
        sim = parse_sim_log(args.sim_log)
        rows = merge_sim_steps(policy, sim)
    else:
        rows = to_rows(policy)

    out_path = args.out
    if not out_path:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    suffix = Path(out_path).suffix.lower()
    if suffix == ".parquet":
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out_path, index=False)
    elif suffix == ".csv":
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_path, index=False)
    else:
        Path(out_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
