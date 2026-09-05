#!/usr/bin/env python3
"""Record one offline-EE evaluation run into the single local SQLite.

Runs LOCALLY (pandas + stdlib sqlite3). Reads a synced run directory produced by
evaluate_ee.py (offline_metrics.json + offline_metrics_by_task.csv) plus the optional
inference stats JSON emitted by batch_inference.py, and upserts one row keyed by
(model_id, checkpoint_id). Per confirmed schema only the predictions file path is kept;
every other path stays out of the table, and the metrics live in results_json.

Usage:
  python evaluation/record_offline_sqlite.py \
    --run-dir /path/to/X3_ckpt-20000_20260905 \
    --db     /path/to/offline_evaluations.sqlite \
    [--stats-json /path/to/X3_ckpt-20000_inference_stats.json] \
    [--flat-csv /path/to/offline_ee_results.csv] \
    [--eval-date 2026-09-05]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

LEAD_CURVE = "lead"
EXEC_CURVE = "execution"
# same metric set as evaluate_ee.py, stored per (stage, curve, step)
METRIC_NAMES = (
    "left_position_cm",
    "right_position_cm",
    "mean_position_cm",
    "left_position_mse_cm2",
    "right_position_mse_cm2",
    "left_rotation_deg",
    "right_rotation_deg",
    "mean_rotation_deg",
    "left_rotation_mse_deg2",
    "right_rotation_mse_deg2",
    "left_gripper_mae",
    "right_gripper_mae",
    "mean_gripper_mae",
    "left_gripper_mse",
    "right_gripper_mse",
)


def build_metrics_node(rows: pd.DataFrame) -> dict:
    """Nest by_task rows into {stage: {curve: {step: {metric: value, comparisons}}}}."""
    node: dict[str, dict] = {}
    for (stage, curve, step), group in rows.groupby(["stage", "curve", "step"], sort=True):
        node.setdefault(stage, {}).setdefault(curve, {})[str(int(step))] = {
            metric: float(group[metric].iloc[0])
            for metric in METRIC_NAMES
        } | {"comparisons": int(group["comparisons"].iloc[0])}
    return node


def build_results_json(
    summary: dict,
    by_task: pd.DataFrame,
    stats: dict | None,
    eval_date: str,
    dataset: str,
    num_predictions: int | None,
) -> dict:
    overall = by_task[by_task["task_index"] == -1]
    per_task = by_task[by_task["task_index"] != -1]
    tasks_node: dict[str, dict] = {}
    for task_index, group in per_task.groupby("task_index", sort=True):
        tasks_node[str(task_index)] = {
            "name": summary.get("task_names", {}).get(str(task_index), ""),
            "num_episodes": int(group["num_episodes"].iloc[0]),
            "metrics": build_metrics_node(group),
        }
    return {
        "eval_date": eval_date,
        "model_id": summary["model_id"],
        "checkpoint_id": summary["checkpoint_id"],
        "action_type": summary.get("action_type", "ee"),
        "action_horizon": int(summary["action_horizon"]),
        "dataset": dataset,
        "validation_episodes": int(summary["validation_episodes"]),
        "num_predictions": int(num_predictions or int(overall.loc[
            (overall["stage"] == "__all__") & (overall["curve"] == LEAD_CURVE) & (overall["step"] == 1),
            "comparisons",
        ].iloc[0])),
        "inference": stats or {},
        "task_names": {k: v for k, v in summary.get("task_names", {}).items()},
        "metrics": {
            "overall": {
                "num_episodes": int(overall["num_episodes"].iloc[0]),
                "metrics": build_metrics_node(overall),
            },
            "tasks": tasks_node,
        },
    }


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS offline_evaluations (
            model_id TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            eval_date TEXT NOT NULL,
            action_horizon INTEGER NOT NULL,
            num_predictions INTEGER NOT NULL,
            predictions_csv TEXT NOT NULL,
            results_json TEXT NOT NULL,
            PRIMARY KEY (model_id, checkpoint_id)
        )"""
    )


def upsert(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO offline_evaluations
           (model_id, checkpoint_id, eval_date, action_horizon,
            num_predictions, predictions_csv, results_json)
           VALUES (:model_id, :checkpoint_id, :eval_date, :action_horizon,
                   :num_predictions, :predictions_csv, :results_json)""",
        row,
    )


def write_flat_csv(db_path: str | Path, out_csv: str | Path | None) -> None:
    """Regenerate the human-readable one-row-per-run table from the DB."""
    if out_csv is None:
        return
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT model_id, checkpoint_id, eval_date, action_horizon, num_predictions,"
            " predictions_csv, results_json FROM offline_evaluations"
        ).fetchall()
    records = []
    for model_id, checkpoint_id, eval_date, horizon, n_pred, pred_csv, results_json in rows:
        r = json.loads(results_json)
        om = r["metrics"]["overall"]["metrics"]
        flat = {
            "run_date": eval_date,
            "model_id": model_id,
            "checkpoint_id": checkpoint_id,
            "action_horizon": horizon,
            "num_predictions": n_pred,
        }
        all_stage = om.get("__all__", {})
        for curve in (LEAD_CURVE, EXEC_CURVE):
            steps = all_stage.get(curve, {}) or {}
            for label in ("1", "5", "10", "20", "30"):
                if label in steps:
                    stem = f"{'lead' if curve == LEAD_CURVE else 'win'}{label}"
                    flat[f"{stem}_pos_cm"] = round(steps[label]["mean_position_cm"], 3)
                    flat[f"{stem}_rot_deg"] = round(steps[label]["mean_rotation_deg"], 3)
                    flat[f"{stem}_grp_mae"] = round(steps[label]["mean_gripper_mae"], 4)
        inf = r.get("inference", {})
        for key in ("batch_size", "n_batches", "total_s", "per_batch_s", "gpu_peak_gb"):
            if key in inf:
                flat[key] = inf[key]
        flat["predictions_csv_server"] = pred_csv
        records.append(flat)
    df = pd.DataFrame(records)
    df = df.reindex(sorted(df.columns), axis=1)
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[record] flat table -> {out_csv} ({len(df)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="synced evaluate_ee.py run directory")
    parser.add_argument("--db", required=True, help="single local SQLite path")
    parser.add_argument("--stats-json", default=None, help="inference stats JSON (batch_inference output)")
    parser.add_argument("--flat-csv", default=None, help="optional flat one-row-per-run CSV to refresh")
    parser.add_argument("--eval-date", default=None, help="YYYY-MM-DD; default = run-dir trailing _YYYYMMDD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "offline_metrics.json").read_text(encoding="utf-8"))
    by_task = pd.read_csv(run_dir / "offline_metrics_by_task.csv")
    stats = None
    if args.stats_json:
        stats = json.loads(Path(args.stats_json).read_text(encoding="utf-8"))

    eval_date = args.eval_date
    if eval_date is None:
        tail = run_dir.name.rsplit("_", 1)[-1]
        eval_date = (
            f"{tail[:4]}-{tail[4:6]}-{tail[6:8]}"
            if len(tail) == 8 and tail.isdigit()
            else date.today().isoformat()
        )
    # dataset identity: only a plain name, not a path (paths stay out of the DB)
    dataset = str(summary.get("baseline_csv", ""))
    dataset = dataset.split("/")[-2] if dataset else ""
    if dataset == "":  # fallback to run dir dataset token if present
        parts = summary.get("predictions_csv", "").split("/")
        dataset = parts[-3] if len(parts) >= 3 else "unknown"

    row_payload = build_results_json(summary, by_task, stats, eval_date, dataset, None)
    row = {
        "model_id": row_payload["model_id"],
        "checkpoint_id": row_payload["checkpoint_id"],
        "eval_date": eval_date,
        "action_horizon": row_payload["action_horizon"],
        "num_predictions": row_payload["num_predictions"],
        "predictions_csv": summary["predictions_csv"],  # the one allowed file path
        "results_json": json.dumps(row_payload, ensure_ascii=False),
    }
    db = Path(args.db).expanduser()
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        ensure_schema(conn)
        upsert(conn, row)
        conn.commit()
    print(f"[record] upserted (model_id={row['model_id']}, checkpoint_id={row['checkpoint_id']}) into {db}")
    write_flat_csv(db, args.flat_csv)


if __name__ == "__main__":
    main()
