from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import torch


SCHEMA_VERSION = 1


def sample_key(episode_index: int, frame_index: int) -> str:
    return f"{int(episode_index)}:{int(frame_index)}"


class FeatureCacheWriter:
    """Append-only SQLite cache. BF16 is stored losslessly as raw uint16 bytes."""

    def __init__(self, path: str | Path, metadata: dict, overwrite: bool = False):
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"cache exists: {self.path}; pass --overwrite intentionally")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.conn.execute(
            "CREATE TABLE features (sample_key TEXT PRIMARY KEY, episode INTEGER NOT NULL, "
            "frame INTEGER NOT NULL, is_key INTEGER NOT NULL, shape TEXT NOT NULL, data BLOB NOT NULL)"
        )
        metadata = {**metadata, "schema_version": SCHEMA_VERSION, "dtype": "bfloat16"}
        self.conn.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [(k, json.dumps(v, sort_keys=True)) for k, v in metadata.items()],
        )
        self.pending = 0

    def add(self, episode: int, frame: int, is_key: bool, feature: torch.Tensor):
        value = feature.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        # int16 is a bit container here (not a numeric conversion). Older torch
        # releases do not expose torch.uint16, while BF16 and int16 are both 16 bit.
        raw = value.view(torch.int16).numpy().tobytes()
        self.conn.execute(
            "INSERT INTO features VALUES (?, ?, ?, ?, ?, ?)",
            (sample_key(episode, frame), int(episode), int(frame), int(is_key),
             json.dumps(list(value.shape)), sqlite3.Binary(raw)),
        )
        self.pending += 1
        if self.pending >= 64:
            self.conn.commit()
            self.pending = 0

    def close(self):
        self.conn.commit()
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_features_keyflag ON features(is_key)")
        self.conn.commit()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.conn.rollback()
            self.conn.close()


class FeatureCacheReader:
    def __init__(self, path: str | Path):
        self.path = str(Path(path).resolve())
        self._conn = None
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as conn:
            self.metadata = {
                key: json.loads(value)
                for key, value in conn.execute("SELECT key, value FROM metadata")
            }
            self.entries = {
                sample_key(ep, frame): bool(is_key)
                for ep, frame, is_key in conn.execute(
                    "SELECT episode, frame, is_key FROM features"
                )
            }
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported cache schema: {self.metadata}")

    def _connection(self):
        # DataLoader workers must not inherit a live sqlite connection.
        if self._conn is None:
            self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        return self._conn

    @property
    def allowlist(self):
        return {
            tuple(map(int, key.split(":")))
            for key in self.entries
        }

    def get(self, episode: int, frame: int) -> torch.Tensor:
        key = sample_key(episode, frame)
        row = self._connection().execute(
            "SELECT shape, data FROM features WHERE sample_key=?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(f"teacher cache miss: {key}")
        shape, blob = json.loads(row[0]), row[1]
        # Copy detaches from sqlite's immutable bytes buffer and makes Tensor writable.
        array = np.frombuffer(blob, dtype=np.int16).copy().reshape(shape)
        return torch.from_numpy(array).view(torch.bfloat16)


def inspect_cache(path: str | Path) -> dict:
    reader = FeatureCacheReader(path)
    key_count = sum(reader.entries.values())
    first_key = next(iter(reader.entries), None)
    first = reader.get(*map(int, first_key.split(":"))) if first_key else None
    return {
        "path": reader.path,
        "samples": len(reader.entries),
        "key_samples": key_count,
        "regular_samples": len(reader.entries) - key_count,
        "feature_shape": list(first.shape) if first is not None else None,
        "finite_ratio": float(torch.isfinite(first.float()).float().mean()) if first is not None else None,
        "metadata": reader.metadata,
    }
