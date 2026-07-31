"""Deterministic episode-level split for production-enterprise expert data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERT_TRAIN_FRACTION = 0.80
EXPERT_VALIDATION_FRACTION = 0.10
EXPERT_SPLIT_SEED = 20260729
EPISODE_START_VALUE = 0.1
EPISODE_START_ATOL = 1e-5


@dataclass(frozen=True)
class ExpertDataSplit:
    train_values: np.ndarray
    validation_values: np.ndarray
    test_values: np.ndarray
    train_source_rows: np.ndarray
    validation_source_rows: np.ndarray
    test_source_rows: np.ndarray
    train_episode_labels: np.ndarray
    validation_episode_labels: np.ndarray
    test_episode_labels: np.ndarray
    metadata: dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_expert_episode_split(
    csv_path: Path,
    *,
    train_fraction: float = EXPERT_TRAIN_FRACTION,
    validation_fraction: float = EXPERT_VALIDATION_FRACTION,
    split_seed: int = EXPERT_SPLIT_SEED,
) -> ExpertDataSplit:
    """Split the 37-column expert file by complete episodes.

    The data-processing code identifies an episode start when the first state
    field equals 0.1.  This function preserves each resulting episode wholly
    within exactly one of the training, validation, or final-test partitions.
    """
    csv_path = Path(csv_path)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")
    raw = pd.read_csv(csv_path, header=None)
    if raw.shape[1] < 37:
        raise ValueError(f"Expected at least 37 columns in {csv_path}, found {raw.shape[1]}.")

    numeric = raw.iloc[:, :37].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(values).all(axis=1)
    first_state = values[:, 0]
    episode_start = np.isfinite(first_state) & np.isclose(
        first_state, EPISODE_START_VALUE, atol=EPISODE_START_ATOL, rtol=0.0
    )
    if len(episode_start) == 0 or not episode_start[0]:
        raise ValueError(
            "The first expert row is not an episode start; episode-level splitting is unsafe."
        )
    episode_ids = np.cumsum(episode_start).astype(np.int64)
    unique_episodes = np.unique(episode_ids[valid_mask])
    if unique_episodes.size < 3:
        raise ValueError("At least three complete expert episodes are required for splitting.")

    rng = np.random.default_rng(int(split_seed))
    shuffled = unique_episodes.copy()
    rng.shuffle(shuffled)
    train_episode_count = int(round(unique_episodes.size * train_fraction))
    validation_episode_count = int(round(unique_episodes.size * validation_fraction))
    train_episode_count = min(max(train_episode_count, 1), unique_episodes.size - 2)
    validation_episode_count = min(
        max(validation_episode_count, 1), unique_episodes.size - train_episode_count - 1
    )
    train_episode_ids = np.sort(shuffled[:train_episode_count])
    validation_end = train_episode_count + validation_episode_count
    validation_episode_ids = np.sort(shuffled[train_episode_count:validation_end])
    test_episode_ids = np.sort(shuffled[validation_end:])
    train_mask = valid_mask & np.isin(episode_ids, train_episode_ids)
    validation_mask = valid_mask & np.isin(episode_ids, validation_episode_ids)
    test_mask = valid_mask & np.isin(episode_ids, test_episode_ids)
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise RuntimeError("Episode split produced an empty partition.")

    metadata: dict[str, Any] = {
        "expert_csv": str(csv_path.resolve()),
        "expert_csv_sha256": file_sha256(csv_path),
        "split_unit": "episode",
        "episode_start_rule": "state_0 equals 0.1 within absolute tolerance 1e-5",
        "split_seed": int(split_seed),
        "train_fraction_requested": float(train_fraction),
        "validation_fraction_requested": float(validation_fraction),
        "test_fraction_requested": float(1.0 - train_fraction - validation_fraction),
        "total_rows": int(len(raw)),
        "valid_rows": int(valid_mask.sum()),
        "dropped_nonfinite_rows": int((~valid_mask).sum()),
        "total_episodes": int(unique_episodes.size),
        "train_episodes": int(train_episode_ids.size),
        "validation_episodes": int(validation_episode_ids.size),
        "test_episodes": int(test_episode_ids.size),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "validation_episode_ids": [int(value) for value in validation_episode_ids],
        "test_episode_ids": [int(value) for value in test_episode_ids],
    }
    return ExpertDataSplit(
        train_values=values[train_mask].astype(np.float32),
        validation_values=values[validation_mask].astype(np.float32),
        test_values=values[test_mask].astype(np.float32),
        train_source_rows=np.flatnonzero(train_mask).astype(np.int64),
        validation_source_rows=np.flatnonzero(validation_mask).astype(np.int64),
        test_source_rows=np.flatnonzero(test_mask).astype(np.int64),
        train_episode_labels=episode_ids[train_mask].astype(np.int64),
        validation_episode_labels=episode_ids[validation_mask].astype(np.int64),
        test_episode_labels=episode_ids[test_mask].astype(np.int64),
        metadata=metadata,
    )
