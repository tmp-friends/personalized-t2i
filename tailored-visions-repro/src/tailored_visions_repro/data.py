"""PIP (Personalized Image-Prompt) dataset loading.

The released dataset is one JSON-lines file per user, laid out as::

    "train_samples:"
    {... train sample ...}      # 1 .. -4
    ...
    "test_samples:"
    {... test sample ...}       # -2
    {... test sample ...}       # -1

Train samples carry ``prompt`` (the full prompt the user actually typed).
Test samples additionally carry ``query`` -- the ChatGPT-shortened version of
``prompt`` that the paper feeds to the rewriter. ``prompt`` is the ground truth
for ROUGE-L.

``result_url`` points at a CDN that has been offline since 2024, so the
ground-truth images referenced by the paper's Image-Align metric are gone.
See ``docs/DEVIATIONS.md``.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

TRAIN_MARKER = "train_samples:"
TEST_MARKER = "test_samples:"


@dataclass
class TestSample:
    """One evaluation instance: a shortened query + its ground-truth prompt."""

    user_id: str
    sample_id: str
    query: str  # shortened prompt -- the rewriter's input
    prompt: str  # original full prompt -- ROUGE-L reference
    result_url: str = ""
    nsfw: str = ""
    resolution_power: str = ""
    proportion: str = ""

    @property
    def key(self) -> str:
        return f"{self.user_id}/{self.sample_id}"


@dataclass
class User:
    """A user's full record: history prompts plus the two held-out test samples."""

    user_id: str
    history: list[dict] = field(default_factory=list)
    test_samples: list[TestSample] = field(default_factory=list)

    @property
    def history_prompts(self) -> list[str]:
        return [h["prompt"] for h in self.history]


def read_jsonl(path: str | os.PathLike) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(rows: Iterable[dict], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_user(path: str | os.PathLike) -> User:
    """Parse one user's jsonl file into a :class:`User`."""
    path = Path(path)
    lines = read_jsonl(path)
    if not (isinstance(lines[0], str) and lines[0].startswith(TRAIN_MARKER.rstrip(":"))):
        raise ValueError(f"{path}: expected '{TRAIN_MARKER}' on the first line")
    if not (isinstance(lines[-3], str) and lines[-3].startswith(TEST_MARKER.rstrip(":"))):
        raise ValueError(f"{path}: expected '{TEST_MARKER}' at line -3")

    user_id = path.stem
    history = lines[1:-3]
    tests = []
    for row in lines[-2:]:
        tests.append(
            TestSample(
                user_id=user_id,
                sample_id=str(row.get("id", "")),
                query=row["query"],
                prompt=row["prompt"],
                result_url=row.get("result_url", ""),
                nsfw=row.get("NSFW", ""),
                resolution_power=row.get("resolution_power", ""),
                proportion=row.get("proportion", ""),
            )
        )
    return User(user_id=user_id, history=history, test_samples=tests)


def iter_users(data_dir: str | os.PathLike) -> Iterator[User]:
    """Yield every user in ``data_dir`` in sorted-by-filename order."""
    for path in sorted(Path(data_dir).glob("*.jsonl")):
        yield load_user(path)


def load_users(
    data_dir: str | os.PathLike,
    user_ids: Sequence[str] | None = None,
) -> list[User]:
    if user_ids is None:
        return list(iter_users(data_dir))
    data_dir = Path(data_dir)
    return [load_user(data_dir / f"{uid}.jsonl") for uid in user_ids]


def list_user_ids(data_dir: str | os.PathLike) -> list[str]:
    return sorted(p.stem for p in Path(data_dir).glob("*.jsonl"))


def sample_user_ids(
    data_dir: str | os.PathLike,
    n: int,
    seed: int = 0,
) -> list[str]:
    """Deterministically sample ``n`` user ids.

    Used to build the fixed subset the image-based metrics run on, so that every
    method is scored on exactly the same users.
    """
    ids = list_user_ids(data_dir)
    if n >= len(ids):
        return ids
    rng = random.Random(seed)
    return sorted(rng.sample(ids, n))


def dataset_stats(data_dir: str | os.PathLike) -> dict:
    n_users = n_train = n_test = 0
    for user in iter_users(data_dir):
        n_users += 1
        n_train += len(user.history)
        n_test += len(user.test_samples)
    return {
        "users": n_users,
        "train_samples": n_train,
        "test_samples": n_test,
        "total": n_train + n_test,
    }
