"""Datasets over the feature cache + a batch sampler that spreads users across a batch
(the dispersion loss needs ≥2 distinct users per batch)."""
from __future__ import annotations

import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler

from .cache import FeatureCache


class PreferenceDataset(Dataset):
    """Items = (user, preferred image, its prompt).  Returns cached features."""

    def __init__(self, manifest: dict, user_ids: list[str], cache: FeatureCache, user_index: dict[str, int],
                 per_user_limit: int | None = None, sample_latent: bool = True, kind: str = "pos"):
        self.cache, self.sample_latent = cache, sample_latent
        self.items: list[tuple[str, str, str]] = []
        for u in user_ids:
            pairs = manifest["users"][u][kind]
            if per_user_limit is not None:
                pairs = pairs[:per_user_limit]
            for x in pairs:
                if cache.has_latent(x["image"]) and cache.has_text(x["prompt"]):
                    self.items.append((u, x["image"], x["prompt"]))
        self.user_index = user_index
        self.by_user: dict[str, list[int]] = defaultdict(list)
        for i, (u, _, _) in enumerate(self.items):
            self.by_user[u].append(i)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        u, img, prompt = self.items[i]
        lat = self.cache.load_latent(img, sample=self.sample_latent)
        t5, pooled = self.cache.load_text(prompt)
        return {"latent": lat, "t5": t5, "pooled": pooled, "user_idx": self.user_index[u], "user": u,
                "image": img, "prompt": prompt}


def collate(batch: list[dict]) -> dict:
    return {
        "latent": torch.stack([b["latent"] for b in batch]),
        "t5": torch.stack([b["t5"] for b in batch]),
        "pooled": torch.stack([b["pooled"] for b in batch]),
        "user_idx": torch.tensor([b["user_idx"] for b in batch], dtype=torch.long),
        "user": [b["user"] for b in batch],
        "image": [b["image"] for b in batch],
        "prompt": [b["prompt"] for b in batch],
    }


class MultiUserBatchSampler(Sampler[list[int]]):
    """Each batch draws its samples from `batch_size` distinct users (users cycle
    uniformly; each user's items cycle in a shuffled order).  Infinite."""

    def __init__(self, dataset: PreferenceDataset, batch_size: int, seed: int = 0):
        self.ds, self.bs = dataset, batch_size
        self.rng = random.Random(seed)
        self.users = sorted(dataset.by_user.keys())
        self._user_queue: list[str] = []
        self._item_queues: dict[str, list[int]] = {}

    def _next_user(self) -> str:
        if not self._user_queue:
            self._user_queue = list(self.users)
            self.rng.shuffle(self._user_queue)
        return self._user_queue.pop()

    def _next_item(self, u: str) -> int:
        q = self._item_queues.get(u)
        if not q:
            q = list(self.ds.by_user[u])
            self.rng.shuffle(q)
            self._item_queues[u] = q
        return q.pop()

    def __iter__(self):
        while True:
            users = []
            while len(users) < self.bs:
                u = self._next_user()
                if u not in users or len(self.users) < self.bs:
                    users.append(u)
            yield [self._next_item(u) for u in users]

    def __len__(self):
        return 10**9
