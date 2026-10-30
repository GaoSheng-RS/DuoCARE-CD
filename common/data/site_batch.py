"""CPU-side site batches using the unchanged ``dataCD.py`` augmentation pipeline.

No files are read at import time. Only ``make_train_loader`` constructs a real
dataset; the sampler itself inspects ``dataset.samples`` without loading images.
"""

from __future__ import annotations

import math
import operator
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


def site_of(name: str) -> str:
    """Use the existing filename convention: prefix before the first underscore."""
    if not isinstance(name, str) or not name or not name.split("_", 1)[0]:
        raise ValueError("Every sample must have a nonempty string name and site prefix.")
    return name.split("_", 1)[0]


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


class SiteBatchSampler(Sampler[list[int]]):
    """Draw ``M`` distinct sites and ``K`` tiles per site, with grouped indices.

    Site-selection weights are ``N_s ** site_power`` (default square root).
    Sampling within a site uses replacement only when it contains fewer than K
    samples. The default epoch contains ``ceil(len(dataset.samples)/(M*K))``
    batches; weighted sampling does not promise to visit every tile once.

    Repeating iteration with the same seed and epoch reproduces sample indices.
    """

    def __init__(
        self,
        dataset: Any,
        M: int = 2,
        K: int = 4,
        site_power: float = 0.5,
        steps: int | None = None,
        seed: int = 0,
    ) -> None:
        self.M = _integer(M, "M", 1)
        self.K = _integer(K, "K", 1)
        self.seed = _integer(seed, "seed", 0)
        self.epoch = 0
        self.site_power = float(site_power)
        if not math.isfinite(self.site_power):
            raise ValueError("site_power must be finite.")
        if not hasattr(dataset, "samples"):
            raise TypeError("dataset must expose samples in dataset-index order.")
        self.names = tuple(sample["name"] for sample in dataset.samples)
        if not self.names:
            raise ValueError("Cannot sample an empty dataset.")
        members: dict[str, list[int]] = defaultdict(list)
        for index, name in enumerate(self.names):
            members[site_of(name)].append(index)
        self.site_ids = tuple(sorted(members))
        if len(self.site_ids) < self.M:
            raise ValueError(
                f"Need {self.M} distinct sites per batch; dataset has {len(self.site_ids)}."
            )
        self.members = tuple(np.asarray(members[site], dtype=np.int64) for site in self.site_ids)
        self.sizes = np.asarray([len(indices) for indices in self.members], dtype=np.int64)
        weights = self.sizes.astype(np.float64) ** self.site_power
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("site_power produces invalid site-selection weights.")
        self.probabilities = weights / weights.sum()
        self.steps = (
            math.ceil(len(self.names) / (self.M * self.K))
            if steps is None
            else _integer(steps, "steps", 1)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _integer(epoch, "epoch", 0)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        for _ in range(self.steps):
            chosen = rng.choice(len(self.site_ids), self.M, replace=False, p=self.probabilities)
            batch: list[int] = []
            for site_index in chosen:
                indices = self.members[int(site_index)]
                selected = rng.choice(indices, self.K, replace=len(indices) < self.K)
                batch.extend(int(index) for index in selected)
            yield batch


def same_site_reference_indices(names: Sequence[str]) -> list[list[int]]:
    """Return batch indices of distinct other tiles in each query's own site.

    Repeated names represent augmentations of one tile. Exclude every occurrence
    of the query name and retain only the first occurrence of each other name.
    A singleton site receives an empty list; no other-site fallback is used.
    """
    sites = [site_of(name) for name in names]
    references: list[list[int]] = []
    for query_index, query_name in enumerate(names):
        seen: set[str] = set()
        selected: list[int] = []
        for index, name in enumerate(names):
            if sites[index] != sites[query_index] or name == query_name or name in seen:
                continue
            selected.append(index)
            seen.add(name)
        references.append(selected)
    return references


def _seed_cpu_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # torch.manual_seed also seeds accelerator RNGs; seed only the CPU generator.
    torch.random.default_generator.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seed Python, NumPy and CPU Torch from DataLoader's assigned worker seed."""
    del worker_id
    _seed_cpu_rngs(torch.initial_seed())


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """Reset sampling and augmentation RNGs before creating an epoch iterator.

    The same seed, epoch, worker count and iteration order reproduce this data
    pipeline. With zero workers this deliberately reseeds process-global CPU
    RNGs because the unchanged transforms use Python and Torch global RNGs.
    Call only between iterators; persistent workers are intentionally disabled.
    """
    sampler = loader.batch_sampler
    if not isinstance(sampler, SiteBatchSampler) or loader.generator is None:
        raise TypeError("Expected a loader created by make_train_loader.")
    sampler.set_epoch(epoch)
    epoch_seed = int(np.random.SeedSequence([sampler.seed, sampler.epoch]).generate_state(1, dtype=np.uint64)[0])
    loader.generator.manual_seed(epoch_seed)
    if loader.num_workers == 0:
        _seed_cpu_rngs(epoch_seed)


def make_train_loader(
    dataset_path: str,
    *,
    M: int = 2,
    K: int = 4,
    site_power: float = 0.5,
    steps: int | None = None,
    seed: int = 0,
    epoch: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build the existing train dataset and attach site-aware batch sampling.

    This is an explicit construction helper, never called on module import. It
    preserves the existing image/label loading, all augmentation and return
    values, including sample names. It neither moves data to an accelerator nor
    starts model training. Call ``set_loader_epoch(loader, epoch)`` each epoch.
    """
    from .dataCD import build_dataCD

    num_workers = _integer(num_workers, "num_workers", 0)
    dataset = build_dataCD(dataset_path=dataset_path, mode="train", return_name=True)
    sampler = SiteBatchSampler(dataset, M=M, K=K, site_power=site_power, steps=steps, seed=seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator(device="cpu"),
    )
    set_loader_epoch(loader, epoch)
    return loader
