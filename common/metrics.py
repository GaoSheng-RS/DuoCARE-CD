"""Streaming image-only evaluation; no optimizer, support images or labels in forward.

All decisions use ``foreground_probability >= threshold``. Threshold counts are
exact for the evaluated float32 probabilities. AP uses 4096 probability bins by
default and is explicitly an approximation, not exact pixel average precision.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor, nn

from .data.site_batch import site_of


DEFAULT_THRESHOLDS = tuple(index / 100 for index in range(5, 96))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def counts_to_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Undefined ratios are JSON-safe None, including empty foreground union."""
    tp, fp, fn, tn = (int(value) for value in (tp, fp, fn, tn))
    if min(tp, fp, fn, tn) < 0:
        raise ValueError("Confusion counts must be nonnegative")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "valid_pixels": tp + fp + fn + tn,
        "positive_pixels": tp + fn,
        "predicted_positive_pixels": tp + fp,
        "iou": _ratio(tp, tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Thresholds must be finite numbers in [0, 1]")
    return value


class StreamingMetrics:
    """Keep O(number of sites * thresholds + AP bins) counters, not pixels."""

    def __init__(
        self,
        *,
        thresholds: Sequence[float] | None = None,
        selected_threshold: float | None = None,
        ap_bins: int = 4096,
    ):
        if not isinstance(ap_bins, int) or isinstance(ap_bins, bool) or ap_bins < 2:
            raise ValueError("ap_bins must be an integer >= 2")
        candidates = DEFAULT_THRESHOLDS if thresholds is None else thresholds
        self.candidates = tuple(sorted({_threshold(value) for value in candidates}))
        self.selected_threshold = None if selected_threshold is None else _threshold(selected_threshold)
        all_thresholds = {*self.candidates, 0.5}
        if self.selected_threshold is not None:
            all_thresholds.add(self.selected_threshold)
        self.thresholds = tuple(sorted(all_thresholds))
        self.boundaries = torch.tensor(self.thresholds, dtype=torch.float32)
        self.ap_bins = ap_bins
        self.positive_hist = torch.zeros(ap_bins, dtype=torch.int64)
        self.negative_hist = torch.zeros(ap_bins, dtype=torch.int64)
        self.pooled = torch.zeros((2, len(self.thresholds) + 1), dtype=torch.int64)
        self.sites: dict[str, Tensor] = {}
        self.site_samples: dict[str, int] = {}
        self.samples = 0
        self.ignored_pixels = 0

    def update(self, probabilities: Tensor, labels: Tensor, names: Sequence[str]) -> None:
        """Consume one batch [B,H,W]; row 0 counts background, row 1 foreground."""
        if probabilities.ndim != 3 or labels.shape != probabilities.shape:
            raise ValueError("Probabilities and labels must have the same [B,H,W] shape")
        if labels.dtype != torch.long or len(names) != probabilities.shape[0]:
            raise ValueError("Labels must be torch.long and names must match the batch")
        scores = probabilities.detach().to(device="cpu", dtype=torch.float32)
        targets = labels.detach().to(device="cpu")
        if not bool(torch.isfinite(scores).all()) or bool(((scores < 0) | (scores > 1)).any()):
            raise ValueError("Probabilities must be finite and in [0,1]")
        if not bool(((targets == 0) | (targets == 1) | (targets == 255)).all()):
            raise ValueError("Labels must contain only 0, 1 and ignore=255")
        for probability, target, name in zip(scores, targets, names):
            site = site_of(name)
            self.samples += 1
            self.site_samples[site] = self.site_samples.get(site, 0) + 1
            site_hist = self.sites.setdefault(site, torch.zeros_like(self.pooled))
            valid = target != 255
            self.ignored_pixels += int((~valid).sum())
            probability, target = probability[valid], target[valid]
            # right=True assigns p == threshold to the positive side (>=).
            threshold_bin = torch.bucketize(probability, self.boundaries, right=True)
            ap_bin = (probability * self.ap_bins).long().clamp(max=self.ap_bins - 1)
            for cls, ap_hist in ((0, self.negative_hist), (1, self.positive_hist)):
                chosen = target == cls
                count = torch.bincount(threshold_bin[chosen], minlength=self.pooled.shape[1])
                self.pooled[cls] += count
                site_hist[cls] += count
                ap_hist += torch.bincount(ap_bin[chosen], minlength=self.ap_bins)

    def _at(self, histogram: Tensor, threshold: float) -> dict:
        index = self.thresholds.index(threshold) + 1
        negatives, positives = int(histogram[0].sum()), int(histogram[1].sum())
        fp, tp = int(histogram[0, index:].sum()), int(histogram[1, index:].sum())
        return counts_to_metrics(tp, fp, positives - tp, negatives - fp)

    def _per_site(self, threshold: float) -> dict:
        return {
            name: {"samples": self.site_samples[name], **self._at(histogram, threshold)}
            for name, histogram in sorted(self.sites.items())
        }

    @staticmethod
    def _site_mean(per_site: dict) -> float | None:
        values = [record["iou"] for record in per_site.values() if record["iou"] is not None]
        return sum(values) / len(values) if values else None

    def compute(self) -> dict:
        per_site = self._per_site(0.5)
        positives = self.positive_hist.flip(0).to(torch.float64)
        negatives = self.negative_hist.flip(0).to(torch.float64)
        total_positive = int(positives.sum())
        tp, fp = positives.cumsum(0), negatives.cumsum(0)
        precision = tp / (tp + fp).clamp_min(1)
        ap = float((precision * positives).sum() / total_positive) if total_positive else None
        result = {
            "samples": self.samples,
            "valid_pixels": int(self.pooled.sum()),
            "ignored_pixels": self.ignored_pixels,
            "site_count": len(self.sites),
            "decision_rule": "foreground_probability >= threshold",
            "undefined_metric_policy": "None when its denominator is zero; exclude undefined site IoU from site mean",
            "pooled_at_0_5": self._at(self.pooled, 0.5),
            "per_site_at_0_5": per_site,
            "site_mean_iou_at_0_5": self._site_mean(per_site),
            "binned_ap": {"value": ap, "bins": self.ap_bins, "exact": False,
                          "bin_rule": "floor(probability * bins), clamp p=1 to bins-1; tied scores within each bin"},
            "threshold_search": None,
        }
        if self.candidates:
            records = [{"threshold": value, **self._at(self.pooled, value)} for value in self.candidates]
            eligible = [record for record in records if record["iou"] is not None]
            best = max(eligible, key=lambda record: (record["iou"], -record["threshold"])) if eligible else None
            result["threshold_search"] = {
                "candidates": records,
                "selection_metric": "pooled_iou",
                "tie_break": "smaller threshold",
                "best_threshold": best["threshold"] if best else None,
                "best_metrics": {key: value for key, value in best.items() if key != "threshold"} if best else None,
            }
        if self.selected_threshold is not None:
            selected_sites = self._per_site(self.selected_threshold)
            result.update(
                selected_threshold=self.selected_threshold,
                pooled_at_selected_threshold=self._at(self.pooled, self.selected_threshold),
                per_site_at_selected_threshold=selected_sites,
                site_mean_iou_at_selected_threshold=self._site_mean(selected_sites),
            )
        return result


@contextmanager
def _preserve_evaluation_state(model: nn.Module, loader):
    """Restore modes and global RNGs even if the loader/model/evaluation fails."""
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    # Do not initialize CUDA just to evaluate/test a CPU model.
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    loader_generator = getattr(loader, "generator", None)
    loader_state = loader_generator.get_state() if isinstance(loader_generator, torch.Generator) else None
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if loader_state is not None:
            loader_generator.set_state(loader_state)


def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: str | torch.device,
    *,
    thresholds: Sequence[float] | None = None,
    selected_threshold: float | None = None,
    ap_bins: int = 4096,
) -> dict:
    """Evaluate only model(T1,T2), keeping no pixel predictions after each batch.

    Default thresholds are 0.05..0.95 in steps of 0.01. Use ``thresholds=()`` on
    test and pass the previously frozen val ``selected_threshold``; this disables
    threshold selection on test. All metric values are proportions in [0,1].
    The caller owns model placement and supplies deterministic eval preprocessing.
    """
    accumulator = StreamingMetrics(thresholds=thresholds, selected_threshold=selected_threshold, ap_bins=ap_bins)
    device = torch.device(device)
    with _preserve_evaluation_state(model, loader), torch.inference_mode():
        for batch in loader:
            if len(batch) != 4:
                raise ValueError("Evaluation batches must contain T1,T2,labels,names")
            x1, x2, labels, names = batch
            logits = model(x1.to(device, non_blocking=True), x2.to(device, non_blocking=True))
            if not isinstance(logits, Tensor) or logits.ndim != 4 or logits.shape[1] != 2:
                raise ValueError("Model must return two-class logits [B,2,H,W]")
            if logits.shape[0] != labels.shape[0] or logits.shape[-2:] != labels.shape[-2:]:
                raise ValueError("Prediction and GT spatial/batch shapes must match")
            probabilities = logits.float().softmax(dim=1)[:, 1]
            accumulator.update(probabilities, labels, names)
    return accumulator.compute()
