# DuoCARE-CD

Architectural change detection at World Cultural Heritage sites with two complementary detectors,
trained independently and combined by probability averaging:

- **Correspondence-aware detector** (`correspondence_aware/`): location-wise change evidence is verified by
  image-wide cross-temporal appearance matching and then propagated among locations with compatible appearance.
- **Registration-equivariant detector** (`registration_equivariant/`): feature differences are additionally
  computed after aligning the two images with a registration shift estimated from the imagery
  (phase correlation; unreliable pairs use the median shift of reliable pairs at the same site).

Both use a frozen DINOv2 ViT-L/14 encoder with LoRA adapters on the query-key-value projections.

## Results on the WCH-CD test set

Pooled over all test pixels, threshold 0.5, no post-processing (`python evaluate.py`):

| Model | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| Correspondence-aware detector | 78.75 | 77.27 | 78.01 | 63.94 |
| Registration-equivariant detector | 82.95 | 72.42 | 77.33 | 63.04 |
| **DuoCARE-CD (ensemble)** | **82.08** | **76.18** | **79.02** | **65.32** |

## Layout

```
common/                    shared encoder, DINOv2 baseline, data pipeline, training loop, metrics
correspondence_aware/      model.py, objective.py, train.py, config.json
registration_equivariant/  model.py, objective.py, data.py, train.py, config.json,
                           estimate_shifts.py, shifts/shift_table.json (shifts used in the paper)
checkpoints/               trained weights of the two detectors
predict.py                 ensemble inference for one image pair
evaluate.py                reproduces the table above
```

## Requirements

Python 3.10+, PyTorch 2.x (tested with 2.10 + CUDA 12.8), torchvision, numpy, scipy, Pillow.
Download the official DINOv2 ViT-L/14 weights (without registers),
[`dinov2_vitl14_pretrain.pth`](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth).
The checkpoints in `checkpoints/` store only the trainable parameters (LoRA and decoders) and require these weights.

## Data

```
<data_root>/{train,val,test}/{T1,T2,GT}/<site>_<index>.png
```
256 × 256 tiles; GT values 0 (unchanged), 1 (changed), 255 (ignored).
The site is the part of the file name before the first `_`; training batches draw 2 sites × 4 tiles,
and train/val/test must not share sites.

## Usage

All scripts are run directly (e.g. from VSCode); paths are set in the `__main__` block at the end of each file.

- **Evaluate:** `python evaluate.py`
- **Predict one pair:** `python predict.py` (set the T1/T2 paths and, if known, the shift `(dx, dy)`;
  with `(0, 0)` the registration-equivariant detector falls back to unaligned comparison).
- **Estimate shifts for new data:** `python registration_equivariant/estimate_shifts.py`
- **Train:** `python correspondence_aware/train.py` and `python registration_equivariant/train.py`.
  Protocol: 20,000 updates, AdamW (lr 1e-4, weight decay 0.05), lr halved after 5,000 and 10,000 updates,
  FP32, validation every 1,000 updates; the checkpoint with the best validation IoU at 0.5 is kept as `best.pth`
  and evaluated on the test set at the end. Set `preflight = 2` to run a two-batch forward/backward check only.
  Each detector peaks at about 20 GB of GPU memory (batch size 8); training took 7–9 h on one RTX 5090.
