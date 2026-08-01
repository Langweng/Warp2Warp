# WarpHomo

This is the cleaned GitHub-ready version of the WarpHomo project. It keeps the core IHN homography-estimation pipeline:

- self-supervised IDR training
- supervised training
- checkpoint evaluation/testing
- GoogleMap-style manifest generation

Large datasets, logs, checkpoints, subjective visualization assets, efficiency analysis, generalization-test scripts, PRD notes, and external baseline projects are intentionally not included.

## Project Layout

```text
WarpHomo_github/
├── configs/experiments/          # Core IHN train/test configs
├── data/
│   ├── manifests/                # Manifest generator and local JSON manifests
│   └── raw/                      # Local datasets, ignored by git
├── datasets/                     # Warp-Fix and Warp2-Fix dataset loaders
├── engine/                       # Training step logic
├── evaluation/                   # Evaluation and metric reporting
├── models/                       # IHN model and model utilities
├── scripts/                      # Lightweight train/test launchers
├── training/                     # Train/test entry points and loops
└── utils/                        # Homography, IO, metrics, seed, visualization helpers
```

## Install

```bash
cd WarpHomo_github
python3 -m pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the `torch` and `torchvision` wheels that match your CUDA version before installing the rest of the requirements.

## Prepare Data

Put the GoogleMap data under:

```text
data/raw/GoogleMap/
├── train_fixA/
├── train_fixB/
├── val_fixA/
└── val_fixB/
```

Generate the manifest:

```bash
python3 data/manifests/gen_json_googlemap.py \
  --data_dir data/raw/GoogleMap \
  --output_file data/manifests/googlemap.json
```

The generated `*.json` manifest and raw images are ignored by git.

## Train

Self-supervised IDR training:

```bash
python3 training/train.py \
  --config configs/experiments/ihn_idr_googlemap_320_to_256.yaml \
  --device cuda:0
```

Or use the launcher:

```bash
DEVICE=cuda:0 scripts/train_googlemap_ihn_idr.sh
```

Supervised training:

```bash
python3 training/train.py \
  --config configs/experiments/ihn_supervised_googlemap_320_to_256_mapped.yaml \
  --device cuda:0
```

## Test

Evaluate a checkpoint:

```bash
python3 training/test.py \
  --config configs/experiments/ihn_test_googlemap_320_to_256_eval128.yaml \
  --checkpoint logs/IHN_IDR_GoogleMap_320to256_l2/checkpoint/best_model.pth \
  --device cuda:0
```

For deterministic evaluation labels:

```bash
python3 training/test.py \
  --config configs/experiments/ihn_test_googlemap_320_to_256_eval128.yaml \
  --checkpoint logs/IHN_IDR_GoogleMap_320to256_l2/checkpoint/best_model.pth \
  --generate-fixed-labels logs/IHN_Test_GoogleMap_320to256_Eval128/eval/fixed_labels.json \
  --device cuda:0
```

## Notes

This repository version focuses on the main IHN pipeline. External baselines such as RHWF, LocalTrans, and MHN, plus generalization and efficiency analysis scripts, were left out to keep the upload small and easy to review.
