# MMAR Multi-Modal Action Recognition

A compact and reproducible implementation for multi-modal action recognition on the MMAR dataset (RGB, Depth, Infrared). This repository contains data utilities, training scripts, and multiple inference pipelines including the recommended final pipeline that combines ConvNeXt features, MediaPipe pose, and classical classifiers (SVM / Logistic Regression / Ensemble).

---

## Project structure

```
.
├── data/                   # data loaders and preprocessing utilities
│   ├── dataset.py
│   ├── extract_data.py
│   └── diagnose_features.py
├── model/                  # feature extractors and inference pipelines
│   ├── final_model.py      # recommended final inference pipeline (ConvNeXt + Pose + Ensemble)
│   ├── zero_shot_inference_*.py
│   └── fine_tune_convnext_95plus_final.py
├── train/                  # training scripts
│   ├── train.py
│   ├── train_8gb.py
│   └── train_convnext_large_95plus.py
├── scripts/                # convenience run scripts (optional)
├── checkpoints/            # model checkpoints (not tracked)
├── requirements.txt
└── README.md
```

---

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
# Recommended additional packages (for final pipeline):
pip install mediapipe scikit-learn joblib
```

2. Prepare data: download and unpack the MMAR dataset into a top-level `MMAR/` directory with the following structure:

```
MMAR/
├── train_500/
│   ├── rgb_data/
│   ├── depth_data/
│   ├── ir_data/
│   └── train_videofolder_500.txt
└── test_200/
    ├── rgb_data/
    ├── depth_data/
    ├── ir_data/
    └── test_videofolder_200.txt
```

Notes:
- GPU recommended for training; for limited memory use `train/train_8gb.py`.
- Do not commit large data files or checkpoints; add them to `.gitignore`.

---

## Usage

### Training

Standard (16GB+ GPU):

```bash
python train/train.py \
  --data_root MMAR/train_500 \
  --video_list MMAR/train_500/train_videofolder_500.txt \
  --num_segments 8 \
  --batch_size 4 \
  --epochs 50 \
  --lr 0.001 \
  --base_model resnet18 \
  --fusion_method late \
  --val_ratio 0.2 \
  --save_dir checkpoints \
  --gpu 0
```

8GB GPU (memory friendly):

```bash
python train/train_8gb.py \
  --data_root MMAR/train_500 \
  --video_list MMAR/train_500/train_videofolder_500.txt \
  --batch_size 1 \
  --accumulation_steps 4 \
  --epochs 50 \
  --lr 0.001 \
  --base_model resnet18 \
  --fusion_method late \
  --val_ratio 0.2 \
  --use_amp \
  --num_workers 2 \
  --gpu 0
```

### Final inference / submission

The recommended final inference script is `model/final_model.py` which implements the reported pipeline (ConvNeXt features + optional MediaPipe pose + SVM/Logistic/Ensemble + TTA/PCA).

Example:

```bash
python model/final_model.py \
  --data_root_test MMAR/test_200 \
  --video_list_test MMAR/test_200/test_videofolder_200.txt \
  --backbone_names convnextv2_large,convnext_large \
  --use_pose \
  --classifier ensemble \
  --ensemble_models logistic,svm,rf \
  --tta_times 10 \
  --output submission_final.csv \
  --gpu 0
```

Common args (final pipeline):
- `--backbone_names`: comma-separated backbones (default includes `convnext_large`)
- `--use_pose`: enable MediaPipe Pose (requires `mediapipe` installed)
- `--classifier`: `logistic` | `svm` | `rf` | `gbdt` | `ensemble`
- `--ensemble_models`: `logistic,svm,rf,gbdt`
- `--tta_times`: TTA count (recommended 5-10)
- `--use_pca`: enable PCA on features (`--pca_components` to set components)

---

## Final model summary (what we used)

- Visual backbone: ConvNeXt V2 Large (or ConvNeXt Large fallback) for strong feature extraction.
- Pose features: MediaPipe Pose (33 keypoints × 4 values = 132 dims) to capture fine-grained body motion.
- Fusion: per-modality feature extraction (RGB, Depth, IR) + temporal mean pooling, then concatenate pose features.
- Classifiers: SVM (RBF/linear), Logistic Regression, RandomForest, and weighted ensembles (Voting/Stacking).
- Inference tricks: Test-Time Augmentation (TTA), PCA (optional), and robust scaling before classifiers.

---

## Practical tips

- Keep `MMAR/` and `checkpoints/` out of git (add to `.gitignore`).
- If you need to track large model files, use Git LFS.
- Use `train_8gb.py` and gradient accumulation for memory-constrained setups.
- Use `--tta_times` to reduce variance and improve final submission score.

---

## Action classes (20)

1. switch light (0)
2. up the stairs (1)
3. pack backpack (2)
4. ride a bike (3)
5. turn around (4)
6. fold clothes (5)
7. hug somebody (6)
8. long jump (7)
9. move the chair (8)
10. open the umbrella (9)
11. orchestra conducting (10)
12. rope skipping (11)
13. shake hands (12)
14. squat (13)
15. swivel (14)
16. tie shoes (15)
17. tie hair (16)
18. twist waist (17)
19. wear hat (18)
20. down the stairs (19)

---

## References

- Baseline code: https://github.com/happylinze/multi-modal-tsm
- TSM paper: "Temporal Shift Module for Efficient Video Understanding"

---

COPYRIGHT © 2025, Zhaokai Yin, Peng Hong  
All Rights Reserved.  
Licensed under the Ecplise Public License 2.0.

