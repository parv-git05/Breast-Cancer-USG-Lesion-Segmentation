# Breast Ultrasound AI Pipeline (BUSI Phase 1)

Reproducible medical computer-vision research pipeline for B-mode breast ultrasound (BUS) lesion segmentation and malignancy classification using the Breast Ultrasound Images (BUSI) dataset.

> **Disclaimer**: This is a research and educational codebase. It is **NOT** a medical diagnostic device and must not be used to make or infer clinical diagnoses.

---

## 1. Dataset State & Quality Contracts

- **Raw Dataset Root**: `D:\Dataset_BUSI_with_GT\` (Strictly read-only; never modified, overwritten, or deleted).
- **Subfolders**: `benign/`, `malignant/`, `normal/`.
- **Project Workspace**: `D:\Dataset_BUSI_with_GT\BUSI_AI_Project\` (Writable repository root for scripts, configs, and outputs).
- **Phase 1 Single-Mask Clean Set**: Exactly **628 images** (420 benign, 208 malignant), each with exactly one ground-truth mask.
- **Held-out Multi-Mask Cases**: 17 cases recorded in `data/manifests/BUSI_phase1_excluded.csv` (Held out for Phase 2 curation; never used in Phase 1).
- **Held-out Normal Cases**: 133 cases (Empty mask annotations; held out for Phase 2).
- **Duplicate Pair Audit**: Contradictory duplicate pair `benign (433).png` / `malignant (145).png` is confirmed deleted and absent.

### ⚠️ Patient-ID Splitting Caveat
> **Important Limitation**: Patient IDs are **not** present in the local copy of the BUSI dataset. Consequently, all train/val/test splits in Phase 1 are **image-level stratified** by lesion class (`benign` / `malignant`), not patient-wise. If verified patient-ID mappings become available in future phases, the pipeline will be updated to patient-wise splitting.

---

## 2. Directory Layout

```
D:\Dataset_BUSI_with_GT\
├── benign\                                 [RAW DATA - IMMUTABLE]
├── malignant\                              [RAW DATA - IMMUTABLE]
├── normal\                                 [RAW DATA - IMMUTABLE]
└── BUSI_AI_Project\                        [PROJECT WORKSPACE]
    ├── configs\
    │   ├── split.yaml
    │   └── exp_seg_01.yaml
    ├── data\
    │   ├── manifests\                      (BUSI_phase1_manifest.csv, BUSI_phase1_excluded.csv)
    │   ├── splits\                         (train.csv, val.csv, test.csv, split_metadata.json)
    │   └── processed\
    ├── src\
    │   ├── audit.py
    │   ├── manifest.py
    │   ├── split.py
    │   ├── transforms.py
    │   ├── dataset.py
    │   ├── losses.py
    │   ├── metrics.py
    │   ├── visualize.py
    │   ├── utils.py
    │   └── models\
    │       └── unet.py
    ├── tests\
    │   ├── test_data.py
    │   └── test_metrics.py
    ├── outputs\
    │   ├── checkpoints\
    │   ├── logs\
    │   └── reports\
    ├── requirements.txt
    ├── README.md
    └── .gitignore
```

---

## 3. Quality Gates

| Gate | Stage | Criteria |
| :--- | :--- | :--- |
| **GATE 1** | Raw Data & Manifest | Manifest rows = 628 (420 benign, 208 malignant), all files exist, zero overlap with excluded list, no empty masks. |
| **GATE 2** | Stratified Split | Splits sum to 628, mutually disjoint (0 overlap), class balance preserved within ±2%, metadata JSON written. |
| **GATE 3** | Preprocessing & DataLoader | Batch shape `(B, 1, 256, 256)`, mask values strictly in `{0, 1}`, no NaNs/Infs, deterministic eval transforms, synchronous paired augmentations. |
| **GATE 4** | Test Evaluation | Training completed with zero test leakage, best checkpoint selected by Val Dice only, test evaluated exactly once. |

---

## 4. Execution Workflow (Phase 0 & Phase 1)

All commands should be executed from within `D:\Dataset_BUSI_with_GT\BUSI_AI_Project\`:

### Step 1: Validate Manifest & Raw Data (Gate 1)
```bash
python src/audit.py --raw_root D:/Dataset_BUSI_with_GT
python src/manifest.py --validate
```

### Step 2: Generate Stratified Splits (Gate 2)
```bash
python src/split.py --config configs/split.yaml
```

### Step 3: Run Smoke & Unit Test Suite (Gate 3)
```bash
pytest tests/test_data.py -v
pytest tests/test_metrics.py -v
```

### Step 4: Generate Sample Visualizations
```bash
python src/visualize.py --split train --n 16 --output outputs/reports/train_samples.png
python src/visualize.py --split val --n 16 --output outputs/reports/val_samples.png
```

---

## 5. Leakage Prevention Rules

1. **Paired Integrity**: Image and paired mask always remain in the same split.
2. **Disjointness**: Programmatic assertion verifies zero image overlap across train, val, and test splits.
3. **Deterministic Evaluation**: Val and test loaders have random augmentations turned OFF (resize + normalization only).
4. **Train-Only Normalization**: Dataset statistics are not derived from validation/test data.
5. **Single Test Evaluation**: The test split is evaluated exactly once per experiment after model selection on the validation split.
