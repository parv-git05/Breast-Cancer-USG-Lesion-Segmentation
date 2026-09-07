# Breast Ultrasound AI Pipeline (BUSI Phase 1)

Reproducible medical computer-vision research pipeline for B-mode breast ultrasound (BUS) lesion segmentation and malignancy classification using the Breast Ultrasound Images (BUSI) dataset.

> **Disclaimer**: This is a research and educational codebase. It is **NOT** a medical diagnostic device and must not be used to make or infer clinical diagnoses.

---

## 1. Dataset State & Quality Contracts

- **Raw Dataset Root**: `F:\Breast-Cancer-USG-Lesion-Segmentation\` (Strictly read-only; never modified, overwritten, or deleted).
- **Subfolders**: `benign/`, `malignant/`, `normal/`.
- **Project Workspace**: `BUSI_AI_Project/` (Repository root for scripts, configs, and outputs).
- **Phase 1 Single-Mask Clean Set**: Exactly **628 images** (420 benign, 208 malignant), each with exactly one ground-truth mask.
- **Held-out Multi-Mask Cases**: 17 cases recorded in `data/manifests/BUSI_phase1_excluded.csv` (Held out for Phase 2 curation; never used in Phase 1).
- **Held-out Normal Cases**: 133 cases (Empty mask annotations; held out for Phase 2).
- **Duplicate Pair Audit**: Contradictory duplicate pair `benign (433).png` / `malignant (145).png` is confirmed deleted and absent.

### ⚠️ Patient-ID Splitting Caveat
> **Important Limitation**: Patient IDs are **not** present in the local copy of the BUSI dataset. Consequently, all train/val/test splits in Phase 1 are **image-level stratified** by lesion class (`benign` / `malignant`), not patient-wise. If verified patient-ID mappings become available in future phases, the pipeline will be updated to patient-wise splitting.

---

## 2. Experimental Design (EXP-SEG-01 vs EXP-SEG-02)

| Experiment | Role | Dataset Split | Protocol | Test Set Isolation |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-SEG-01** | Historical Baseline | 628 images total | 3-Fold Stratified CV on all 628 images | None (Each image in val once, train twice) |
| **EXP-SEG-02** | Official Benchmark | 565 Development + 63 Final Test | 3-Fold Stratified CV on 565 dev images | **100% Isolated locked holdout test set (63 images)** |

---

## 3. Directory Layout

```
BUSI_AI_Project/
├── configs/
│   ├── split.yaml
│   └── exp_seg_02.yaml                     (Official benchmark config)
├── data/
│   ├── manifests/
│   │   ├── BUSI_phase1_manifest.csv        (628 total clean cases)
│   │   ├── BUSI_phase1_excluded.csv        (17 multi-mask cases)
│   │   ├── BUSI_development_manifest.csv   (565 development cases)
│   │   └── BUSI_final_test_manifest.csv    (63 locked holdout test cases)
│   ├── test_set/                           (Physical copy for easy isolation & inspection)
│   │   ├── images/                         (63 test ultrasound images)
│   │   └── masks/                          (63 corresponding GT masks)
│   └── folds/
│       └── exp_seg_02/                     (fold_{1,2,3}_{train,val}.csv, cv_metadata.json)
├── src/
│   ├── audit.py                            (Data audit and leakage verification)
│   ├── manifest.py                         (Manifest generation and validation)
│   ├── split.py                            (Two-stage stratified holdout & CV splitting)
│   ├── transforms.py                       (Synchronized paired transforms)
│   ├── dataset.py                          (PyTorch dataset & leakage-free dataloaders)
│   ├── losses.py                           (Combined BCE + Dice loss)
│   ├── metrics.py                          (Decoupled segmentation & classification metrics)
│   ├── train.py                            (Stratified 3-fold CV training loop)
│   ├── evaluate_test.py                    (One-time locked test evaluation & plots)
│   ├── visualize.py                        (Overlay & grid generation)
│   ├── utils.py                            (Seeding, device, I/O)
│   └── models/
│       └── unet.py                         (Plain U-Net depth=4, channels=64)
├── tests/
│   ├── test_data.py
│   ├── test_metrics.py
│   └── test_unet.py
└── outputs/
    ├── checkpoints/
    │   └── exp_seg_02/
    ├── logs/
    │   └── exp_seg_02/
    └── reports/
        └── exp_seg_02/
```

---

## 4. Quality Gates

| Gate | Stage | Criteria |
| :--- | :--- | :--- |
| **GATE 1** | Raw Data & Manifest | Manifest rows = 628 (420 benign, 208 malignant), all files exist, zero overlap with excluded list, no empty masks. |
| **GATE 2** | Two-Stage Stratified Split | 63 locked test + 565 dev images, mutually disjoint (0 overlap), 3 dev folds disjoint, class balance preserved within ±2%. |
| **GATE 3** | Preprocessing & DataLoader | Batch shape `(B, 1, 256, 256)`, mask values strictly in `{0, 1}`, no NaNs/Infs, deterministic eval transforms, synchronous paired augmentations. |
| **GATE 4** | Isolated Test Evaluation | Zero test leakage during training/CV, best checkpoints selected by Val Dice only, test evaluated exactly once. |

---

## 5. Execution Workflow for EXP-SEG-02

All commands should be executed from within `BUSI_AI_Project/`:

### Step 1: Validate Splits & Verify Zero Leakage
```bash
python src/audit.py --audit_splits configs/exp_seg_02.yaml
```

### Step 2: Run Unit Tests
```bash
pytest tests/ -v
```

### Step 3: Train 3-Fold Development CV
```bash
python -u src/train.py --config configs/exp_seg_02.yaml
```

### Step 4: Evaluate on Locked Holdout Test Set
```bash
python -u src/evaluate_test.py --config configs/exp_seg_02.yaml
```

---

## 6. Leakage Prevention Rules

1. **Paired Integrity**: Image and paired mask always remain in the same split.
2. **Permanent Test Freeze**: Test manifest is created once with a fixed seed and never modified or reopened during development.
3. **Disjointness**: Programmatic assertion verifies zero image overlap between test set and any training or validation fold.
4. **Deterministic Evaluation**: Validation and test loaders have random augmentations turned OFF (resize + normalization only).
5. **Train-Only Normalization**: Dataset statistics are not derived from validation/test data.
6. **Single Test Evaluation**: The test split is evaluated exactly once per experiment after model selection on the validation split.
