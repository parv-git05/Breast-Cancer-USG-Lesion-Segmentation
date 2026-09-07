# BUSI AI Project: Experiment Execution Runbook

## Project Purpose
The BUSI_AI_Project focuses on Breast Ultrasound Lesion Segmentation. The primary objective is to build robust, automated segmentation models that accurately outline breast lesions in ultrasound imagery, supporting downstream diagnostic workflows.

---

## Methodological Overview: EXP-SEG-01 vs EXP-SEG-02

### 1. Historical Baseline (EXP-SEG-01)
- **Designation**: Historical reference baseline
- **Dataset**: All 628 Phase-1 single-mask images (420 benign, 208 malignant)
- **Protocol**: 3-Fold Stratified Cross-Validation on all 628 images
- **Limitation**: Every image was in validation once and in training twice. **No independent unseen holdout test set** was isolated.
- **Reference Result**: Pooled OOF Dice = 0.7130 ± 0.2941, IoU = 0.6195 ± 0.2915.
- **Report Location**: `outputs/reports/exp_seg_01/cv_summary_report.md`

### 2. Official Standard Protocol (EXP-SEG-02)
- **Designation**: Official leakage-free segmentation benchmark
- **Holdout Test Set (~10%)**: Exactly **63 images** (42 benign, 21 malignant) locked and isolated before any training.
  - Manifest: `data/manifests/BUSI_final_test_manifest.csv`
  - Physical Folder: `data/test_set/images/` and `data/test_set/masks/`
  - Rule: Test images NEVER participate in training, validation, early stopping, hyperparameter tuning, or architecture decisions.
- **Development Set (~90%)**: Exactly **565 images** (378 benign, 187 malignant).
  - Manifest: `data/manifests/BUSI_development_manifest.csv`
  - Protocol: 3-Fold Stratified Cross-Validation strictly on the 565 development images (`data/folds/exp_seg_02/`).
- **Final Evaluation**: Evaluated ONCE on the locked 63-image final test set using the 3-Fold Ensemble and Best Single CV Fold model.

---

## Dataset Requirements (Phase 1 Clean Single-Mask Subset)
The dataset must follow the Phase 1 inclusion criteria (628 valid images):
- 420 benign
- 208 malignant
- 17 multi-mask cases excluded (`data/manifests/BUSI_phase1_excluded.csv`)
- 133 normal cases excluded (empty masks)
- Deleted duplicate pair verified absent (`benign (433)` and `malignant (145)`).

**Portability Note:**
`raw_root` path in configs is configured to `".."`, resolving to `F:\Breast-Cancer-USG-Lesion-Segmentation` when run from `BUSI_AI_Project`.

---

## Execution Instructions for EXP-SEG-02

All commands should be executed from within `BUSI_AI_Project/`:

### Step 1: Preflight Split & Data-Leakage Audit
Before starting training, run the automated leakage gate:
```bash
python src/audit.py --audit_splits configs/exp_seg_02.yaml
```
*Ensures 63 test / 565 dev disjoint split, 0 overlap, and 100% file presence on disk.*

### Step 2: Train Development 3-Fold CV
To train the Plain U-Net baseline on the 3 development folds with live unbuffered logging:
```bash
python -u src/train.py --config configs/exp_seg_02.yaml
```
- Trains Fold 1 (376 train, 189 val), Fold 2 (377 train, 188 val), Fold 3 (377 train, 188 val).
- Early stopping patience: 20 epochs.
- Best model checkpoints saved to: `outputs/checkpoints/exp_seg_02/fold_{1,2,3}/best_model.pt`.
- OOF Development CV Report: `outputs/reports/exp_seg_02/cv_summary_report.md`.

### Step 2.5: Train Final Baseline Model on All 565 Development Images
Train the final Plain U-Net baseline model on the complete 565 development dataset using the verified median fixed budget of 86 epochs:
```bash
python -u src/train.py --config configs/exp_seg_02.yaml --train_final --final_epochs 86
```
- Uses ALL 565 development images (378 benign, 187 malignant).
- Fixed 86-epoch budget with zero early stopping.
- Hard pre-training data gate strictly enforced (zero test leakage).
- Checkpoint: `outputs/checkpoints/exp_seg_02/final_model/final_model.pt`.
- History log: `outputs/logs/exp_seg_02/final_model/training_history.csv`.
- Training curves: `outputs/reports/exp_seg_02/final_model/training_curves.png`.
- Metadata: `outputs/reports/exp_seg_02/final_model/training_metadata.json`.

### Step 3: Run Final Test Set Evaluation & Visualization
After development training completes, evaluate the models on the locked test set:
```bash
python -u src/evaluate_test.py --config configs/exp_seg_02.yaml
```
- Evaluates the 3-Fold Ensemble and Best Single Fold model on the 63 test images.
- Computes per-image Dice, IoU, Precision, Recall, Specificity.
- Generates 63 individual case visualizations in `outputs/reports/exp_seg_02/test_cases_individual/`.
- Generates summary sample grids (Best 4, Median 4, Worst 4 failure cases).
- Generates diagnostic plots (Plot A: Per-image Dice, Plot B: Dice histogram, Plot C: IoU plots, Plot D: Box plots, Plot E: Pixel confusion matrix).
- Outputs official markdown report: `outputs/reports/exp_seg_02/final_test_report.md`.

---

## Expected Output Structure
```
outputs/
├── checkpoints/
│   └── exp_seg_02/
│       ├── fold_{1,2,3}/best_model.pt
│       └── final_model/final_model.pt
├── logs/
│   └── exp_seg_02/
│       ├── fold_{1,2,3}/training_history.csv
│       └── final_model/training_history.csv
└── reports/
    └── exp_seg_02/
        ├── cv_summary_report.md
        ├── cv_summary.json
        ├── cv_out_of_fold_predictions.csv
        ├── final_model/
        │   ├── training_curves.png
        │   └── training_metadata.json
        ├── final_test_report.md
        ├── final_test_metrics.json
        ├── final_test_predictions_ensemble.csv
        ├── plot_a_per_image_dice.png
        ├── plot_b_dice_distribution_histogram.png
        ├── plot_c_iou_distribution.png
        ├── plot_d_metric_boxplots.png
        ├── plot_e_pixel_confusion_matrix.png
        ├── test_best_samples.png
        ├── test_median_samples.png
        ├── test_worst_failure_samples.png
        └── test_cases_individual/
```

