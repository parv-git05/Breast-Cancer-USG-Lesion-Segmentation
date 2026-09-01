# EXP-SEG-01 Execution Runbook

## Project Purpose
The BUSI_AI_Project focuses on Breast Ultrasound Lesion Segmentation. The primary objective is to build robust, automated segmentation models that accurately outline breast lesions in ultrasound imagery, ultimately supporting diagnostic workflows.

## Current Phase & Status
- **Phase:** Phase 1 (Baseline Development)
- **Status:** Data processing and validation pipelines are complete. The baseline Plain U-Net model is ready for 3-Fold Stratified Cross-Validation training. **NO TRAINING HAS BEEN RUN YET ON THE LOCAL LAPTOP.** Training should be executed on the designated target machine (PC).

## Authoritative Experiment Configuration
- **Experiment:** EXP-SEG-01 (Baseline U-Net)
- **Config File:** `configs/exp_seg_01.yaml`
- **Device Configuration:** Configured to `device: "auto"` (Uses CUDA GPU if available, falls back to CPU).

## Dataset Requirements
The dataset must follow the Phase 1 inclusion criteria (628 valid images):
- 420 benign
- 208 malignant
- 17 multi-mask cases excluded
- 133 normal cases excluded
- Data is defined by relative paths in `data/manifests/BUSI_phase1_manifest.csv`.

**Portability Note:** 
To run on another machine, ensure the `raw_root` path in `configs/exp_seg_01.yaml` is set correctly for your system, or leave it as `""` if you run the script from the directory containing the `benign`, `malignant`, and `normal` folders.
- Laptop default: `D:\Dataset_BUSI_with_GT`
- PC target: `F:\Breast-Cancer-USG-Lesion-Segmentation`

## 3-Fold CV Methodology
- The 628-image dataset is split into 3 mutually exclusive, stratified folds.
- For each fold:
  - 2 folds are used for Training.
  - 1 fold is used for Validation.
- Metrics (Dice, IoU, BCE+Dice Loss) are tracked, and early stopping is applied based on validation performance.
- Models are initialized from scratch at the start of every fold.
- Final CV performance is calculated by pooling the validation samples across all 3 folds.

## Exact Command to Start EXP-SEG-01
To monitor the training process live with epoch-by-epoch progress in your terminal, run the following command from the project root directory:

```bash
python -u src/train.py --config configs/exp_seg_01.yaml
```
*(The `-u` flag ensures unbuffered stdout, making logs immediately visible.)*

**IMPORTANT**: If no valid checkpoint exists, training MUST start from scratch. Ensure you do not carry over any weights accidentally.

## Expected Outputs
Upon successful execution, the pipeline will generate:
- **Checkpoints:** Best models per fold saved to `outputs/checkpoints/exp_seg_01/`
- **Logs:** Tensorboard logs or text logs saved to `outputs/logs/exp_seg_01/`
- **Reports:** Final aggregated JSON metrics saved to `outputs/reports/exp_seg_01/`
- **Terminal Output:** You will see the current fold (1/3, 2/3, 3/3), current epoch / total epochs, train/val loss, val Dice, val IoU, best validation Dice so far, early stopping status, and fold completion.

## How to Verify the Selected Device
At the very beginning of the `python -u src/train.py` output, look for the following log lines:
```text
================================================================================
STARTING STRATIFIED 3-FOLD CV EXPERIMENT: EXP-SEG-01 (Plain U-Net Baseline)
Device: cuda  (or 'cpu' if no GPU is detected)
================================================================================
```
If it says `Device: cpu` but you expect a GPU to be used, verify your PyTorch CUDA installation before continuing.
