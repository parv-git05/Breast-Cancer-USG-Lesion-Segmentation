# EXP-SEG-02 Plain U-Net: Official Locked Test Evaluation Report

## 1. Experiment & Model Information
- **Experiment ID**: `EXP-SEG-02`
- **Model Architecture**: Plain U-Net (Depth=4, Base Channels=64, Output Channels=1)
- **Evaluated Checkpoint**: `outputs/checkpoints/exp_seg_02/final_model/final_model.pt`
- **Training Details**: Trained on all 565 development images for exactly 86 epochs
- **Test Set Size**: Total N = 63 (Benign N = 42, Malignant N = 21)
- **Segmentation Threshold**: Binary 0.5 (No threshold tuning or mask post-processing)

## 2. Overall Test Set Performance (N=63)

| Metric | Mean ± Std | Median | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| **Dice Similarity Coefficient** | **0.7216 ± 0.2971** | 0.8423 | 0.0000 | 0.9647 |
| **Intersection over Union (IoU)** | **0.6306 ± 0.2932** | 0.7276 | 0.0000 | 0.9318 |
| **Pixel Precision** | **0.7704 ± 0.3049** | 0.8988 | 0.0000 | 1.0000 |
| **Pixel Recall (Sensitivity)** | **0.7389 ± 0.3095** | 0.8722 | 0.0000 | 0.9965 |
| **Pixel Specificity** | **0.9837 ± 0.0229** | 0.9940 | 0.9147 | 1.0000 |

## 3. Subgroup Performance Analysis

### Benign Subgroup (N=42)

| Metric | Mean ± Std | Median | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| **Dice Similarity Coefficient** | **0.7352 ± 0.3214** | 0.9032 | 0.0000 | 0.9647 |
| **Intersection over Union (IoU)** | **0.6588 ± 0.3159** | 0.8235 | 0.0000 | 0.9318 |
| **Pixel Precision** | **0.7484 ± 0.3275** | 0.8987 | 0.0000 | 0.9962 |
| **Pixel Recall (Sensitivity)** | **0.7826 ± 0.3196** | 0.9270 | 0.0000 | 0.9965 |
| **Pixel Specificity** | **0.9866 ± 0.0194** | 0.9958 | 0.9188 | 1.0000 |

### Malignant Subgroup (N=21)

| Metric | Mean ± Std | Median | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| **Dice Similarity Coefficient** | **0.6943 ± 0.2465** | 0.8105 | 0.0200 | 0.9103 |
| **Intersection over Union (IoU)** | **0.5741 ± 0.2383** | 0.6813 | 0.0101 | 0.8353 |
| **Pixel Precision** | **0.8145 ± 0.2556** | 0.8988 | 0.0211 | 1.0000 |
| **Pixel Recall (Sensitivity)** | **0.6514 ± 0.2746** | 0.7262 | 0.0189 | 0.9954 |
| **Pixel Specificity** | **0.9780 ± 0.0284** | 0.9927 | 0.9147 | 1.0000 |

## 4. Test Set Integrity & Isolation Verification
- **Test N**: 63 samples (42 benign, 21 malignant)
- **Lock Status**: Test set was locked prior to final model training
- **Zero Contamination**: No test images were used during training, validation, early stopping, or threshold tuning
- **Single Evaluation**: All 63 test images were evaluated exactly once
- **Development/Test Overlap**: 0 overlapping images verified
- **Qualitative Verification**: All 63 individual 4-panel case visualizations generated and stored in `qualitative/`
