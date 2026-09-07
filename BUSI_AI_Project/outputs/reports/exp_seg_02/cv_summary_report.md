# EXP-SEG-02 Plain U-Net: Development 3-Fold Cross-Validation Report

## Overview & Protocol
- **Model**: Plain U-Net (Depth=4, BaseChannels=64)
- **Evaluation Protocol**: 3-Fold Stratified Development Cross-Validation
- **Total Development Images Evaluated**: 565 (378 benign, 187 malignant)
- **Total Training Duration**: 65.84 minutes

## Primary Out-of-Fold Development Performance

| Metric | Mean ± Std / Value |
| :--- | :--- |
| **Dice Coefficient (DSC)** | **0.7149 ± 0.2822** |
| **Intersection over Union (IoU)** | **0.6169 ± 0.2812** |
| **Pixel Precision** | 0.7945 |
| **Pixel Recall (Sensitivity)** | 0.6667 |
| **Pixel Specificity** | 0.9821 |

## Fold-Wise Results Breakdown

| Fold | Val Images | Best Epoch | Val Dice (Mean ± Std) | Val IoU (Mean ± Std) | Duration (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fold 1 | 1 | Ep 70 | 0.7282 ± 0.2710 | 0.6288 ± 0.2699 | 1211.2s |
| Fold 2 | 2 | Ep 86 | 0.7157 ± 0.2697 | 0.6140 ± 0.2763 | 1354.9s |
| Fold 3 | 3 | Ep 100 | 0.7007 ± 0.3054 | 0.6078 ± 0.2979 | 1356.2s |
