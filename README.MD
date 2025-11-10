# CNN Image Segmentation & Classification

This project explores semantic segmentation of aerial imagery—specifically, detecting road networks—using multiple convolutional neural network architectures implemented in TensorFlow/Keras.

## Project Overview

- **Dataset**: Massachusetts Roads aerial imagery; 1,108 RGB tiles paired with binary road masks.
- **Goal**: Compare several CNN-based segmentation models and analyze their performance using metrics tailored for imbalanced data.
- **Notebook**: `image-segmentation-classification.ipynb` (Jupyter Notebook containing the full workflow).

## Contents

1. **Data Loading & Preprocessing**
   - Reads images/masks, normalizes RGB inputs to `[0, 1]`, binarizes masks, and expands channels for model compatibility.
2. **Exploratory Data Analysis**
   - Random visual samples and class-imbalance statistics (≈5% road pixels).
3. **Metrics & Losses**
   - Dice coefficient, IoU, accuracy, plus combined BCE+Dice loss.
4. **Model Architectures**
   - Baseline encoder–decoder CNN.
   - U-Net with skip connections.
   - Fully Convolutional Network (FCN) leveraging multi-scale features.
5. **Training**
   - Train/validation split: 80/20.
   - Early stopping, model checkpoints, and learning-rate scheduling.
6. **Evaluation**
   - Per-model metrics table (Loss, Dice, IoU, accuracy, training time).
   - Visual comparison of predictions and IoU distribution plots.
7. **Conclusions**
   - Summary of key findings and next steps (e.g., data augmentation, longer training).

## Key Results

| Model         | Dice | IoU  | Accuracy | Train Time |
|---------------|------|------|----------|------------|
| Baseline CNN  | ~0.46|~0.30| ~0.95    | ~50 min    |
| U-Net         | ~0.45|~0.29| ~0.95    | ~184 min   |
| FCN           | ~0.06|~0.03| ~0.95    | ~119 min   |

- Accuracy is high across the board because background pixels dominate; IoU/Dice reflect true segmentation quality.
- The lightweight baseline slightly outperforms U-Net with the current data/training setup.
- FCN struggles due to model size vs. limited data/epochs.

## Getting Started

1. Clone the repository and open the notebook:
  
   git clone <repo-url>
   cd cnn-image-segmentation-classification
   2. Install dependencies (assuming Python ≥3.8):
  
   pip install -r requirements.txt
   3. Launch Jupyter:
  
   jupyter notebook image-segmentation-classification.ipynb
   4. Ensure the Massachusetts Roads dataset is placed at `data/roads/` (or update the config in the notebook).

## Possible Improvements

- On-the-fly data augmentation (flips, rotations, brightness).
- Longer training schedules with tuned learning rates.
- Class imbalance strategies (weighted losses, focal loss).
- Experiment with lighter FCN backbones or transfer learning.

## Acknowledgments

- Massachusetts Roads dataset.
- TensorFlow/Keras community resources and tutorials.