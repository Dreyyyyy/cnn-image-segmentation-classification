# CNN Image Segmentation & Classification

This project explores semantic segmentation of aerial imagery—specifically, detecting road networks—using multiple convolutional neural network architectures implemented in TensorFlow/Keras. The project includes both traditional full-image segmentation approaches and a patch-based CNN implementation based on the Massachusetts Roads Dataset paper.

## Project Overview

- **Dataset**: Massachusetts Roads aerial imagery; 1,108 RGB tiles paired with binary road masks.
- **Goal**: Compare several CNN-based segmentation models and analyze their performance using metrics tailored for imbalanced data.
- **Notebook**: `image-segmentation-classification_with_paper.ipynb` (Jupyter Notebook containing the full workflow).

## Contents

1. **Data Loading & Preprocessing**
   - Reads images/masks, normalizes RGB inputs to `[0, 1]`, binarizes masks, and expands channels for model compatibility.
   - White pixel filtering to remove low-quality images (>50% white pixels).
   - Predefined train/val/test splits for consistent evaluation.
2. **Exploratory Data Analysis**
   - Random visual samples and class-imbalance statistics (≈5% road pixels).
3. **Metrics & Losses**
   - Dice coefficient, IoU, accuracy, plus combined BCE+Dice loss.
   - Weighted binary cross-entropy for class imbalance.
   - Precision-Recall breakeven point (as used in the paper).
4. **Model Architectures**
   - **Baseline encoder–decoder CNN**: Simple encoder-decoder architecture.
   - **U-Net (Original)**: U-Net with skip connections.
   - **U-Net (Improved)**: Enhanced U-Net with BatchNormalization, BCE+Dice loss, and data augmentation (adapted from [Kaggle reference](https://www.kaggle.com/code/quangdamvinh/roads-segmentation)).
   - **Fully Convolutional Network (FCN)**: Multi-scale feature extraction.
   - **Paper's Patch-Based CNN**: Exact implementation from the Massachusetts Roads Dataset paper (Section 6.4.1).
5. **Training**
   - Train/validation split: 80/20 (original models) or predefined splits (improved models).
   - Early stopping, model checkpoints, and learning-rate scheduling.
   - Sample-based training termination (paper's approach: 2^25 samples).
   - On-the-fly data augmentation (RandomResizedCrop, ColorJitter).
6. **Evaluation**
   - Per-model metrics table (Loss, Dice, IoU, accuracy, training time).
   - Visual comparison of predictions and IoU distribution plots.
   - Comparison with paper's reported results.

## Key Results

### Original Implementations

| Model         | Dice | IoU  | Accuracy | Train Time |
|---------------|------|------|----------|------------|
| Baseline CNN  | 0.46 | 0.30 | ~0.95    | ~50 min    |
| U-Net (Original) | 0.45 | 0.29 | ~0.95    | ~184 min   |
| FCN           | 0.06 | 0.03 | ~0.95    | ~119 min   |

### Improved Implementations

| Model         | Dice | IoU  | Precision | Recall | Notes |
|---------------|------|------|-----------|--------|-------|
| **U-Net (Improved)** | **0.67** | **0.51** | 0.66 | 0.70 | +76% IoU improvement |
| **Paper's CNN** | **~0.82** | **~0.70** | - | - | Patch-based approach |

### Performance Comparison

- **Original U-Net → Improved U-Net**: IoU improved from 0.29 to **0.51** (+76% improvement)
- **Improved U-Net vs. Reference**: Achieves 93% of reference notebook's IoU (0.51 vs. 0.547)
- **Paper's CNN**: Achieves ~2.4x better IoU than original implementations
- Accuracy is high across all models because background pixels dominate; IoU/Dice reflect true segmentation quality.

### Key Improvements

The improved U-Net implementation includes:
- ✅ **BatchNormalization** after each convolutional layer
- ✅ **BCE + Dice Loss** for better class imbalance handling
- ✅ **Data augmentation** (RandomResizedCrop, ColorJitter)
- ✅ **White pixel filtering** to remove low-quality training images
- ✅ **Proper data splits** using predefined train/val/test sets

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd cnn-image-segmentation-classification
   ```

2. Install dependencies (assuming Python ≥3.8):
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook image-segmentation-classification_with_paper.ipynb
   ```

4. Ensure the Massachusetts Roads dataset is placed at `data/roads/tiff/` with the following structure:
   ```
   data/roads/tiff/
   ├── train/
   ├── train_labels/
   ├── val/
   ├── val_labels/
   ├── test/
   └── test_labels/
   ```

## Approach Comparison

### Full-Image Segmentation (Original)
- **Input**: Full 256×256 or 512×512 images
- **Output**: Full-resolution segmentation masks
- **Pros**: Fast inference, global context, simple pipeline
- **Cons**: Limited training data, prone to overfitting
- **Best for**: Large datasets (10k+ images), real-time applications

### Patch-Based CNN (Paper's Approach)
- **Input**: 64×64 patches extracted from images
- **Output**: 16×16 center region predictions
- **Pros**: Massive training data (millions of patches), robust predictions, matches paper results
- **Cons**: Slower inference (sliding window), more complex pipeline
- **Best for**: Limited datasets (1k images), high-quality requirements

## Lessons Learned
1. **BatchNormalization is crucial**: Stabilizes training and improves convergence
2. **Loss function matters**: BCE+Dice is much better than BCE alone for segmentation
3. **Data quality > quantity**: Filtering low-quality images is essential
4. **Data augmentation helps**: Significantly reduces overfitting
5. **Proper data splits**: Using predefined splits avoids data leakage

## Possible Future Improvements
- Experiment with different patch sizes and strides for the paper's approach
- Try ensemble methods combining multiple models
- Implement test-time augmentation for better predictions
- Explore transfer learning with pre-trained backbones
- Fine-tune hyperparameters for even better results

## Acknowledgments

- **Massachusetts Roads dataset** - The aerial imagery dataset used for training and evaluation.
- **Kaggle Notebook Reference** - The improved U-Net implementation was adapted from the excellent work by [quangdamvinh](https://www.kaggle.com/code/quangdamvinh/roads-segmentation) on Kaggle, which provided valuable insights into architecture design, loss functions, and data preprocessing techniques.
- **TensorFlow/Keras community** - Resources and tutorials that helped shape this project.