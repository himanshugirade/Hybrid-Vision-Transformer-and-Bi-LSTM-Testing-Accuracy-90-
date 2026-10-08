# Skin Lesion Classification Using Multimodal Vision Transformer and BiLSTM

A deep learning approach for multi-class skin lesion classification using dermoscopic images along with patient metadata. The project combines a pretrained Vision Transformer (ViT-B/16) with a Bidirectional LSTM and attention mechanism, while incorporating metadata such as age, sex, and lesion localization.

## Overview

Skin lesion classification is challenging because different lesion categories can have visually similar characteristics and the HAM10000 dataset is highly imbalanced.

This project uses a multimodal architecture that combines:

* Dermoscopic image features extracted using **Vision Transformer (ViT-B/16)**
* Sequential modeling of image patch features using **Bidirectional LSTM**
* **Attention mechanism** to focus on informative patch representations
* Clinical metadata including **age, sex, and localization**
* **Focal Loss** for handling difficult and imbalanced samples
* **Weighted Random Sampling** during training
* Image augmentation and **MixUp** to improve model generalization

## Dataset

The project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset.

The dataset contains seven lesion categories:

| Code  | Lesion Type                                   |
| ----- | --------------------------------------------- |
| nv    | Melanocytic nevi                              |
| mel   | Melanoma                                      |
| bkl   | Benign keratosis-like lesions                 |
| bcc   | Basal cell carcinoma                          |
| akiec | Actinic keratoses / intraepithelial carcinoma |
| vasc  | Vascular lesions                              |
| df    | Dermatofibroma                                |

The model uses both:

* Dermoscopic images
* Patient metadata

Metadata features include:

* Age
* Sex
* Lesion localization

## Approach

The overall pipeline is:

```text
Dermoscopic Image
        |
        v
Image Preprocessing
        |
        v
Vision Transformer (ViT-B/16)
        |
        v
Patch Token Features
        |
        v
Bidirectional LSTM
        |
        v
Attention Mechanism
        |
        +------------------+
        |                  |
        v                  v
Image Representation   CLS Token
        |                  |
        +--------+---------+
                 |
                 v
        Metadata Features
        (Age, Sex, Location)
                 |
                 v
        Feature Fusion
                 |
                 v
        Classification Head
                 |
                 v
        7 Lesion Classes
```

## Image Preprocessing

Before training, the dermoscopic images are processed to reduce unwanted artifacts.

### Hair Removal

A morphological **Black-Hat filter** is used to detect dark hair structures. The detected regions are then reconstructed using **Telea inpainting**.

```text
Original Image
      |
      v
Grayscale Conversion
      |
      v
Black-Hat Morphological Operation
      |
      v
Hair Mask
      |
      v
Telea Inpainting
      |
      v
Cleaned Image
```

Images are resized to:

```text
384 × 384
```

## Data Augmentation

Training images use several augmentation techniques:

* Random horizontal flip
* Random vertical flip
* Random rotation
* Random affine translation
* Color jitter
* MixUp

Validation images are resized and normalized without training-time augmentation.

## Handling Class Imbalance

HAM10000 contains a significant class imbalance.

To reduce the effect of this imbalance, the training pipeline uses:

### Weighted Random Sampling

Sample weights are calculated using the inverse of class frequency:

```text
class weight = 1 / class count
```

A `WeightedRandomSampler` is then used during training.

### Focal Loss

The classification objective uses Focal Loss with:

```text
Gamma = 2.0
```

This gives more importance to difficult samples instead of treating every training example equally.

## Model Architecture

### Vision Transformer

The image backbone is:

```text
ViT-B/16
```

implemented using the `timm` library with a pretrained `vit_base_patch16_384` model.

The model extracts:

* CLS token
* Patch token representations

The CLS token is retained as a global image representation, while the patch tokens are passed to the sequential modeling stage.

### Bidirectional LSTM

The patch tokens are processed using a BiLSTM with:

```text
Hidden Units = 384
Bidirectional = Yes
```

This allows the model to learn relationships between different patch representations.

### Attention Mechanism

An attention layer assigns different weights to the BiLSTM outputs.

The weighted representations are aggregated into a single contextual feature vector.

### Metadata Fusion

The learned image representation is combined with encoded metadata:

```text
Image Features
     +
CLS Representation
     +
Age
     +
Sex
     +
Lesion Localization
     |
     v
Feature Fusion
```

The fused representation is passed through fully connected layers for final classification.

## Training Configuration

| Parameter             | Value                 |
| --------------------- | --------------------- |
| Image Size            | 384 × 384             |
| Backbone              | ViT-B/16              |
| BiLSTM Hidden Size    | 384                   |
| Number of Classes     | 7                     |
| Batch Size            | 16                    |
| Optimizer             | AdamW                 |
| Initial Learning Rate | 5e-5                  |
| Weight Decay          | 1e-4                  |
| Loss                  | Focal Loss            |
| Focal Gamma           | 2.0                   |
| Epochs                | 40                    |
| Dropout               | 0.2 / 0.3 / 0.4       |
| Sampler               | WeightedRandomSampler |
| LR Scheduler          | Cosine Annealing      |

The backbone is initially frozen and the classification head is trained first. The full model is then unfrozen for further fine-tuning with a lower learning rate.

## Evaluation

The model evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Multiclass ROC Curve
* Precision-Recall Curve

The notebook also includes visualization of sample predictions by comparing the actual and predicted lesion classes.

## Technologies Used

* Python
* PyTorch
* Torchvision
* timm
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* PIL
* Kaggle

## Project Structure

```text
skin-lesion-classification/
│
├── notebook.ipynb
├── README.md
│
└── models/
    └── multimodal_vit_bilstm_best.pth
```

The trained `.pth` file is generated after training and contains the best model weights based on validation AUC.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/skin-lesion-classification.git
cd skin-lesion-classification
```

### 2. Install dependencies

```bash
pip install torch torchvision timm
pip install opencv-python pandas numpy scikit-learn matplotlib seaborn pillow tqdm
```

### 3. Prepare the dataset

Download the HAM10000 dataset and place the following files/directories in the expected dataset location:

```text
HAM10000/
├── HAM10000_metadata.csv
├── ham10000_images_part_1/
└── ham10000_images_part_2/
```

The notebook automatically searches for the dataset paths when running in the Kaggle environment.

### 4. Run the notebook

Open:

```text
notebook.ipynb
```

and execute the cells sequentially.

## Model Output

The trained model produces a probability distribution over the seven lesion categories.

```text
Input Image + Metadata
          |
          v
      Hybrid Model
          |
          v
   Classification Head
          |
          v
7-Class Probability Output
```

## Results

Evaluation metrics and plots are generated directly from the trained model in the notebook.

The repository includes code for generating:

* Classification report
* Confusion matrix
* Multiclass ROC curves
* Precision-Recall curves
* Sample prediction visualizations

> Results should be reported from the actual training run rather than using estimated performance values.

## Notes

This project is intended for **research and educational purposes**. It is not a medical diagnostic system and should not be used as a substitute for professional medical evaluation.

## Future Improvements

Possible improvements include:

* External validation on an independent dermatology dataset
* More extensive hyperparameter optimization
* Calibration and uncertainty estimation
* Explainability using attention visualization or Grad-CAM-based methods
* Evaluation across different image acquisition conditions
* Deployment as a clinical decision-support prototype

## Author

**Himanshu Girade**

B.Tech – Artificial Intelligence & Data Science

---

If you find this project useful, consider giving the repository a ⭐.
