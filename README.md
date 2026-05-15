# Skin Cancer Detection by Lesion Images with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-CNN-D00000.svg)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Eval-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Convolutional Neural Network that classifies seven types of pigmented skin lesions from dermatoscopic images, trained on the **HAM10000** dataset. Built as a course project at Hacettepe University.

The repository contains **two versions of the same notebook** — English (`_ENG`) and Turkish (`_TR`) — with identical code but localized narration.

## Overview

For dermatologists, examining skin lesions is the first and most important step in diagnosing skin cancer. This project builds a CNN that learns the visual patterns of seven lesion classes from the HAM10000 dataset and predicts the lesion type from a 28×28 RGB image.

The pipeline covers:
1. **Data understanding** — class distribution, confirmation source, gender / age / body-part distributions
2. **Data preparation** — pixel/label separation, one-hot encoding, 28×28×3 reshape
3. **Data augmentation** — `ImageDataGenerator` (rotations, shifts, flips) to expand the training set
4. **Modeling** — Keras Sequential CNN with Conv2D + MaxPooling + Dense layers
5. **Hyperparameter tuning** — grid sweep over batch size, epochs, and learning rate
6. **Evaluation** — accuracy / loss curves, AUC, confusion matrix, classification report

## Dataset

- **Source**: [HAM10000 — Skin Cancer MNIST on Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) (see `dataset_kaggle_link.txt`)
- **Size**: 10,015 dermatoscopic images, 28×28×3 RGB
- **Classes** (7): `nv` (melanocytic nevi), `mel` (melanoma), `bkl` (benign keratosis), `bcc` (basal cell carcinoma), `akiec` (actinic keratoses), `vasc` (vascular lesions), `df` (dermatofibroma)
- **Metadata**: `HAM10000_metadata.csv` is included in this repository

> The image pixel CSV (`hmnist_28_28_RGB.csv`) is **not committed** due to size — download it from Kaggle and place it in the working directory before running the notebooks.

## Tech Stack

- **Language**: Python 3.8+
- **Deep Learning**: TensorFlow / Keras (`Sequential`, `Conv2D`, `MaxPooling2D`, `Dense`, `Dropout`, `BatchNormalization`)
- **Training utilities**: `ImageDataGenerator`, `EarlyStopping`, `ReduceLROnPlateau`, `RMSprop`
- **Evaluation**: scikit-learn (`classification_report`, `confusion_matrix`)
- **Data / viz**: pandas, numpy, matplotlib, seaborn
- **Environment**: Jupyter Notebook

## Project Structure

```
.
├── Skin_Cancer_Lesion_Detection_ENG.ipynb   # Full pipeline (English narration)
├── Skin_Cancer_Lesion_Detection_TR.ipynb    # Aynı pipeline (Türkçe anlatım)
├── HAM10000_metadata.csv                    # Patient & lesion metadata
├── dataset_kaggle_link.txt                  # Where to download the image CSV
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/Msubasi1/Skin-Cancer-Detection-by-Lesion-Images-with-ML.git
cd Skin-Cancer-Detection-by-Lesion-Images-with-ML
python3 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the pixel CSV from Kaggle (link in `dataset_kaggle_link.txt`) and place `hmnist_28_28_RGB.csv` in the project root.

## Usage

Open either notebook in Jupyter:

```bash
jupyter notebook Skin_Cancer_Lesion_Detection_ENG.ipynb
# veya
jupyter notebook Skin_Cancer_Lesion_Detection_TR.ipynb
```

Run all cells. Training on CPU takes ~20–40 minutes depending on the chosen epoch count.

## Model Architecture

A `Sequential` CNN:

| Block | Layers |
|-------|--------|
| 1 | 2 × Conv2D (32 filters, 3×3, ReLU) → MaxPooling2D (2×2) → Dropout |
| 2 | 2 × Conv2D (64 filters, 3×3, ReLU) → MaxPooling2D (2×2) → Dropout |
| 3 | Flatten → Dense (ReLU) → BatchNormalization → Dropout → Dense (7, softmax) |

Loss: `categorical_crossentropy` · Optimizer: `RMSprop`

### Best Hyperparameters (from grid sweep)

| Parameter | Value |
|-----------|-------|
| batch_size | 128 |
| epochs | 50 |
| learning_rate | 0.0001 |

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **~75%** |
| Train/Test split | 80 / 20 |
| AUC | High (see notebook) |

The model's loss shows a consistent downward trend across epochs, and the AUC score on the test set indicates strong discrimination across the seven lesion classes despite class imbalance dominated by `nv`.

## Authors

- **Muhammet Subaşı** ([@Msubasi1](https://github.com/Msubasi1)) — 21627601
- **Harun Burkuk** — 21627045

## License

Released under the [MIT License](LICENSE).
