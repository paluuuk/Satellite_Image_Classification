# Satellite Land-Cover Classification with CNNs

Convolutional neural network for classifying satellite imagery into multiple land-cover categories using **TensorFlow/Keras**.

The project demonstrates an end-to-end image-classification workflow covering preprocessing, model training, regularization, validation, and evaluation on remote-sensing imagery.

## Problem

Land-cover classification assigns a semantic class to satellite imagery, supporting applications such as:

- environmental monitoring
- land-use mapping
- agricultural assessment
- urban planning
- disaster-response analysis

The model learns visual features directly from satellite pixels rather than relying on manually engineered descriptors.

## ML Pipeline

```text
Satellite Images
      ↓
Normalization / Reshaping
      ↓
Label Encoding
      ↓
CNN Feature Extraction
      ↓
Dropout Regularization
      ↓
Softmax Classification
      ↓
Validation / Evaluation
```

## Model

The classifier is implemented in **Keras** and trained on preprocessed satellite imagery.

Key elements include:

- convolutional layers for spatial feature extraction
- normalized image inputs
- one-hot encoded labels
- dropout for regularization
- categorical classification objective
- validation monitoring during training

The model achieved approximately **97% validation accuracy** during development.

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Repository Structure

```text
Satellite_Image_Classification/
├── satellite_one_layer_NN.ipynb
├── README.md
└── .gitignore
```

## Running the Project

```bash
git clone https://github.com/paluuuk/Satellite_Image_Classification.git
cd Satellite_Image_Classification
pip install tensorflow numpy pandas matplotlib jupyter
jupyter notebook
```

Open `satellite_one_layer_NN.ipynb` and update any local dataset paths before running the training workflow.

## Current Status

This repository is intentionally a compact experimental ML project. It is useful as supporting evidence of computer-vision and remote-sensing experience, while larger projects in this portfolio demonstrate deeper engineering and deployment work.

Potential next steps include extracting the training pipeline into reusable modules, adding reproducible configuration, reporting per-class precision/recall and a confusion matrix, and exposing inference through a lightweight API or CLI.