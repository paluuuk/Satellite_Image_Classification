# Satellite Image Classification

This repository contains an implementation of a convolutional neural network (CNN) to classify satellite imagery into different land cover categories. The model was developed using Keras and trained on preprocessed satellite image data. The classification task is designed to distinguish between multiple terrain types such as forest, water, urban, and barren land.

During development, the model achieved strong validation performance through standard preprocessing steps (normalization, reshaping, and one-hot encoding) and the use of dropout for regularization.

---

## Project Overview

Satellite image classification is a remote sensing task where an input image is assigned a label corresponding to a specific land cover type. Automated classification has applications in:

- Land use and land cover mapping  
- Environmental monitoring  
- Agricultural assessment  
- Urban planning  
- Disaster response  

Deep learning models such as CNNs are well-suited for this task because they can learn hierarchical spatial features directly from image pixels without manual feature engineering.

---

## Repository Structure

| File / Directory | Description |
|------------------|-------------|
| `satellite_one_layer_NN.ipynb` | Jupyter notebook with model definition, training, and evaluation |
| `README.md` | This documentation file |
| `.gitignore` | Standard Git ignore configuration |

---

## Model Description

The model is a convolutional neural network implemented in Keras. Key characteristics include:

- Input format: multi-band satellite images reshaped into appropriate tensor format
- Preprocessing: normalization of pixel values and one-hot encoding of labels
- Training: model fitted with training and validation data
- Regularization: dropout layers to reduce overfitting
- Evaluation: standard accuracy metrics used to assess performance

The trained model achieved approximately **97% validation accuracy** over a 10-epoch run, indicating reliable performance for the classification categories defined in this project.

---

## Getting Started

### Prerequisites

Before running the code locally, ensure you have:

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required libraries installed (TensorFlow/Keras, NumPy, Pandas, Matplotlib)

### Installation

Clone the repository:

```bash
git clone https://github.com/paluuuk/Satellite_Image_Classification.git
cd Satellite_Image_Classification
