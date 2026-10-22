# Satellite Land-Cover Classification

A four-class image-classification experiment using 28 × 28 satellite patches with red, green, blue, and near-infrared channels. The notebook compares a linear softmax baseline with a small convolutional neural network.

## Models and data flow

| Component | Implementation |
| --- | --- |
| Inputs | Headerless feature CSV with 3,136 values per image; a matching four-column one-hot label CSV |
| Baseline | Flattened inputs → four-unit softmax layer; Adam and categorical cross-entropy |
| CNN | 32- and 64-filter convolution blocks, max pooling, a 128-unit dense layer, dropout, and four-class softmax |
| CNN labels | Integer class IDs with sparse categorical cross-entropy |
| Preprocessing | Float32 image tensors scaled by 255 |
| Validation | 10% random holdout for the CNN, with split seed 42 |
| Classes | Barren land, trees, grassland, other |

The two model sections are independent experiments in `satellite_one_layer_NN.ipynb`.

## Run locally

```bash
git clone https://github.com/paluuuk/Satellite_Image_Classification.git
cd Satellite_Image_Classification
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
jupyter notebook satellite_one_layer_NN.ipynb
```

Place `X_test_sat4.csv` and `y_test_sat4.csv` beside the notebook, or change its input paths. Dataset files and trained weights are not included. The current code reads numeric CSVs without headers and derives the number of images from the input.

## Evaluation and scope

This is an exploratory notebook, with validation drawn from the supplied CSV rather than a separately managed benchmark test set. The filenames alone do not establish an independent train/test protocol.

Saved outputs were cleared when correcting label/loss consistency, CNN normalization, and prediction-display indexing. Retrain the models before reporting new metrics; this repository does not present a newly verified accuracy figure.

## Repository

- [Notebook](satellite_one_layer_NN.ipynb): preprocessing, baseline, CNN, and prediction inspection.
- [Dependencies](requirements.txt): Python packages imported by the notebook.

**Stack:** Python, TensorFlow/Keras, NumPy, pandas, scikit-learn, scikit-image, Matplotlib.
