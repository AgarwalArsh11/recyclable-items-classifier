# Grayscale Image Classifier: Bottles, Cans, and Cups

This project involves training a deep learning model to classify grayscale images into three categories: **plastic bottles**, **plastic cans**, and **cups** (plastic or paper). The goal is to use this model in a real-world machine setup controlled by an ESP32-CAM module.

## Project Objective

The model detects whether an image belongs to one of the three supported classes. If it does, the system sends a positive response to trigger a machine mechanism that opens a shutter and generates a QR code. If not, the input is ignored.

## Dataset

- Images: Grayscale (100x100 pixels)
- Classes: `bottle`, `can`, `cup`
- Structure: Data is organized into subfolders for each class
- Source: Collected from multiple sources, preprocessed, and manually labeled

## Model Details

- Framework: TensorFlow and Keras
- Architecture: CNN
- Input Shape: 100x100x1 (grayscale)
- Output: 3-class softmax
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Accuracy:
  - Training: ~99%
  - Validation: ~84%
- Saved Model: `final_balanced_grayscale_model_with_unknown.h5`

## Usage

The notebook used for training is `grayscale_cup_classifier.ipynb`. You can test the trained model using new grayscale images by loading the model and running inference with OpenCV and NumPy.

## Deployment

The trained model is intended for cloud deployment. An ESP32-CAM module captures and sends images to the cloud endpoint. Based on the model’s response, the hardware either activates or ignores the input.

## Files

- `grayscale_cup_classifier.ipynb` – Google Colab training notebook
- `final_balanced_grayscale_model_with_unknown.h5` – Saved model
- `README.md` – Project description
