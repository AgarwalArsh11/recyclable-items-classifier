 Plastic Waste Image Classification using MobileNetV2

This project builds a deep learning model using **MobileNetV2** to classify images of different types of plastic waste. It leverages **transfer learning** and **image augmentation** to train on a custom dataset stored on Google Drive. The model is trained in **Google Colab** and can be used for waste recognition tasks like smart recycling systems.

---

## 📁 Dataset Structure

The dataset is expected to be organized into subfolders under:

/plastic_project/dataset/
├── bottle/
├── can/
├── cup/
├── paper/
├── ... (any additional classes)

Each subfolder contains training images for that category.

---

## 🧠 Model Architecture

- **Base Model**: [MobileNetV2](https://arxiv.org/abs/1801.04381) (pretrained on ImageNet)
- **Custom Layers**:
  - Global Average Pooling
  - Dropout (0.3)
  - Dense Softmax for classification

---

## 🚀 Training Pipeline

- Framework: TensorFlow / Keras
- Preprocessing: Image Augmentation with `ImageDataGenerator`
- Split: 80% train / 20% validation
- Optimizer: Adam (lr=1e-4)
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Class Imbalance: Handled using `compute_class_weight`
- Callbacks:
  - `EarlyStopping` (monitoring `val_loss`)
  - `ModelCheckpoint` (saves best model)

---

## 📈 Visualization

Training and validation **accuracy** and **loss** are plotted at the end of training to analyze model performance.

---

## ✅ Output Files

The following files are generated and saved to Google Drive:

- `final_balanced_rgb_model_with_unknown.h5`: Best model based on validation loss
- `final_balanced_rgb_model_with_unknown.h5`: Final model after all epochs

---

## 🧪 Prediction

A helper function `load_and_predict(img_path)` is included to:
- Load any image
- Preprocess and predict the class
- Display the image along with its predicted label and probabilities

Example:

python
load_and_predict("/content/drive/MyDrive/plastic_project/test_images/image_1.jpg")

📊 Sample Output

Predicted probabilities for image_1.jpg: [0.01 0.02 0.95 0.01 0.01]
Prediction: plastic_cup
