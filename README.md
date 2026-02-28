# 🩺 Skin Cancer Classification using CNN-LSTM Hybrid Model

An automated medical imaging project built with **PyTorch** to classify skin cancer lesions from the BCN20000 dataset into 8 diagnostic categories.

## 📌 Project Overview
Early detection of skin cancer can significantly increase survival rates. This project implements a Deep Learning pipeline that combines the visual feature extraction power of **Convolutional Neural Networks (CNN)** with **Long Short-Term Memory (LSTM)** units to analyze dermatoscopic images.

## 📊 Dataset
The model is trained on the **BCN20000** dataset, which contains high-quality images of skin lesions.
- **Dataset Source:** [Kaggle - BCN20000 Dataset](https://www.kaggle.com/datasets/pasutchien/bcn20000)
- **Code Source:** [Skin Cancer CNN+LSTM Using Pytorch](https://www.kaggle.com/code/ahmedhussien710/skin-cancer-cnn-lstm-using-pytorch)
- **Classes:** 8 diagnostic categories (MEL, NV, BCC, BKL, AK, SCC, VASC, DF).
- **Size:** ~20,000 images.

## 🏗️ Model Architecture
The project uses a hybrid architecture:
1. **Feature Extractor:** `ResNet18` (Pre-trained on ImageNet). The final fully connected layer was replaced with an `Identity` layer to extract 512-dimensional feature vectors.
2. **Sequential Learner:** A single-layer `LSTM` (Hidden size: 256) to process the extracted features.
3. **Classifier:** A Linear layer mapping the LSTM output to the 8 target classes.



## 🛠️ Tech Stack
- **Framework:** PyTorch
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Image Processing:** PIL (Pillow), Torchvision

## 🚀 Key Features
- **Custom Dataset Class:** Robust handling of CSV metadata and image loading.
- **Visualizations:**
  - Training/Validation Loss and Accuracy plots.
  - Class distribution analysis.
  - Confusion Matrix for performance evaluation.
- **Model Checkpointing:** Code to save and load trained models for inference.

## 📈 Results
The model achieves competitive performance across the 8 classes. (Loss: 0.0825, Accuracy: 97.22%).
