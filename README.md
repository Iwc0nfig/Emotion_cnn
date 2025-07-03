# Emotion Recognition on FER2013 Dataset

This project focuses on emotion recognition using a Convolutional Neural Network (CNN) trained on the FER2013 dataset. It addresses class imbalance by augmenting the underrepresented "disgusted" emotion class.

## 📁 Project Structure

- `data.zip` — Contains both training and validation datasets (unzipped during preprocessing).
- `data_augmentation.py` — Script to generate additional images specifically for the "disgusted" class (originally ~450 samples).
- `plotting.py` — Plots and saves training/validation accuracy and loss curves to `result.png`.
- `model.py` — (Optional) Define and train your CNN model here.
- `training.py` — (Optional) Main training loop.

## 🚀 How to Run

1. **Unzip the data**  
   Extract `data.zip`:
   ```bash
   unzip data.zip -d ./data
