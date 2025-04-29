# Pneumonia Detection from Chest X-ray Images using CNN

## Project Summary

This project focuses on building a Convolutional Neural Network (CNN) model to automatically detect Pneumonia from chest X-ray images.  
The model classifies chest X-rays into two categories: **Normal** and **Pneumonia**.

The project was developed in **Google Colab** using **TensorFlow** and **Keras** frameworks, with dataset handling from **Kaggle**.

---

## Tools and Libraries Used
- Python
- TensorFlow and Keras
- TensorFlow Datasets (TFDS)
- Matplotlib
- Seaborn
- Scikit-learn
- Kaggle Datasets

---

## Dataset
- **Source**: [Chest X-ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: 
  - Normal
  - Pneumonia

---

## Project Workflow

1. **Dataset Download and Extraction**  
   - Downloaded from Kaggle and extracted in Google Colab.

2. **Data Preprocessing**  
   - Resized images to 180×180 pixels.
   - Normalized pixel values to range 0–1.
   - Created training, validation, and test datasets.

3. **Model Building**  
   - Built a Sequential CNN model:
     - Conv2D ➔ MaxPooling2D ➔ Flatten ➔ Dense ➔ Dropout ➔ Dense Output
   - Used ReLU and Sigmoid activations.

4. **Model Training**
   - Trained for 10 epochs using Adam optimizer and Binary Crossentropy loss.

5. **Model Evaluation**
   - Evaluated using confusion matrix and accuracy metrics.
   - Visualized actual vs predicted labels for sample images.

---

## Model Results

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~97%
- **Confusion Matrix**:

|               | Predicted Normal | Predicted Pneumonia |
|---------------|------------------|---------------------|
| Actual Normal | 58                | 176                 |
| Actual Pneumonia | 1              | 389                 |

- **Observation**:
  - High Pneumonia detection sensitivity.
  - Some confusion in predicting Normal images as Pneumonia.

---

## Key Learnings
- Dataset handling and preprocessing.
- Building and training a CNN from scratch.
- Binary classification using deep learning.
- Model evaluation with metrics and visualizations.

---

## Future Work
- Train with more epochs.
- Apply data augmentation techniques.
- Explore transfer learning with pre-trained models like ResNet/VGG.

---

## Created By

**Bareera Mushthak**  
*Aspiring AI Engineer | Computer Vision Enthusiast*

---
