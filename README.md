# 🧠 Brain Tumor Detection System

### Using Machine Learning & Deep Learning

## 📌 Project Overview

This project implements a **complete Brain Tumor Detection System** that classifies MRI brain images into **Tumor Present** and **No Tumor** categories.

Both **traditional Machine Learning models** and **Deep Learning models (CNN & Transfer Learning)** are trained, evaluated, and compared to identify the **best-performing approach** for medical image classification.

The system is developed **without OpenCV**, using **PIL (Python Imaging Library)** for image preprocessing, making it lightweight and easy to execute.

---

## 🎯 Objectives

* Accurately detect brain tumors from MRI images
* Compare Machine Learning and Deep Learning techniques
* Evaluate models using standard performance metrics
* Automatically select and save the best-performing model

---

## 🛠️ Technologies Used

**Programming Language**

* Python

**Libraries & Frameworks**

* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn
* TensorFlow / Keras
* PIL (Python Imaging Library)

**Development Environment**

* Jupyter Notebook / Python

---

## 📂 Dataset Description

* MRI brain images stored in a local dataset folder
* Images are categorized into:

  * **Yes** – Tumor Present
  * **No** – No Tumor
* Class labels are automatically extracted from image file names

---

## 🔄 System Workflow

1. Load MRI images using PIL
2. Resize and normalize image data
3. Perform Exploratory Data Analysis (EDA)
4. Split dataset into training and testing sets
5. Train multiple Machine Learning models
6. Train Deep Learning CNN models
7. Evaluate and compare all models
8. Select and save the best-performing model

---

## 🤖 Machine Learning Models Implemented

* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Gradient Boosting
* Decision Tree
* Naive Bayes

---

## 🧠 Deep Learning Models Implemented

* Custom Convolutional Neural Network (CNN)
* VGG16 Transfer Learning Model

---

## 📊 Model Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve

---

## 🏆 Best Model Selection

All models are compared using **Accuracy and F1-Score**.
The model with the highest performance is automatically identified and saved.

* Machine Learning models are saved as `.pkl` files
* Deep Learning models are saved as `.h5` files
* Label encoder is saved for future inference

---

## 💾 Saved Outputs

* Best-trained classification model
* Label encoder
* Performance comparison visualizations

---

## 🚀 How to Run the Project

1. Clone the repository
2. Place the dataset folder in the project directory
3. Install required dependencies
4. Run the notebook or Python script

```bash
pip install -r requirements.txt
```

---

## ⭐ Project Highlights

* No OpenCV dependency
* Complete ML vs DL model comparison
* Modular and well-structured code
* Suitable for medical image classification research
* Beginner-friendly and well-documented implementation

---



