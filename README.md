🧠 Brain Tumor Detection System using Machine Learning & Deep Learning
 
 Project Overview
This project presents an end-to-end Brain Tumor Detection System that classifies MRI brain images as Tumor Present or No Tumor.
The system compares traditional machine learning models with deep learning (CNN & transfer learning) to identify the most accurate approach.

The entire pipeline is implemented without OpenCV, using PIL for image processing, making it lightweight and easy to run.

 Objectives

Detect brain tumors from MRI images accurately

Compare multiple ML and DL models

Evaluate models using standard performance metrics

Automatically select and save the best-performing model


🛠️ Technologies Used

Programming Language: Python

Libraries & Frameworks:

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

TensorFlow / Keras

PIL (Python Imaging Library)

Development Environment: Jupyter Notebook / Python


📂 Dataset Description

MRI brain images stored in a local folder

Images are classified into two categories:

Yes → Tumor Present

No → No Tumor

Labels are automatically extracted from image file names


🔄 System Workflow

Load MRI images using PIL

Resize and normalize images

Perform Exploratory Data Analysis (EDA)

Split data into training and testing sets

Train multiple Machine Learning models

Train Deep Learning CNN models

Compare models using performance metrics

Select and save the best model


🤖 Machine Learning Models Implemented

Random Forest

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Gradient Boosting

Decision Tree

Naive Bayes


🧠 Deep Learning Models Implemented

Custom CNN Model

VGG16 Transfer Learning Model


📊 Model Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

ROC Curve


🏆 Best Model Selection

All models are compared based on accuracy and F1-score.
The best-performing model is automatically identified and saved for future use.

ML models are saved as .pkl files

Deep Learning models are saved as .h5 files

Label encoder is saved for inference


💾 Saved Outputs

Trained best model

Label encoder

Performance comparison visualizations


🚀 How to Run the Project

Clone the repository

Place the dataset folder in the project directory

Install required libraries

Run the notebook or Python file

pip install -r requirements.txt


📈 Project Highlights

No OpenCV dependency

Complete ML & DL comparison in one project

Clean modular code structure

Suitable for medical image classification research

Beginner-friendly and well-documented
