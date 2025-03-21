**Automated Depression Detection Using Facial
 Behavior and Head Gestures with Hybrid and
 Universal Learning Models**


This project implements deep learning (Bidirectional LSTM) and machine learning (XGBoost, SVM, Random Forest) models for depression detection using facial behavior data. The dataset contains facial action units (AU), eye and mouth classification probabilities, head movements (Euler angles), and 133 landmark points extracted from participants’ face recordings. Depression labels are assigned using PHQ-9 scores.

**Key Features**

**Feature Engineering**: Extracted Action Units, Head Movements, Eye Open Probabilities, and Facial Landmarks from JSON data.

**Data Preprocessing:**
The dataset undergoes several preprocessing steps to ensure clean, balanced, and normalized input for the model:

Sorting & Label Extraction

The data is sorted by timestamp (start_ts) to maintain time-sequence order.
The target variable (depression_episode) is extracted for classification.
Feature Selection & Cleaning

Non-relevant columns (pid, timestamps, PHQ-9 scores) are dropped.
Non-numeric features (e.g., boundingBox) are removed.
Missing Value Handling

Applied Mean Imputation using SimpleImputer to fill missing values.
Feature Scaling

Used StandardScaler to normalize features for better model performance.
Class Imbalance Handling

SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the dataset.
Reshaping for Deep Learning Models

Data is reshaped into 3D format (samples, timesteps, features) for LSTM models.
Class Weights Calculation

Used compute_class_weight to adjust for imbalanced classes, improving model learning.

**Deep Learning Model - Bidirectional LSTM:**
Universal Model (LOPO - Leave-One-Participant-Out Cross-Validation)
Hybrid Model (LOPDO - Leave-One-Participant-Day-Out Cross-Validation)

**Machine Learning Models:**
XGBoost, SVM, Random Forest for comparative analysis.
Performance Metrics: Confusion Matrix, Precision, Recall, F1-score, ROC-AUC Curve.

📊 **Model Performance**
Hybrid Model (LOPDO) - Bidirectional LSTM
Universal Model (LOPO) - Bidirectional LSTM
Comparison with XGBoost, SVM, and Random Forest
