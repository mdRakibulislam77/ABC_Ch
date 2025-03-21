Automated Depression Detection Using Facial
 Behavior and Head Gestures with Hybrid and
 Universal Learning Models


This project implements deep learning (Bidirectional LSTM) and machine learning (XGBoost, SVM, Random Forest) models for depression detection using facial behavior data. The dataset contains facial action units (AU), eye and mouth classification probabilities, head movements (Euler angles), and 133 landmark points extracted from participants’ face recordings. Depression labels are assigned using PHQ-9 scores.

🚀 Key Features
Feature Engineering: Extracted Action Units, Head Movements, Eye Open Probabilities, and Facial Landmarks from JSON data.
Data Preprocessing:
KNN Imputation for missing values.
RobustScaler & PCA for feature scaling & dimensionality reduction.
ADASYN oversampling for class balance.
Deep Learning Model - Bidirectional LSTM:
Universal Model (LOPO - Leave-One-Participant-Out Cross-Validation)
Hybrid Model (LOPDO - Leave-One-Participant-Day-Out Cross-Validation)
Machine Learning Models:
XGBoost, SVM, Random Forest for comparative analysis.
Performance Metrics: Confusion Matrix, Precision, Recall, F1-score, ROC-AUC Curve.
📊 Model Performance
Hybrid Model (LOPDO) - Bidirectional LSTM
Universal Model (LOPO) - Bidirectional LSTM
Comparison with XGBoost, SVM, and Random Forest
