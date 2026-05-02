# Credit Card Fraud Detection System 🛡️

An end-to-end machine learning application that predicts the likelihood of credit card fraud in real-time. This project features a robust LightGBM classifier and a user-friendly web interface built with Streamlit.

---

## 🚀 Overview

Fraud detection is a critical challenge in financial services due to extreme data imbalance, where fraudulent transactions are significantly rarer than legitimate ones. This project addresses the problem using advanced machine learning techniques and feature engineering approaches.

The system includes:

- **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance.
- **Geospatial Feature Engineering** using the Haversine formula to calculate distances between cardholders and merchants.
- **Real-time Fraud Prediction** through an interactive Streamlit web application.
- **LightGBM Classifier** optimized for high performance and fast inference.

---

## 📊 Model Performance

The model was trained on a dataset containing approximately **400,000 transactions** and evaluated on a separate test dataset.

| Metric | Score |
|--------|--------|
| ROC AUC | 93.7% |
| Recall (Fraud) | 91.0% |
| F1-Score | 0.94 |

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning Framework
- LightGBM

### Data Processing & Analysis
- Pandas
- NumPy

### Pre-processing & Model Utilities
- Scikit-Learn
  - LabelEncoder
  - SMOTE
  - Train-Test Split
  - Evaluation Metrics

### Web Framework
- Streamlit

### Geospatial Processing
- Geopy

---

## ⚙️ Features

- Real-time fraud prediction
- Interactive web-based dashboard
- Imbalanced data handling with SMOTE
- Distance-based transaction analysis
- High recall for fraud detection
- Scalable and lightweight architecture

---

## 📁 Project Structure

```bash
Credit-Card-Fraud-Detection/
│
├── app.py                  # Streamlit web application
├── model.pkl               # Trained LightGBM model
├── preprocessing.py        # Data preprocessing pipeline
├── train_model.py          # Model training script
├── requirements.txt        # Project dependencies
├── dataset/                # Dataset files
└── README.md               # Project documentation
