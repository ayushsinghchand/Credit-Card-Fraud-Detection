# Credit-Card-Fraud-Detection
Credit Card Fraud Detection System 🛡️
An end-to-end machine learning application that predicts the likelihood of credit card fraud in real-time. This project features a robust LightGBM classifier and a user-friendly web interface built with Streamlit.
🚀 Overview
    Fraud detection is a critical challenge in financial services due to extreme data imbalance (fraudulent transactions are rare compared to legitimate ones).         This project addresses this by implementing: 
      Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset. 
      Geospatial Feature Engineering to calculate distances between cardholders and merchants using the Haversine formula.  
      Real-time Inference via a web application.  
📊 Model Performance
The model was trained on a dataset of ~400,000 transactions and evaluated on a hold-out test set with the following results: 
    MetricScoreROC AUC93.7%
    Recall (Fraud)91.0%
    F1-Score0.94
🛠️ Tech Stack
    Language: PythonML 
    Framework: LightGBM  
    Data Handling: Pandas, NumPy  
    Pre-processing: Scikit-Learn (LabelEncoder, SMOTE) 
    Web Framework: Streamlit  
    Geospatial Logic: Geopy  
