Heart Diseases prediction using support vector machine 



Heart Disease Risk Assessment System
##  Project Overview

The Heart Disease Risk Assessment System is a machine learning–based web application designed to predict the risk of heart disease in patients using clinical and medical parameters. The system provides real-time predictions, stores patient records, and generates downloadable medical reports in PDF format.

This project combines Machine Learning, Data Processing, Database Management, and Web Development to create a practical healthcare solution.



 Machine Learning Prediction
Predicts whether a patient is at High Risk or Low Risk of heart disease.
Uses trained model (heart_disease_model.pkl) with preprocessing.

## User-Friendly Interface
Built using Streamlit for easy interaction.
Simple input fields for medical data.

## Patient Data Storage
Uses SQLite database to store patient history.
Automatically generates unique Patient IDs.

##  PDF Medical Report Generation
Generates a professional medical report.
Includes:
Patient details
Clinical inputs
Prediction result
Risk probabilities
Recommendations

## Admin Dashboard
View all patient records
Track total patients
Analyze stored data


## Technologies Used
Frontend/UI: Streamlit
Backend: Python
Machine Learning: Scikit-learn
Database: SQLite
Data Processing: Pandas, NumPy
Model Storage: Pickle
PDF Generation: ReportLab



# Input Parameters

## The model predicts based on the following medical attributes:

Age
Gender
Chest Pain Type
Exercise-induced Angina
Maximum Heart Rate
ST Depression (Oldpeak)
ST Slope
Fasting Blood Sugar


## How It Works
User enters patient details in the web interface
Data is preprocessed and arranged according to model requirements
Machine learning model predicts:
Risk category (High/Low)
Probability of disease
Result is displayed instantly
Data is saved in database
PDF report is generated and available for download