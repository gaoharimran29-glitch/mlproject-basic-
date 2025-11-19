# Basic Machine Learning Projects – Web Application

**Live Website:** https://mlproject-basic.onrender.com/

This project is a collection of **10 Basic Machine Learning Projects**, neatly organized into **Supervised**, **Unsupervised**, and **Association Rule Learning**.  
Each project has its own ML model, HTML form, and Flask route to collect user inputs and generate predictions or clustering results.

All models are trained separately and integrated into a single Flask-based web app.  
The entire application is deployed using **Render**.

---

## Supervised Learning

### Regression Models
These models predict continuous numerical values.

#### **1. Student Performance Predictor**  
Predicts the expected marks/performance of a student based on inputs such as study hours, attendance, and habits.

#### **2. Employee Salary Predictor**  
Estimates employee salary using factors like experience, job role, skill level, and age.

#### **3. Calories Burned Predictor**  
Predicts the number of calories burned based on exercise intensity, duration, and biometric details.

---

### Classification Models  
These models predict categories or classes.

#### **1. Home Loan Approval Predictor**  
Predicts whether a home loan application will be approved based on financial and personal details.

#### **2. Diabetes Risk Predictor**  
Predicts the risk of diabetes using medical data such as BMI, glucose level, and age.

#### **3. Telecom Customer Churn Predictor**  
Predicts whether a customer is likely to leave a telecom service provider.

---

## Unsupervised Learning

### Clustering Models  
These models group similar data points.

#### **1. Customer Segmentation Project**  
Clusters customers into different groups based on spending patterns and demographic features.

#### **2. Real Estate House Clustering**  
Groups houses based on features like price per sqft and living area to identify housing tiers (low, medium, high).

---

## Association Rule Learning

### Rule Mining Models  
These models find patterns and relationships in data.

#### **1. Market Basket Rules Mining**  
Discovers product associations, such as which items are frequently bought together.

#### **2. Disease Symptoms Rules Mining**  
Finds associations between symptoms to understand disease patterns.

---

## How the System Works

### Model Training  
Data cleaning, preprocessing, feature engineering, and model training were performed separately.  
All trained model files (.pkl) are stored and loaded inside the Flask app during runtime.
File name `modeltraining&datasets` contains all the codes of how I performed data cleaning, preprocessing, feature engineering and model training and datasets.

### Flask App  
The `mlprojects` folder contains:

- `app.py` (main Flask application)  
- Templates for all 10 projects  
- Loaded ML models  
- Routes for every prediction page  

User enters inputs → Flask preprocesses data → ML model predicts → Result shown on webpage.

### Deployment on Render  
The app is deployed using:

- `requirements.txt` for dependencies  
- Main entry point: `app.py`  

---

## Live Demo

**Visit Website:** https://mlproject-basic.onrender.com/

---