# Customer Churn Prediction & Lifetime Value (LTV) Engine

## Project Overview

This project analyzes customer churn and calculates customer Lifetime Value (LTV) using Python and machine learning.

The project helps identify customers who are likely to churn and determines the value of customers based on their monthly charges and tenure.

## Problem Statement

Customer churn is a major business problem. Losing customers can reduce revenue and increase the cost of acquiring new customers.

This project aims to:

- Analyze customer churn patterns
- Prepare customer data for machine learning
- Predict customer churn
- Calculate Customer Lifetime Value (LTV)
- Identify high-value customers who have churned
- Present insights through an interactive dashboard

## Dataset

The dataset contains customer information including:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Contract
- Payment Method
- Monthly Charges
- Total Charges
- Churn

After data cleaning, the project contains 7,021 customer records.

## Project Workflow

Raw Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Engineering
↓
Machine Learning Preparation
↓
Churn Prediction
↓
Model Evaluation
↓
LTV Calculation
↓
Interactive Dashboard
↓
Business Insights

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Git
- GitHub

## Stage 1 - Data Cleaning

The raw customer dataset was cleaned by handling missing and invalid values and preparing the data for analysis.

The cleaned dataset contains 7,021 customers.

## Stage 2 - Exploratory Data Analysis

Customer churn was analyzed based on important factors such as:

- Contract type
- Tenure
- Monthly charges
- Services
- Payment method

Month-to-month contract customers showed substantially higher churn compared with customers on longer contracts.

## Stage 3 - Feature Engineering

Additional features were created to improve analysis and machine learning.

New features include:

- TenureGroup
- MonthlyChargeGroup
- TotalChargeGroup
- ServiceCount

## Stage 4 - Machine Learning Preparation

Categorical variables were encoded using one-hot encoding.

The dataset was divided into:

- Training data
- Testing data

The target variable was:

- Churn = 0 for No
- Churn = 1 for Yes

## Stage 5 - Churn Prediction

A Logistic Regression model was developed to predict customer churn.

The model uses customer demographic, service, contract, tenure and billing information.

## Stage 6 - Lifetime Value Calculation

Customer Lifetime Value was calculated using:

LTV = Monthly Charges × Tenure

Example:

Monthly Charges = 70

Tenure = 24 months

LTV = 70 × 24 = 1,680

Customers were divided into:

- Low LTV
- Medium LTV
- High LTV

### LTV Results

- Total customers: 7,021
- Average LTV: 2,286.61
- Low LTV customers: 2,885
- Medium LTV customers: 1,934
- High LTV customers: 2,202
- High-LTV customers who churned: 345

## Stage 7 - Dashboard

An interactive Streamlit dashboard was created to visualize:

- Total customers
- Churned customers
- Churn rate
- Average LTV
- Churn distribution
- Churn by contract
- Churn by tenure
- LTV categories
- High-LTV customers who churned

To run the dashboard:

```bash
python -m streamlit run dashboard.py