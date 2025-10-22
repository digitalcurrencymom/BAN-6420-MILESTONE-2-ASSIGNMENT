## Cancer PCA Project
 Overview
This project applies Principal Component Analysis (PCA) to the Breast Cancer Wisconsin dataset from sklearn.datasets. The goal is to reduce the dataset’s dimensionality to two principal components for visualization and feature selection. As an optional enhancement, a logistic regression model is trained on the reduced data to demonstrate predictive performance.

This work supports the Anderson Cancer Center’s initiative to identify key variables for donor funding decisions by simplifying complex data into actionable insights.

## Tools & Technologies
Language: Python 3.8+

Libraries:

scikit-learn

pandas

matplotlib

seaborn

## Project Files
pca_cancer.py: Main script that loads the dataset, standardizes features, applies PCA, visualizes the results, and optionally fits a logistic regression model.

pca_2components.png: Scatter plot showing the PCA-reduced data with class separation.

logreg_report.txt: Classification report from logistic regression, including accuracy, precision, recall, and F1-score.

README.md: Project documentation and instructions.

## How to Run
1. Install Dependencies
Make sure Python 3.8+ is installed. Then install required packages:

bash
pip install scikit-learn pandas matplotlib seaborn
2. Execute the Script
Run the script from your terminal or PowerShell:

bash
python path/to/pca_cancer.py
Replace path/to/ with the actual location of the script on your machine.

 
 ## What the Script Does
 PCA Implementation
Loads the Breast Cancer dataset

Standardizes features using StandardScaler

Applies PCA to reduce the dataset to 2 components

Saves a 2D scatter plot of the PCA results

## Bonus: Logistic Regression
Splits PCA-reduced data into training and test sets

Trains a logistic regression model

Evaluates model performance and saves a detailed report

## Outputs
pca_2components.png: Visualizes the separation between malignant and benign cases using PCA.

logreg_report.txt: Shows model accuracy of 99.12%, with strong precision and recall across both classes.

## Notes
PCA is an unsupervised technique used to reduce dimensionality while preserving variance.

Logistic regression is a supervised model used here to validate the usefulness of PCA-reduced features.

This project is part of Milestone Assignment 2 for BAN6420 at Nexford University.

 ## Author
Kendra Onah
Data Analyst 
Nexford University 
