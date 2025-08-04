# Kaggle-House-Price-Predictor

## Project Overview
This project is a comprehensive, end-to-end data science solution for the Kaggle "House Prices: Advanced Regression Techniques" competition. The goal is to predict the sale price of residential homes in Ames, Iowa, using a dataset of 79 explanatory variables.

The project demonstrates a full data science workflow, including:

- In-depth Exploratory Data Analysis (EDA)

- Creative and robust Feature Engineering

- Training and evaluation of multiple advanced regression models

- Development of an interactive web application to serve the final model.

## Key Steps & Methodology
- Exploratory Data Analysis (EDA): The SalePrice target variable was found to be right-skewed and was log-transformed to normalize its distribution. A correlation analysis identified key features like OverallQual, GrLivArea, and TotalBsmtSF as strong predictors.

- Feature Engineering: A systematic approach was taken to handle the dataset's extensive missing values, imputing them based on the data dictionary's descriptions. Skewed numerical features were log-transformed, and categorical features were converted to a numerical format using one-hot encoding. New features, such as TotalSF, were created to capture combined effects.

- Modeling: Several advanced regression models were trained and evaluated using 5-fold cross-validation, including Ridge, Lasso, XGBoost, and LightGBM. The final model selected for deployment was XGBoost, which achieved a cross-validated RMSE of approximately 0.12.

- Interactive Application: The trained XGBoost model was saved and served via an interactive web application built with Streamlit, allowing users to get real-time price predictions by adjusting key house features.

## Project Structure
- House_Price_Prediction_Notebook.ipynb: The main Jupyter Notebook containing the complete analysis, from EDA to model training and submission file generation.

- app.py: A Python script that creates an interactive web application using Streamlit to serve the trained XGBoost model.

- house_price_model.pkl: The saved (pickled) final XGBoost model, ready for use in the app.

- model_columns.pkl: A saved list of the feature columns the model was trained on.

- submission.csv: The final prediction file generated for submission to the Kaggle competition.

## How to Run
1. Setup: Clone the repository and install the required packages:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib streamlit

2. Run the Notebook: Open and run the House_Price_Prediction_Notebook.ipynb in JupyterLab to perform the analysis and generate the model files.

3. Launch the App: In your terminal, run the following command:
streamlit run app.py

## Tech Stack
- Languages & Libraries: Python, Pandas, NumPy, Scikit-learn

- Modeling: XGBoost, LightGBM, Ridge, Lasso

- Visualization: Matplotlib, Seaborn

- Deployment: Streamlit

- Tools: JupyterLab, Git, GitHub
