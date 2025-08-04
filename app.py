# CODE FOR: Streamlit Web App (app.py)
# To run: streamlit run app.py

import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore

# --- LOAD SAVED MODEL AND COLUMNS ---

try:
    model = joblib.load('house_price_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run the notebook first to generate 'house_price_model.pkl' and 'model_columns.pkl'.")
    st.stop()

# --- APP LAYOUT ---
st.set_page_config(page_title="Ames House Price Predictor", layout="wide")
st.title('🏡 Ames, Iowa House Price Predictor')
st.write("""
This app uses a pre-trained XGBoost regression model to predict the sale price of a house based on its key features. 
Adjust the sliders and inputs on the left to get a real-time price prediction!
""")

# --- USER INPUTS IN THE SIDEBAR ---
st.sidebar.header('House Features')

# These were identified during the Exploratory Data Analysis (EDA) in the notebook.
overall_qual = st.sidebar.slider('Overall Quality (1-10)', 1, 10, 5)
gr_liv_area = st.sidebar.number_input('Above Grade Living Area (sq ft)', min_value=500, max_value=5000, value=1500)
year_built = st.sidebar.slider('Year Built', 1870, 2010, 2000)
total_bsmt_sf = st.sidebar.number_input('Total Basement Area (sq ft)', min_value=0, max_value=6000, value=1000)
garage_cars = st.sidebar.slider('Garage Capacity (cars)', 0, 4, 2)
full_bath = st.sidebar.slider('Full Bathrooms', 0, 4, 2)

neighborhoods = ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer']
neighborhood = st.sidebar.selectbox('Neighborhood', options=neighborhoods)

# --- PREDICTION LOGIC ---
def predict_price():
    # 1. Create a DataFrame from the user's input.
    # We start with a DataFrame of zeros that has the exact same columns as the data the model was trained on.
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0

    # 2. Populate the DataFrame with the user's selections from the sidebar.
    neighborhood_col = f'Neighborhood_{neighborhood}'
    if neighborhood_col in model_columns:
        input_df.at[0, neighborhood_col] = 1

    # numerical features.
    input_df.at[0, 'OverallQual'] = overall_qual
    input_df.at[0, 'GrLivArea'] = gr_liv_area
    input_df.at[0, 'YearBuilt'] = year_built
    input_df.at[0, 'TotalBsmtSF'] = total_bsmt_sf
    input_df.at[0, 'GarageCars'] = garage_cars
    input_df.at[0, 'FullBath'] = full_bath
    
    # 3. Apply the same log transformations as in the training notebook.
    for feat in ['GrLivArea', 'TotalBsmtSF']:
        if feat in input_df.columns and input_df.at[0, feat] > 0:
            input_df.at[0, feat] = np.log1p(input_df.at[0, feat])

    # 4. Make a prediction using the loaded model.
    prediction_log = model.predict(input_df)[0]
    
    # 5. Reverse the log transformation to get the final, human-readable price.
    prediction = np.expm1(prediction_log)
    
    return prediction

# --- DISPLAY PREDICTION ---
# When the user clicks the button, the prediction logic is triggered.
if st.sidebar.button('Predict Price', type="primary"):
    prediction = predict_price()

    st.header('Predicted Sale Price')
    st.subheader(f'**${prediction:,.2f}**')
    st.balloons()
else:
    st.info("Adjust the features in the sidebar and click 'Predict Price' to see the result.")

