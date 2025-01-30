import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# API endpoint
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

uploaded_file = st.file_uploader(
    "Choose a CSV file for bulk prediction - Upload the file and click on Predict",
    type="csv",
)

user_input = st.text_input("Enter text and click on Predict", "")

if st.button("Predict"):
    try:
        if uploaded_file is not None:
            # Bulk prediction
            file = {"file": uploaded_file.getvalue()}
            response = requests.post(prediction_endpoint, files={"file": file})
            
            if response.status_code == 200:
                response_bytes = BytesIO(response.content)
                response_df = pd.read_csv(response_bytes)
                st.write("Prediction Results:")
                st.dataframe(response_df)

                st.download_button(
                    label="Download Predictions",
                    data=response_bytes,
                    file_name="Predictions.csv",
                )
            else:
                st.error(f"Error: {response.json()['error']}")

        elif user_input.strip():
            # Single text prediction
            response = requests.post(prediction_endpoint, data={"text": user_input})
            if response.status_code == 200:
                st.write(f"Predicted sentiment: {response.json()['prediction']}")
            else:
                st.error(f"Error: {response.json()['error']}")
        else:
            st.warning("Please provide a valid text or file.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
