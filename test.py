import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
import os
from ydata_profiling import ProfileReport
from pycaret.regression import setup as reg_setup, compare_models, pull, save_model
from pycaret.classification import setup as cls_setup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Corrected import for pydantic settings
from pydantic_settings import BaseSettings

ml_type = st.sidebar.radio("Choose ML Type", ["Regression", "Classification"])

if ml_type == "Regression":
    from pycaret.regression import setup, compare_models, pull, save_model
    setup_function = reg_setup
else:
    from pycaret.classification import setup, compare_models, pull, save_model
    setup_function = cls_setup

# Load or create an empty DataFrame
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)
    df = df.dropna()
else:
    df = pd.DataFrame()

with st.sidebar:
    st.title("Automatic Model Analyser & Trainer")
    choice = st.radio("Navigation", ["Dataset Upload", "Profiling", "Modelling", "Download", "Implementation Preview"])
    st.info("This project application helps you build, train, and analyse your data.")

# Function for text data preprocessing
def preprocess_text(text):
    text = text.lower()
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    
    return text

if choice == "Dataset Upload":
    st.title("Upload Your Dataset")
    file_csv = st.file_uploader("Upload CSV File", type=["csv"])
    
    if file_csv:
        df = pd.read_csv(file_csv, index_col=None)

        # Data Preprocessing based on data types
        if "text" in df.select_dtypes(include=["object"]).columns:
            # Apply text data preprocessing steps
            df["text"] = df["text"].apply(lambda x: preprocess_text(x))

        # Save the preprocessed dataset
        df.to_csv('dataset.csv', index=None)
        st.success("CSV Dataset uploaded and preprocessed successfully!")
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if not df.empty:
        profile = ProfileReport(df, title="Profiling Report")
        st.subheader("Visualising various plots:")
        st_profile_report(profile)
    else:
        st.warning("Please upload a dataset before proceeding to profiling.")

if choice == "Modelling":
    if not df.empty:
        st.title("Model Training")
        chosen_target = st.selectbox('Choose the Target Column', df.columns)

        if st.button('Run Modelling'):
            try:
                setup_function(data=df, target=chosen_target)
                setup_df = pull()
                st.subheader("Setup Information:")
                st.dataframe(setup_df)

                st.subheader("Comparing Models:")
                best_model = compare_models()
                compare_df = pull()
                st.dataframe(compare_df)

                save_model(best_model, 'best_model')
                st.success("Model training completed successfully!")
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")

if choice == "Download":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("No trained model available for download. Please run the model training first.")

def predict(model, input_df):  
    prediction = model.predict(input_df)
    return prediction

if choice == "Implementation Preview":
    if os.path.exists('best_model.pkl'):
        st.title("Implementation Preview")
        st.subheader("Under Development")
        st.warning("This feature is currently under development. Please check back later for updates.")
        st.info("We're working on integrating a user-friendly interface for making predictions. Stay tuned for updates!")
    else:
        st.warning("No trained model available for implementation preview. Please run the model training first.")
        st.info("Once you've trained a model, you'll be able to use this feature to make predictions. We're excited to bring this feature to life soon!")