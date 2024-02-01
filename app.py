# import operator  # If needed
import streamlit as st
from pycaret.classification import setup, compare_models, pull, save_model
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import pandas_profiling
from ydata_profiling import ProfileReport

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
    df=df.dropna()

with st.sidebar: 
    # st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Automatic Model Analyser & Trainer ")
    choice = st.radio("Navigation", ["Dataset Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build, train and explore your data.")

if choice == "Dataset Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

# if choice == "Profiling": 
#     st.title("Exploratory Data Analysis")
#     # Use st_pandas_profiling instead of st_profile_report
#     profile = ProfileReport(df, title="Profiling Report")
    
#     plt.figure(figsize=(8, 6))
#     sns.histplot(df['Age'], bins=20)  # Specify your actual column name
#     st.pyplot()

#     # Display the profiling report
#     st_profile_report(profile)
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    # Use ydata_profiling ProfileReport instead of st_pandas_profiling
    # profile = ProfileReport(df, title="Profiling Report")
    profile = pandas_profiling.ProfileReport(df, title="Profiling Report")

    # Display various plots using Seaborn and Matplotlib
    st.subheader("Visualising various plots:")

    # # Count Plot - Home Planet Distribution
    # plt.figure(figsize=(8, 6))
    # sns.countplot(x='HomePlanet', data=df)
    # plt.title('Count of Passengers from Each Home Planet')
    # plt.xlabel('Home Planet')
    # plt.ylabel('Count')
    # plt.xticks(rotation=45)
    # st.pyplot(plt.gcf())

    # # Box Plot - Age Distribution by Transported
    # plt.figure(figsize=(8, 6))
    # sns.boxplot(x='Transported', y='Age', data=df)
    # plt.title('Age Distribution by Transported')
    # plt.xlabel('Transported')
    # plt.ylabel('Age')
    # st.pyplot(plt.gcf())

    # # Bar Plot - VIP Status Distribution
    # plt.figure(figsize=(8, 6))
    # sns.barplot(x='VIP', y='Transported', data=df)
    # plt.title('VIP Status Distribution')
    # plt.xlabel('VIP')
    # plt.ylabel('Transported')
    # st.pyplot(plt.gcf())

    # # Scatter Plot - Shopping Mall vs. Room Service
    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(x='ShoppingMall', y='RoomService', hue='Transported', data=df)
    # plt.title('Shopping Mall vs. Room Service')
    # plt.xlabel('Shopping Mall')
    # plt.ylabel('Room Service')
    # st.pyplot(plt.gcf())

    # # Histogram - Age Distribution
    # plt.figure(figsize=(8, 6))
    # sns.histplot(df['Age'], bins=20)
    # plt.title('Age Distribution')
    # plt.xlabel('Age')
    # plt.ylabel('Count')
    # st.pyplot(plt.gcf())


    st_profile_report(profile)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        # Update compare_models function with appropriate parameters
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")