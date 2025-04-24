import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import traceback
import tempfile
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model
from pycaret.classification import setup as cls_setup, compare_models as cls_compare_models, pull as cls_pull, save_model as cls_save_model
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
st.set_page_config(
    page_title="AutoML Model Trainer",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .highlight { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    .stButton>button { color: white; background-color: #4CAF50; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #45a049; transform: scale(1.05); }
    </style>
""", unsafe_allow_html=True)
def preprocess_text(text):
    """Preprocess text data by converting to lowercase and removing stopwords"""
    text = str(text).lower()
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text
def load_model_from_upload(uploaded_model, ml_type):
    """
    Load a machine learning model from an uploaded file (tabular).
    Writes the uploaded file to a temporary file (without extension) then renames it so that
    PyCaret's load_model (which appends ".pkl") can locate it.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="") as tmp:
            tmp.write(uploaded_model.read())
            tmp.flush()
            tmp_noext = tmp.name  # File without extension
        tmp_with_ext = tmp_noext + ".pkl"
        os.rename(tmp_noext, tmp_with_ext)
        
        if ml_type == "Regression":
            from pycaret.regression import load_model as reg_load_model
            model = reg_load_model(tmp_noext)
        else:
            from pycaret.classification import load_model as cls_load_model
            model = cls_load_model(tmp_noext)
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.error(traceback.format_exc())
        return None
def predict_with_model(model, input_df):
    """Make predictions with a tabular model with error handling"""
    try:
        prediction = model.predict(input_df)
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.error(traceback.format_exc())
        return None
#########################################
# Utility Functions for Image Data
#########################################
def load_image_dataset(uploaded_file):
    """
    Load an image dataset from an Excel or CSV file.
    The file should have columns 'image_path' and 'label'.
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file)
    elif file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return df
def train_image_model(df):
    """
    Train an image classification model using transfer learning (MobileNetV2).
    Expects a dataframe with columns: "image_path" and "label".
    Splits data into training and validation sets, and saves the trained model and class list.
    """
    try:
        classes = df['label'].unique().tolist()
        
        # Ensure there are enough images per class.
        if len(df) < len(classes) * 2:
            st.error(f"Not enough images for training. At least {len(classes)*2} images are required, but only {len(df)} images were provided.")
            return None, None
        try:
            train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        except ValueError as ve:
            st.error("Error during train-test splitting: " + str(ve))
            return None, None
        target_size = (224, 224)
        batch_size = 16
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            rotation_range=20,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='image_path',
            y_col='label',
            target_size=target_size,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True
        )
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='image_path',
            y_col='label',
            target_size=target_size,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=False
        )
        # Build model using MobileNetV2 as base
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(len(classes), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        epochs = st.number_input("Number of Epochs", min_value=1, max_value=50, value=5)
        history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)
        # Save the trained model and class mapping
        model.save("best_model_image.h5")
        with open("best_model_image_classes.pkl", "wb") as f:
            pickle.dump(classes, f)
        return model, classes
    except Exception as e:
        st.error(f"Image model training error: {e}")
        st.error(traceback.format_exc())
        return None, None
def predict_image(model, classes, image_file):
    """Predict the class of an uploaded image using the trained image model."""
    try:
        target_size = (224, 224)
        img = load_img(image_file, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        predicted_class = classes[np.argmax(preds)]
        return predicted_class
    except Exception as e:
        st.error(f"Image prediction error: {e}")
        st.error(traceback.format_exc())
        return None
#########################################
# Main Streamlit Application
#########################################
def main():
    st.sidebar.title("🤖 AutoML Model Trainer")
    
    # For tabular data only
    ml_type = st.sidebar.radio(
        "Select Machine Learning Type", 
        ["Regression", "Classification"],
        help="Choose the type of machine learning task (for tabular data)"
    )
    
    # Choose dataset mode: Tabular or Image
    dataset_mode = st.sidebar.radio(
        "Select Dataset Mode", 
        ["Tabular", "Image"],
        help="Choose the type of dataset: Tabular (CSV/text) or Image"
    )
    # Navigation Menu
    choice = st.sidebar.radio(
        "Navigation Menu", 
        ["Dataset Upload", "Data Profiling", "Model Training", "Model Prediction", "Model Download"],
        help="Navigate through different stages of machine learning workflow"
    )
    # DataFrame to store tabular data or image CSV data
    df = pd.DataFrame()
    if dataset_mode == "Tabular":
        if os.path.exists('./dataset.csv'):
            try:
                df = pd.read_csv('./dataset.csv', index_col=None)
                df = df.dropna()
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    else:
        # For image datasets, use a CSV file with columns "image_path" and "label"
        if os.path.exists('./image_dataset.csv'):
            try:
                df = pd.read_csv('./image_dataset.csv', index_col=None)
            except Exception as e:
                st.error(f"Error loading image dataset CSV: {e}")
    #####################################
    # Dataset Upload Section
    #####################################
    if choice == "Dataset Upload":
        st.title("📤 Dataset Upload")
        if dataset_mode == "Tabular":
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type=["csv"], 
                help="Upload your machine learning dataset (CSV)"
            )
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file, index_col=None)
                    text_columns = df.select_dtypes(include=["object"]).columns
                    for col in text_columns:
                        df[col] = df[col].apply(preprocess_text)
                    df.to_csv('dataset.csv', index=None)
                    st.success("Dataset uploaded and preprocessed successfully!")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error processing dataset: {e}")
        else:
            # Option to upload an Excel or CSV file with image paths and labels
            st.write("Upload an Excel or CSV file with columns 'image_path' and 'label'.")
            uploaded_file = st.file_uploader("Upload Dataset File", type=["csv", "xlsx"], help="File should contain columns: image_path and label.")
            if uploaded_file is not None:
                df_images = load_image_dataset(uploaded_file)
                if df_images is not None:
                    st.success("Image dataset loaded successfully!")
                    st.dataframe(df_images)
                    # Optionally, display thumbnails for preview
                    for idx, row in df_images.iterrows():
                        st.image(row['image_path'], width=150, caption=f"Label: {row['label']}")
                    df_images.to_csv("image_dataset.csv", index=False)
    #####################################
    # Data Profiling Section
    #####################################
    elif choice == "Data Profiling":
        st.title("📊 Exploratory Data Analysis")
        if dataset_mode == "Tabular":
            if not df.empty:
                from ydata_profiling import ProfileReport
                from streamlit_pandas_profiling import st_profile_report
                profile = ProfileReport(df, title="Profiling Report")
                st_profile_report(profile)
            else:
                st.warning("Please upload a dataset first!")
        else:
            if not df.empty:
                st.write("### Uploaded Images")
                for idx, row in df.iterrows():
                    st.image(row['image_path'], width=150, caption=f"Label: {row['label']}")
                st.write("#### Label Distribution")
                st.bar_chart(df['label'].value_counts())
            else:
                st.warning("Please upload an image dataset first!")
    
    #####################################
    # Model Training Section
    #####################################
    elif choice == "Model Training":
        if dataset_mode == "Tabular":
            st.title("🧠 Model Training (Tabular Data)")
            if not df.empty:
                target_column = st.selectbox("Select Target Column", df.columns)
                if st.button("Train Models"):
                    try:
                        if ml_type == "Regression":
                            setup_function = reg_setup
                            compare_models = reg_compare_models
                            pull = reg_pull
                            save_model = reg_save_model
                        else:
                            setup_function = cls_setup
                            compare_models = cls_compare_models
                            pull = cls_pull
                            save_model = cls_save_model
                        setup_function(data=df, target=target_column)
                        best_model = compare_models()
                        save_model(best_model, 'best_model')
                        st.success("Model training completed!")
                        st.dataframe(pull())
                    except Exception as e:
                        st.error(f"Model training error: {e}")
            else:
                st.warning("Please upload a tabular dataset first!")
        else:
            st.title("🧠 Model Training (Image Data)")
            if not df.empty:
                if st.button("Train Image Classification Model"):
                    model, classes = train_image_model(df)
                    if model is not None:
                        st.success("Image model training completed!")
                        st.write("Classes:", classes)
            else:
                st.warning("Please upload an image dataset first!")
    
    #####################################
    # Model Prediction Section
    #####################################
    elif choice == "Model Prediction":
        if dataset_mode == "Tabular":
            st.title("🔮 Model Prediction (Tabular Data)")
            uploaded_model = st.file_uploader(
                "Upload Trained Model", 
                type=['.pkl'], 
                help="Upload your pre-trained machine learning model"
            )
            if uploaded_model is not None:
                model = load_model_from_upload(uploaded_model, ml_type)
                if model is not None:
                    st.subheader("Enter Prediction Data")
                    if os.path.exists('dataset.csv'):
                        original_df = pd.read_csv('dataset.csv')
                        target_column = original_df.columns[-1]
                        input_columns = [col for col in original_df.columns if col != target_column]
                        input_data = {}
                        for col in input_columns:
                            if original_df[col].dtype == 'object':
                                unique_values = original_df[col].unique()
                                input_data[col] = st.selectbox(f"Select {col}", unique_values)
                            elif original_df[col].dtype in ['int64', 'float64']:
                                min_val, max_val = original_df[col].min(), original_df[col].max()
                                input_data[col] = st.number_input(
                                    f"Enter {col}", 
                                    min_value=float(min_val), 
                                    max_value=float(max_val), 
                                    value=float((min_val + max_val) / 2)
                                )
                        if st.button("Predict"):
                            input_df = pd.DataFrame([input_data])
                            prediction = predict_with_model(model, input_df)
                            if prediction is not None:
                                st.success(f"Prediction: {prediction[0]}")
                    else:
                        st.warning("Dataset not found. Please upload a dataset first.")
        else:
            st.title("🔮 Model Prediction (Image Data)")
            uploaded_image = st.file_uploader(
                "Upload an Image for Prediction", 
                type=["jpg", "jpeg", "png"],
                help="Upload an image to predict its class"
            )
            if uploaded_image is not None:
                try:
                    # Load the trained image model and class mapping
                    model = load_model("best_model_image.h5")
                    with open("best_model_image_classes.pkl", "rb") as f:
                        classes = pickle.load(f)
                    predicted_class = predict_image(model, classes, uploaded_image)
                    if predicted_class is not None:
                        st.success(f"Predicted Class: {predicted_class}")
                except Exception as e:
                    st.error(f"Error during image prediction: {e}")
    elif choice == "Model Download":
        st.title("📥 Model Download")
        if dataset_mode == "Tabular":
            if os.path.exists('best_model.pkl'):
                with open('best_model.pkl', 'rb') as f:
                    st.download_button(
                        label="Download Trained Model", 
                        data=f, 
                        file_name="best_model.pkl",
                        help="Download your trained machine learning model"
                    )
            else:
                st.warning("No trained model available. Please train a model first.")
        else:
            if os.path.exists('best_model_image.h5'):
                with open('best_model_image.h5', 'rb') as f:
                    st.download_button(
                        label="Download Trained Image Model", 
                        data=f, 
                        file_name="best_model_image.h5",
                        help="Download your trained image classification model"
                    )
            else:
                st.warning("No trained image model available. Please train a model first.")
if __name__ == "__main__":
    main()