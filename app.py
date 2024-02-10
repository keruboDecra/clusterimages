import streamlit as st
import os
import zipfile
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import joblib

# Load the trained model
model_path = "bestmodel.joblib"
clusters = joblib.load(model_path)

# Function to extract images from a zip file
def extract_images(zip_path):
    images = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            with zip_ref.open(file_info) as file:
                img = Image.open(file)
                img = img.resize((32, 32))
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
    return np.array(images)

# Streamlit app
def main():
    st.title("Image Clustering App")

    # Upload zip file
    uploaded_file = st.file_uploader("Upload a zip file", type="zip")

    if uploaded_file is not None:
        # Extract images from the zip file
        images = extract_images(uploaded_file)

        # Predict clusters for the uploaded images
        uploaded_clusters = clusters.predict(images)

        # Display the clusters
        unique_clusters = np.unique(uploaded_clusters)
        for cluster in unique_clusters:
            st.subheader(f"Cluster {cluster}")
            cluster_images = images[uploaded_clusters == cluster]
            for img_array in cluster_images:
                st.image(img_array, caption='Uploaded Image', use_column_width=True)

if __name__ == "__main__":
    main()
