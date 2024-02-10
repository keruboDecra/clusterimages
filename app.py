import streamlit as st
import os
import zipfile
import requests
import numpy as np
import joblib
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

# Load the trained model from the saved file using joblib
model_path = "path/to/bestmodel.joblib"  # Replace with your actual path
clusters = joblib.load(model_path)

# function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    plt.figure(figsize=(25, 25))
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to +30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

# Streamlit app
def main():
    st.title("Image Clustering Web App")
    
    # Provide a link to download the sample zip file
    st.markdown("You can use the provided sample zip file for testing. [Download Sample Zip File](https://drive.google.com/uc?export=download&id=11RF8O9CSsqSULkYI8fuOvrYviECnksa-)")
    
    uploaded_file = st.file_uploader("Upload a zip file containing images", type="zip")

    if uploaded_file is not None:
        st.text("Processing uploaded zip file...")

        # Create the temporary directory if it doesn't exist
        zip_path = "uploads"
        os.makedirs(zip_path, exist_ok=True)

        with open(os.path.join(zip_path, "uploaded.zip"), "wb") as f:
            f.write(uploaded_file.getvalue())

        with zipfile.ZipFile(os.path.join(zip_path, "uploaded.zip"), "r") as zip_ref:
            zip_ref.extractall(zip_path)

        # Process the images
        images = []
        image_names = []
        for root, dirs, files in os.walk(zip_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    image_names.append(file_path)

                    try:
                        img = Image.open(file_path)
                        img = img.resize((32, 32))
                        img_array = np.array(img, dtype=np.float32)
                        images.append(img_array)
                    except UnidentifiedImageError as e:
                        print(f"Error processing image {file_path}: {e}")
                    except Exception as e:
                        print(f"Unexpected error processing image {file_path}: {e}")
                    finally:
                        # Close the image file to avoid potential issues
                        img.close()

        data = np.array(images)
        data = data / 255.0
        reshaped_data = data.reshape(len(data), -1)

        # Predict clusters for the uploaded images
        uploaded_clusters = clusters.predict(reshaped_data)

        # Display cluster information
        unique_clusters = np.unique(uploaded_clusters)
        for cluster in unique_clusters:
            st.subheader(f"Cluster {cluster}")
            cluster_files = [image_names[i] for i, c in enumerate(uploaded_clusters) if c == cluster]
            st.image(cluster_files, width=100, caption=[f"Image {i+1}" for i in range(len(cluster_files))])

if __name__ == "__main__":
    main()
