import streamlit as st
import os
import numpy as np
from sklearn.cluster import KMeans
import joblib  # Import joblib
from PIL import Image
from zipfile import ZipFile
import matplotlib.pyplot as plt

# Function to crawl images from a directory
def crawl_images(path):
    images = []
    imageNames = []
    with os.scandir(path) as files:
        for file in files:
            with os.scandir(file) as rooms:
                for imgs in rooms:
                    if imgs.name.endswith('.jpg'):
                        file_path = os.path.join(path, file.name, imgs.name)
                        imageNames.append(file_path)
                        img = Image.open(file_path)
                        img = img.resize((32, 32))
                        img = np.array(img)
                        images.append(img)
    return np.array(images), np.array(imageNames)

# Function to process images from a zip file
def process_zip_file(zip_file):
    images = []
    imageNames = []
    with ZipFile(zip_file) as z:
        for file_info in z.infolist():
            if file_info.filename.endswith('.jpg'):
                with z.open(file_info.filename) as file:
                    img = Image.open(file)
                    img = img.resize((32, 32))
                    img = np.array(img)
                    images.append(img)
                    imageNames.append(file_info.filename)
    return np.array(images), np.array(imageNames)

# Function that lets you view a cluster (based on identifier)
def view_cluster(cluster, groups):
    plt.figure(figsize=(25, 25))
    files = groups[cluster]
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to +30")
        files = files[:29]
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1)
        img = Image.open(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

# Main Streamlit app
def main():
    st.title("Image Clustering App")

    # Initialize train_images and groups
    train_images, train_labels = None, None
    groups = None

    # Sidebar: Input method selection
    input_method = st.sidebar.radio("Select Input Method:", ("Web Crawl", "Upload ZIP"))

    if input_method == "Web Crawl":
        st.sidebar.header("Web Crawl Settings")
        crawl_path = st.sidebar.text_input("Enter the directory path for web crawl:")
        if st.sidebar.button("Crawl Images"):
            train_images, train_labels = crawl_images(crawl_path)

    elif input_method == "Upload ZIP":
        st.sidebar.header("ZIP Upload Settings")
        uploaded_zip = st.sidebar.file_uploader("Upload ZIP file", type="zip")
        if uploaded_zip is not None:
            if st.sidebar.button("Process ZIP"):
                train_images, train_labels = process_zip_file(uploaded_zip)

    if train_images is not None and st.button("Train Clustering Model"):
        kmeans = KMeans(n_clusters=14, random_state=22)
        clusters = kmeans.fit(train_images)

        # Save the trained model using joblib
        joblib.dump(clusters, "best_model.joblib")

        # Create or update 'groups' based on clustering results
        groups = {}
        for file, cluster in zip(train_labels, clusters.labels_):
            if cluster not in groups.keys():
                groups[cluster] = []
                groups[cluster].append(file)
            else:
                groups[cluster].append(file)

        st.success("Training and Clustering Successful!")

    if groups is not None:
        st.subheader("Cluster Visualization")
        cluster_id = st.slider("Select Cluster:", min_value=0, max_value=13, step=1)
        view_cluster(cluster_id, groups)

if __name__ == "__main__":
    main()
