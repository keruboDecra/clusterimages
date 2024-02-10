import streamlit as st
import os
import numpy as np
from sklearn.cluster import KMeans
import pickle
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
with open("/best_model.pkl", "rb") as f:
    clusters = pickle.load(f)

# Function to crawl images from a directory
def crawl_images(path):
    images = []
    imageNames = []
    with os.scandir(path) as files:
        for file in files:
            with os.scandir(file) as rooms:
                for imgs in rooms:
                    if imgs.name.endswith('.jpg'):
                        file_path = path + "/" + file.name + "/" + imgs.name
                        imageNames.append(file_path)
                        img = cv2.imread(file_path)
                        try:
                            img = cv2.resize(img, (32, 32))
                            img = img.astype(np.float32)
                        except:
                            break
                        images.append(img)
    return np.array(images), np.array(imageNames)

# Function to process images from a zip file
def process_zip_file(zip_file):
    # Add code to extract and process images from the zip file
    pass

# Main Streamlit app
def main():
    st.title("Image Clustering App")

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

    if st.button("Train and Cluster"):
        kmeans = KMeans(n_clusters=14, random_state=22)
        clusters = kmeans.fit(train_images)
        
        # Save the trained model
        with open(path + "/best_model.pkl", "wb") as f:
            pickle.dump(clusters, f)

        # Display cluster visualization (you can customize this part)
        st.subheader("Cluster Visualization")
        for cluster_id in range(14):
            view_cluster(cluster_id)

# Function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    plt.figure(figsize=(25, 25))
    files = groups[cluster]
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to +30")
        files = files[:29]
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

if __name__ == "__main__":
    main()
