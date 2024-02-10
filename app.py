import streamlit as st
import os
import numpy as np
from sklearn.cluster import KMeans
import joblib  # Import joblib
from PIL import Image
from zipfile import ZipFile

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
def view_cluster(cluster):
    # Implement the load_img function if not already done
    # You can replace it with your actual image loading logic
    pass

# Main Streamlit app
def main():
    st.title("Image Clustering App")

    # Initialize train_images
    train_images = None

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

    if train_images is not None and st.button("Train and Cluster"):
        kmeans = KMeans(n_clusters=14, random_state=22)
        clusters = kmeans.fit(train_images)
        
        # Save the trained model using joblib
        joblib.dump(clusters, "best_model.joblib")

        # Display cluster visualization (you can customize this part)
        st.subheader("Cluster Visualization")
        for cluster_id in range(14):
            view_cluster(cluster_id)

if __name__ == "__main__":
    main()
