import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import joblib
from PIL import Image
import zipfile
import matplotlib.pyplot as plt

# Function to process images from a zip file
def process_zip_file(zip_file):
    images = []
    imageNames = []
    with st.spinner("Processing ZIP file..."):
        with zipfile.ZipFile(zip_file) as z:
            for file_info in z.infolist():
                if file_info.filename.endswith('.jpg'):
                    imageNames.append(file_info.filename)
                    img = Image.open(z.open(file_info.filename))
                    img = img.resize((16, 16))  # Resize to 16x16 pixels
                    img = np.array(img)
                    images.append(img)
                    st.image(img, caption=file_info.filename, use_column_width=True)

    if not images:  # Check if no valid images were processed
        st.error("No valid images found in the ZIP file.")
        return None, None

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
        img = img.resize((16, 16))  # Resize to 16x16 pixels
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
    with st.sidebar.form(key='upload_form'):
        st.sidebar.header("ZIP Upload Settings")
        uploaded_zip = st.file_uploader("Upload ZIP file", type="zip")
        if st.form_submit_button("Process ZIP"):
            train_images, train_labels = process_zip_file(uploaded_zip)

    # Train and Cluster button in the main section
    with st.form(key='train_cluster_form'):
        if train_images is not None and st.form_submit_button("Train and Cluster"):
            kmeans = KMeans(n_clusters=14, random_state=22)
            clusters = kmeans.fit(train_images)

            # Save the trained model using joblib
            joblib.dump(clusters, "best_model.joblib")

            # Display cluster visualization (you can customize this part)
            st.subheader("Cluster Visualization")

            # Create or update 'groups' based on clustering results
            groups = {}
            for file, cluster in zip(train_labels, clusters.labels_):
                if cluster not in groups.keys():
                    groups[cluster] = []
                    groups[cluster].append(file)
                else:
                    groups[cluster].append(file)

            for cluster_id in range(14):
                view_cluster(cluster_id, groups)

if __name__ == "__main__":
    main()
