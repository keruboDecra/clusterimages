import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import joblib
from tensorflow.keras.preprocessing.image import load_img

# Load the trained model from the saved file using joblib
clusters = joblib.load("bestmodel.joblib")

# Define the Streamlit app
def main():
    st.title("Image Clustering App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Preprocess the uploaded image
        img = Image.open(uploaded_file)
        img = img.resize((32, 32))
        img_array = np.array(img, dtype=np.float32) / 255.0
        reshaped_data = img_array.reshape(1, -1)

        # Predict the cluster for the uploaded image
        cluster = clusters.predict(reshaped_data)[0]

        # Display the original and clustered images
        st.image([img, view_cluster(cluster)], caption=["Original Image", f"Cluster {cluster}"])

# function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    files = groups[cluster]
    if len(files) > 30:
        files = files[:29]

    images = []
    for file in files:
        img = load_img(file)
        img = np.array(img)
        images.append(img)

    return np.hstack(images)

if __name__ == "__main__":
    main()
