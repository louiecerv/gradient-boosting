#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

# Define the Streamlit app
def app():

    st.subheader('The task: Classify handwritten digits from 0 to 9 based on a given image.')
    text = """Dataset: MNIST - 70,000 images of handwritten digits (28x28 pixels), each labeled 
    with its corresponding digit (0-9).
    \nModels:
    \nK-Nearest Neighbors (KNN):
    \nEach image is represented as a 784-dimensional vector (28x28 pixels). 
    To classify a new image, its distance is measured to K nearest neighbors in the 
    training data. The majority class label among the neighbors is assigned to the new image.
    \nDecision Tree:
    \nA tree-like structure is built based on features (pixel intensities) of the images. 
    \nThe tree splits the data based on decision rules (e.g., "pixel intensity at 
    position X is greater than Y"). The new image is navigated through the tree based on 
    its features, reaching a leaf node representing the predicted digit class.
    \nRandom Forest:
    \nAn ensemble of multiple decision trees are built, each trained on a random subset of 
    features (pixels) and a random subset of data.
    \nTo classify a new image, it is passed through each decision tree, and the majority class 
    label from all predictions is assigned."""
    st.write(text)

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
        
    st.subheader('First 25 images in the MNIST dataset') 

    # Get the first 25 images and reshape them to 28x28 pixels
    train_images = np.array(X_train)
    train_labels = np.array(y_train)
    images = train_images[:25].reshape(-1, 28, 28)

    # Create a progress bar object
    progress_bar = st.progress(0, text="Generating digit images please wait...")
    
    # Create a 5x5 grid of subplots
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    # Plot each image on a separate subplot
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i], cmap=plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Digit: {train_labels[i]}")
    # Show the plot
    plt.tight_layout()
    st.pyplot(fig)

    for i in range(100):
        # Update progress bar value
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    
    st.success("Image loading completed!")

#run the app
if __name__ == "__main__":
    app()
