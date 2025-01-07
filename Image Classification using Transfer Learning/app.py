import streamlit as st
from tensorflow.keras.applications import VGG19, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg, decode_predictions as decode_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet, decode_predictions as decode_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception, decode_predictions as decode_inception
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Function to load the VGG19 model with pre-trained weights
@st.cache_resource
def load_vgg19():
    model = VGG19(weights="imagenet")  # Load the VGG19 model with ImageNet weights
    return model

# Function to load the ResNet50 model with pre-trained weights
@st.cache_resource
def load_resnet50():
    model = ResNet50(weights="imagenet")  # Load the ResNet50 model with ImageNet weights
    return model

# Function to load the InceptionV3 model with pre-trained weights
@st.cache_resource
def load_inceptionv3():
    model = InceptionV3(weights="imagenet")  # Load the InceptionV3 model with ImageNet weights
    return model

# Preprocess image for VGG19
def preprocess_image_vgg(image):
    image = image.convert("RGB")  # Ensure image is in RGB format
    image = image.resize((224, 224))  # VGG19 requires 224x224 pixel images
    image_array = img_to_array(image)  # Convert image to numpy array
    image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions for batch size of 1
    image_array = preprocess_vgg(image_array)  # Apply VGG19-specific preprocessing
    return image_array

# Preprocess image for ResNet50
def preprocess_image_resnet(image):
    image = image.convert("RGB")  # Ensure image is in RGB format
    image = image.resize((224, 224))  # ResNet50 requires 224x224 pixel images
    image_array = img_to_array(image)  # Convert image to numpy array
    image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions for batch size of 1
    image_array = preprocess_resnet(image_array)  # Apply ResNet50-specific preprocessing
    return image_array

# Preprocess image for InceptionV3
def preprocess_image_inception(image):
    image = image.convert("RGB")  # Ensure image is in RGB format
    image = image.resize((299, 299))  # InceptionV3 requires 299x299 pixel images
    image_array = img_to_array(image)  # Convert image to numpy array
    image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions for batch size of 1
    image_array = preprocess_inception(image_array)  # Apply InceptionV3-specific preprocessing
    return image_array

# Predict the class using the selected model
def predict(image, model, preprocess_func, decode_func):
    """
    This function processes the input image, makes a prediction using the selected model,
    and decodes the top 3 predictions.
    """
    processed_image = preprocess_func(image)  # Preprocess the image using the specified function
    predictions = model.predict(processed_image)  # Get the model's predictions
    decoded_predictions = decode_func(predictions, top=3)[0]  # Decode the predictions to human-readable format
    return decoded_predictions

# Main function to run the Streamlit app
def main():
    # Set the page title and favicon
    st.set_page_config(
    page_title="Image Classification using Transfer Learning",  # Custom page title
    page_icon="C:/Users/Tulsi/Downloads/img_class_favicon.jpg",  # Custom favicon (JPEG image)
    layout="centered",  # You can use "centered" or "wide" layout
    initial_sidebar_state="expanded",  # Sidebar can be "expanded", "collapsed", or "auto"
    )
    
    st.title("Welcome to the Image Classifier!")
    st.write("Upload an image to classify it using Transfer Learning models")

    # Select a model from the tabs (VGG19, ResNet50, or InceptionV3)
    tab_titles = ["VGG19", "ResNet50", "InceptionV3"]
    selected_tab = st.selectbox("Choose a model:", tab_titles)
    
    # Show instructions
    st.write("Upload an image to classify using the selected model")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)  # Open the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)  # Display the uploaded image

        # Load the selected model and corresponding preprocessing function
        if selected_tab == "VGG19":
            model = load_vgg19()  # Load the VGG19 model
            preprocess_func = preprocess_image_vgg  # Use the VGG19 preprocessing function
            decode_func = decode_vgg  # Use the VGG19 decoding function

        elif selected_tab == "ResNet50":
            model = load_resnet50()  # Load the ResNet50 model
            preprocess_func = preprocess_image_resnet  # Use the ResNet50 preprocessing function
            decode_func = decode_resnet  # Use the ResNet50 decoding function

        elif selected_tab == "InceptionV3":
            model = load_inceptionv3()  # Load the InceptionV3 model
            preprocess_func = preprocess_image_inception  # Use the InceptionV3 preprocessing function
            decode_func = decode_inception  # Use the InceptionV3 decoding function

        with st.spinner("Identifying..."):  # Show spinner while the prediction is being made
            predictions = predict(image, model, preprocess_func, decode_func)  # Get predictions

        st.write("Top Predictions:")  # Show the prediction results
        for i, (imagenet_id, label, score) in enumerate(predictions):
            st.write(f"{i + 1}. **{label}** with a confidence of **{score * 100:.2f}%**")  # Display top predictions

if __name__ == "__main__":
    main()  # Run the Streamlit app











