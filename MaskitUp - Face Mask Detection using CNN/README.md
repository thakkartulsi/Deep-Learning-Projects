This project involves a Face Mask Detection model built using Convolutional Neural Networks (CNN), deployed as a web application using Flask. 
The model detects whether a person in an image is wearing a face mask or not.

![demo_img](https://github.com/user-attachments/assets/82d5c81e-f118-48e4-b248-1d4ca6db5f80)

**Project Overview**
The system uses a pre-trained CNN model to classify images into two categories: With Mask and No Mask.
The user can upload images and the application will display whether the person in the image is wearing a mask or not.

**Requirements**
Before running the project, make sure you have the following installed:
Python 3.6+
Flask
TensorFlow/Keras
OpenCV
NumPy
werkzeug

**You can install the necessary dependencies using the following command:**
pip install -r requirements.txt

**Project Structure**

├── app.py                       # Flask app for the web application
├── model/                       # Directory for storing the trained model
│   └── face_mask_model.h5       # Pre-trained CNN model
├── uploads/                     # Directory to temporarily store uploaded images
├── templates/
│   └── index.html               # HTML template for the web interface
└── requirements.txt             # List of dependencies

**How It Works**

**Upload an Image:** Users can upload an image via the web interface. Supported formats include PNG, JPG, and JPEG.
**Image Preprocessing:** The uploaded image is resized to the dimensions expected by the model and normalized.
**Prediction:** The model predicts if the person in the image is wearing a mask or not. The prediction is displayed on the web page.
**Result Display:** After prediction, the result (either "With Mask" or "No Mask") is shown along with the uploaded image.


