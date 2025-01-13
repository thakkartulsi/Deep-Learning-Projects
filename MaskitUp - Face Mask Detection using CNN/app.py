import os
import numpy as np
import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Define the upload folder 
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model('model/face_mask_model.h5')  

# Max file size for image upload (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve the uploaded image file
        image_file = request.files['image']

        # Check if the file is a valid image (PNG, JPG, JPEG)
        if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template('index.html', prediction="Invalid file type. Please upload a valid image (PNG, JPG, JPEG).")
        
        # Check the file size
        image_file.seek(0, os.SEEK_END)  # Move the cursor to the end to get file size
        file_size = image_file.tell()
        image_file.seek(0)  # Reset the cursor back to the start
        if file_size > MAX_FILE_SIZE:
            return render_template('index.html', prediction="File is too large. Please upload an image smaller than 10MB.")
        
        # Save the image to the uploads folder with a secure filename
        image_filename = os.path.join(UPLOAD_FOLDER, secure_filename(image_file.filename))
        image_file.save(image_filename)

        # Read the image using OpenCV
        img = cv2.imread(image_filename)
        
        # Check if the image is loaded correctly
        if img is None:
            return render_template('index.html', prediction="Failed to load image. Please upload a valid image.")
        
        # Preprocess the image (resize, normalize, etc.)
        img = cv2.resize(img, (128, 128))  # Resize to match model input size
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize the image (if necessary based on training)

        # Make a prediction
        prediction = model.predict(img)

        # Print the raw prediction output for debugging
        print(f"Prediction output: {prediction}")

        predicted_class = prediction[0][0]  
        
        # Use argmax or thresholding for prediction
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class with max probability

        # Map the predicted class index to the corresponding label
        if predicted_class == 0:
            mask_class = 'No Mask'
        else:
            mask_class = 'With Mask'
            
        # Pass prediction and image filename to the template
        return render_template('index.html', prediction=mask_class, image_filename=image_file.filename)

if __name__ == '__main__':
    app.run(debug=True)
