# Face Mesh using Mediapipe

This project demonstrates how to use the MediaPipe Face Mesh solution to detect and annotate facial landmarks in both static images and live webcam input.

![Face Mesh using Mediapipe](https://github.com/user-attachments/assets/1842a4e8-3627-4bc2-bc0a-e61fc0b688e4)

## Usage

### Static Images
1. Add image file paths to the `IMAGE_FILES` list in the script.
2. Run the script to process the images and save annotated outputs to the `/tmp/` directory.

### Webcam Input
1. Connect a webcam to your system.
2. Run the script and press `q` to quit the webcam preview.

## Prerequisites
Ensure you have the following installed:

- Python 3.7 or later
- OpenCV
- MediaPipe
