# Hands Detection using Mediapipe

This project uses MediaPipe and OpenCV to detect and track hand landmarks in static images and webcam input.
![hands-det](https://github.com/user-attachments/assets/d156afa8-2d8f-4a62-8900-3e3625d8a45e)

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe

## How to Use

### For Static Images
- Add image file paths to the IMAGE_FILES list in the script.
- Run the script.
- Annotated images will be saved in the /tmp directory.

### For Webcam Input
- Connect your webcam.
- Run the script.
- Press the Esc key to exit.

### Features
- Detects up to 2 hands.
- Works with both static images and real-time webcam input.
