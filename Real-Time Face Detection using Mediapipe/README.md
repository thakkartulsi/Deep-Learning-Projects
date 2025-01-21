# Face Detection with MediaPipe and OpenCV

This project demonstrates how to perform face detection using MediaPipe's Face Detection solution and OpenCV. 

It includes functionality for processing both static images and real-time webcam input.

![fd_using_mp](https://github.com/user-attachments/assets/17f60364-02a6-4d62-9566-e474e67ba3e7)


## Requirements

Before running the code, ensure you have the following libraries installed:

- Python 3.7 or higher
- OpenCV
- MediaPipe

  ## Features

1. **Static Image Processing**:
   - Detects faces in a list of static image files.
   - Annotates the images with bounding boxes and key points (e.g., nose tip).
   - Saves the annotated images to the `/tmp/` directory.

2. **Real-Time Webcam Detection**:
   - Captures video from the webcam and performs face detection in real time.
   - Displays annotated frames with bounding boxes and key points.
   - Provides an option to exit by pressing the `Esc` key.
