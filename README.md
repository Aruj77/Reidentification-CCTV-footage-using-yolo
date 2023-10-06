# Person Re-Identification using YOLO and Feature Extraction

![image](https://github.com/Aruj77/Reidentification-CCTV-footage-using-yolo/assets/68498812/7ac6dc19-b9ec-4c20-a129-9468f8e3879e)


## Overview

This Python code demonstrates a simplified implementation of person re-identification in CCTV footage using YOLO (You Only Look Once) for person detection and deep learning-based feature extraction. The code can be used to identify and track individuals across frames in a video stream.

## Requirements

Make sure you have the following libraries installed:

- OpenCV (cv2)
- NumPy
- TensorFlow and Keras
- scikit-learn (for cosine similarity)

You can install these libraries using `pip` as described in the previous section.

## Implementation Steps

1. **Object Detection with YOLO**:
   - The code uses a pre-trained YOLO model to detect persons in each frame of the input video.

2. **Feature Extraction**:
   - Features are extracted from the detected person's bounding box using a pre-trained deep learning model (ResNet50 in this example).

3. **Feature Storage**:
   - Features and their corresponding person IDs are stored in a database for future re-identification.

4. **Re-Identification Algorithm**:
   - When a person is detected in a frame, their features are compared with the features stored in the database using cosine similarity.
   - The person ID with the highest similarity score is considered a match.

5. **Tracking and Temporal Consistency**:
   - The code uses basic frame-by-frame tracking to maintain the identity of persons across frames.
   - It handles cases where persons may briefly leave and reappear in the camera's view.

## Functions

### `extract_features(person_roi)`
   - Extracts deep features from the region of interest (ROI) containing a person.
   - Uses a pre-trained ResNet50 model for feature extraction.
   
### `reidentify_person(query_features)`
   - Compares features extracted from a detected person with features stored in the database.
   - Calculates cosine similarity to measure similarity between features.
   - Returns the ID of the most similar person in the database.

## Usage

1. Ensure that the required libraries are installed (OpenCV, NumPy, TensorFlow, Keras, and scikit-learn).

2. Load the YOLO model for person detection.

3. Configure the database to store features and person IDs.

4. Process the video frames, perform person detection, feature extraction, and re-identification.

5. Display the video stream with bounding boxes and person IDs.

6. Press 'q' to exit the application.

## Note

This is a simplified example for educational purposes. In practice, more sophisticated re-identification methods and a well-labeled dataset for training are required for accurate person re-identification in CCTV footage.

Feel free to customize and expand upon this code according to your specific project requirements.
