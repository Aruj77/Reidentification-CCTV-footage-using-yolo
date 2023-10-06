import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet("darknet\cfg\yolov3.weights", "darknet\cfg\yolov3.cfg")

# Load COCO class names
with open("darknet\data\coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Set the confidence threshold for detections
confidence_threshold = 0.5

# Initialize video capture (change 0 to your video file path if needed)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get image dimensions and construct a blob from the frame
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the YOLO network
    net.setInput(blob)

    # Forward pass to get detections
    detections = net.forward()

    # Loop over the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold and classes[class_id] == "person":
                # Calculate bounding box coordinates
                center_x, center_y, width, height = obj[:4] * np.array([width, height, width, height])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                w = int(width)
                h = int(height)

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
