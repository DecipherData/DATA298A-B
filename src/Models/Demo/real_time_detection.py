import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO(r'C:\Users\bhati\DATA298-FinalProject\YOLO\runs\detect\train\weights\last.pt')

# Set up video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or a video file path

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Run prediction on the frame
    results = model.predict(frame, stream=True)

    # for result in results:
    #     max_conf_class = None
    #     max_conf_score = -1
    #     boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
    #     for box in boxes:  # Iterate over boxes
    #         r = box.xyxy[0].astype(int)  # Get corner points as int
    #         class_id = int(box.cls[0])  # Get class ID
    #         class_name = model.names[class_id]  # Get class name using the class ID
    #         print(f"Class: {class_name}, Box: {r}")  # Print class name and box coordinates
    #         cv2.rectangle(frame, r[:2], r[2:], (0, 255, 0), 2)  # Draw boxes on the image

    for result in results:
        boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
        for box in boxes:  # Iterate over boxes
            r = box.xyxy[0].astype(int)  # Get corner points as int
            class_id = int(box.cls[0])  # Get class ID
            class_name = model.names[class_id]  # Get class name using the class ID
            confidence = box.conf[0]  # Get confidence score
            print(f"Class: {class_name}, Confidence: {confidence:.2f}, Box: {r}")  # Print class name, confidence score, and box coordinates
            cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)  # Draw boxes on the image
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)  # Display class name and confidence score

    # for result in results:
    #     # Initialize variables to store the class with maximum confidence and its corresponding score
    #     max_conf_class = None
    #     max_conf_score = -1
    #
    #     # Iterate through each detection box
    #     for i in range(len(result.boxes.cls)):
    #         # Get the class index and confidence score for the current box
    #         class_idx = int(result.boxes.cls[i].item())  # Extract the scalar value from the tensor
    #         confidence = float(result.boxes.conf[i].item())  # Extract the scalar value from the tensor
    #
    #         # Check if the current box has higher confidence than the previous maximum
    #         if confidence > max_conf_score:
    #             max_conf_score = confidence
    #             max_conf_class = result.names[class_idx]
    #
    #     # Print the class with the maximum confidence score for the current image
    #     print("Image Class with maximum confidence score:", max_conf_class)
    #     print("Maximum confidence score:", max_conf_score)
    #
    # Display the frame
    cv2.imshow('Real-time Detection', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()


# from ultralytics import YOLO
# import torch
# model = YOLO(r'C:\Users\sonal\OneDrive\Documents\yolo detect\best.pt')
# for m in model.model.modules():
#     if isinstance(m, torch.nn.Conv2d):
#         weights = m.weight
#         print(weights)
#         break  # Stop after finding the first Conv2d layer

