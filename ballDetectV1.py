import cv2
from ultralytics import YOLO

framesProcessed = 0.0
framesWBall = 0.0
# Load the trained YOLOv8 model
model = YOLO("yolov8n.pt")
print(model.names)  # Prints a dictionary mapping class IDs to class names

# Open the video file
cap = cv2.VideoCapture("video3.mp4")

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 640))
    # Perform object detection
    results = model(frame)

    # Iterate through the results and draw bounding boxes
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if class_id == 32:  # Assuming 0 is the ball class
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Ball: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                framesWBall += 1
    # Show the frame with detections
    cv2.imshow("Ball Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    framesProcessed+=1
    if(framesProcessed == 45):
        break;
 
accuracy = (framesWBall/ framesProcessed)*100
print(f"{framesWBall} Frames with ball and {framesProcessed} Frames Processed") 
print(f"Estimated Accuracy = {accuracy}")
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
