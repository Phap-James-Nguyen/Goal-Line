import cv2
from ultralytics import YOLO


###PROGRAM PARAMETERS####
modelName = "GoalLine_V1.pt" #model name
videoName = "video2.mp4" #video name, cam for live

thres = 0.6 # confidence thres for detection

drawLine = False #Booleans
writeVid = False

pt1 = (388, 842) # endpoint1 of line
pt2 = (384, 1277) # endpoint2 of line


# === Utility for side check ===
def is_goal(ball_pos, pt1, pt2,radius):
    bx, by = ball_pos
    bx = bx + radius #Rightmost point
    x1, y1 = pt1
    x2, y2 = pt2
    return (x2 - x1)*(by - y1) - (y2 - y1)*(bx - x1) > 0 #Vector Cross Product

# === Load Model ===
model = YOLO(f"Weights/{modelName}")

# === Load Video ===
videoInput = 0 if videoName == "cam" else f"../Videos/{videoName}"
cap = cv2.VideoCapture(videoInput)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Video Writer (optional) ===
if writeVid:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("Out_Vid/output.mp4", fourcc, fps, (width, height))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (640, 640))

    results = model(resized)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if class_id == 0 and conf > thres:
                x1, y1, x2, y2 = box
                x1 = int(x1 * orig_w / 640)
                y1 = int(y1 * orig_h / 640)
                x2 = int(x2 * orig_w / 640)
                y2 = int(y2 * orig_h / 640)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = int(0.5 * max(x2 - x1, y2 - y1))

                # Draw ball
                cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
                cv2.putText(frame, f"Ball: {conf:.2f}", (center_x - radius, center_y - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check for goal
                if is_goal((center_x, center_y), pt1, pt2,radius):
                    cv2.putText(frame, "GOAL!", (center_x, center_y + 40),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
                    print("GOAL!")
    # Draw goal line
    if drawLine:
        cv2.line(frame, pt1, pt2, (0, 0, 255), 3)

    if writeVid:
        out.write(frame)  # Optional save
    display = cv2.resize(frame, (orig_w // 2, orig_h // 2))
    cv2.imshow("Ball Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
