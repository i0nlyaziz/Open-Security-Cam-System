import cv2
import torch
import time
from datetime import datetime
from ultralytics import YOLO
from facenet_pytorch import MTCNN
from database import create, fill
from face_system import compare_face
from telegram_alert import send_alert, send_photo

# Create Database
create()

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load Models
detect = YOLO("weights/yolo26n.engine")
face = MTCNN(keep_all=True, device=device)

# Settings
threshold = 0.70
face_confidence = 0.90
unknown_duration = 5
alert_cooldown = 10

# Track Unknown Persons
unknown_since = {}
alerted_ids = set()
last_unknown_alert = 0

# Open Webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Cannot open webcam")
    raise SystemExit

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    results = detect.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    current_ids = set()

    for result in results:
        for box in result.boxes:
            # Person Detection
            class_id = int(box.cls[0])
            if class_id != 0:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Keep Coordinates Inside Frame
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Tracking ID
            person_id = int(box.id[0]) if box.id is not None else -1
            if person_id >= 0:
                current_ids.add(person_id)

            # Default Label
            label = "Face Not Found"
            color = (0, 165, 255)

            # Crop Person
            person = frame[y1:y2, x1:x2]
            if person.size != 0:
                # Detect Face
                person_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                face_boxes, probabilities = face.detect(person_rgb)

                if face_boxes is not None and probabilities is not None:
                    for face_box, probability in zip(face_boxes, probabilities):
                        if probability is None or probability < face_confidence:
                            continue

                        fx1, fy1, fx2, fy2 = map(int, face_box)
                        ph, pw = person.shape[:2]
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(pw, fx2), min(ph, fy2)
                        if fx2 <= fx1 or fy2 <= fy1:
                            continue

                        # Crop Face Only
                        face_crop = person[fy1:fy2, fx1:fx2]
                        if face_crop.size == 0:
                            continue

                        # Compare With Registered Face
                        similarity = compare_face(face_crop)
                        if similarity is None:
                            label, color = "Unverified", (0, 165, 255)
                        elif similarity >= threshold:
                            label, color = "Owner", (0, 255, 0)
                        else:
                            label, color = "Unknown", (0, 0, 255)

                        # Draw Face Box
                        ax1, ay1 = x1 + fx1, y1 + fy1
                        ax2, ay2 = x1 + fx2, y1 + fy2
                        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), color, 2)
                        cv2.putText(frame, label, (ax1, max(ay1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # Unknown Person Timer
                        now = time.time()
                        if label == "Unknown" and person_id >= 0:
                            if person_id not in unknown_since:
                                unknown_since[person_id] = now

                            duration = now - unknown_since[person_id]
                            remaining = max(0, unknown_duration - duration)
                            cv2.putText(frame, f"Alert in: {remaining:.1f}s", (x1, min(y2 + 25, h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                            # Send Alert After 5 Seconds
                            can_alert = now - last_unknown_alert >= alert_cooldown
                            if duration >= unknown_duration and person_id not in alerted_ids and can_alert:
                                date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                image_path = "unknown_person.jpg"
                                cv2.imwrite(image_path, frame)

                                try:
                                    send_alert(
                                        "Unknown person detected!\n"
                                        f"Confidence: {confidence:.2f}\n"
                                        f"Face Similarity: {similarity:.2f}\n"
                                        f"Time: {date}"
                                    )
                                    send_photo(image_path)
                                    fill(date, confidence)
                                    alerted_ids.add(person_id)
                                    last_unknown_alert = now
                                    print("Unknown person alert sent:", date)
                                except Exception as error:
                                    print("Alert error:", error)
                        else:
                            # Reset Timer If No Longer Unknown
                            unknown_since.pop(person_id, None)
                        break

            # Draw Person Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Person {person_id} - {label}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Remove Timers For Tracks No Longer Visible
    for track_id in list(unknown_since.keys()):
        if track_id not in current_ids:
            unknown_since.pop(track_id, None)

    # Display Camera
    cv2.imshow("OpenGuard System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release Resources
cam.release()
cv2.destroyAllWindows()