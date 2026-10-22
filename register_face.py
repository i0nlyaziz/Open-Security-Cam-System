import cv2
from face_system import register_face

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow("Register Face", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imwrite("my_face.jpg", frame)
        if register_face("my_face.jpg"):
            print("Registration completed")
        break
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
