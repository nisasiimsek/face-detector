import cv2
from ultralytics import YOLO

# Bulduğun tam yol:
model = YOLO(r'C:\Users\niisa\OneDrive\Desktop\face_detector\face\runs\detect\face_full_train\weights\best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    results   = model(frame)
    annotated = results[0].plot()

    cv2.imshow('Face Detector', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
