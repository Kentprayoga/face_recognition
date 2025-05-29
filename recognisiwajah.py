import cv2
import numpy as np
import os

# Load model dan label
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Dataset/training.xml")
label_ids = np.load("Dataset/labels.npy", allow_pickle=True).item()

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = face_detector.detectMultiScale(abu, 1.3, 5)

    for (x, y, w, h) in wajah:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, conf = recognizer.predict(abu[y:y+h, x:x+w])
        nama = label_ids.get(id, "Unknown")
        text = f"{nama} ({round(conf, 2)})"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    k = cv2.waitKey(1)
    if k == 27 or k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
