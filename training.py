import cv2
import os
import numpy as np

wajahDir = "datawajah"
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_ids = {}
current_id = 0

for nama in os.listdir(wajahDir):
    path_folder = os.path.join(wajahDir, nama)
    if not os.path.isdir(path_folder):
        continue

    label_ids[current_id] = nama
    for filename in os.listdir(path_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(path_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_id)
    current_id += 1

print("Training model...")
recognizer.train(faces, np.array(labels))
os.makedirs("Dataset", exist_ok=True)
recognizer.save("Dataset/training.xml")
np.save("Dataset/labels.npy", label_ids)
print("Training selesai. Model dan label disimpan.")
