import cv2
import numpy as np

# Pilih model
print("Pilih model face recognition:")
print("1. LBPH")
print("2. Eigenfaces")
mode = input("Masukkan pilihan (1 atau 2): ")

if mode == "1":
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Dataset/lbph_model.xml")
    model_name = "LBPH"
elif mode == "2":
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.read("Dataset/eigen_model.xml")
    model_name = "Eigenfaces"
else:
    print("Pilihan tidak valid.")
    exit()

# Load label
label_ids = np.load("Dataset/labels.npy", allow_pickle=True).item()
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

print(f"Model digunakan: {model_name}")

while True:
    ret, frame = cam.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = face_detector.detectMultiScale(abu, 1.3, 5)

    for (x, y, w, h) in wajah:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # ✅ Resize agar ukuran pas untuk model Eigenfaces
        wajah_roi = cv2.resize(abu[y:y+h, x:x+w], (200, 200))
        id, conf = recognizer.predict(wajah_roi)

        nama = label_ids.get(id, "Unknown")
        text = f"{nama} ({round(conf, 2)})"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    k = cv2.waitKey(1)
    if k == 27 or k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
