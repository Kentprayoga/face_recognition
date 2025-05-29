import cv2
import os
import numpy as np

# Folder data wajah
wajahDir = "datawajah"
datasetDir = "Dataset"
os.makedirs(datasetDir, exist_ok=True)  # Buat folder Dataset kalau belum ada

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces = []
labels = []
label_ids = {}
current_id = 0

# Ambil data dari folder
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
            img = cv2.resize(img, (200, 200))  # Uniform size untuk semua algoritma
            faces.append(img)
            labels.append(current_id)
    current_id += 1

if len(faces) == 0:
    print("❌ Tidak ada data wajah ditemukan. Jalankan rekam_wajah.py dulu.")
    exit()

print("✅ Jumlah wajah yang dilatih:", len(faces))

# === Train LBPH ===
lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(faces, np.array(labels))
lbph.save(os.path.join(datasetDir, "lbph_model.xml"))
print("✔ LBPH model disimpan.")

# === Train EigenFace ===
eigen = cv2.face.EigenFaceRecognizer_create()
eigen.train(faces, np.array(labels))
eigen.save(os.path.join(datasetDir, "eigen_model.xml"))
print("✔ Eigenfaces model disimpan.")

# Simpan label ke file
np.save(os.path.join(datasetDir, "labels.npy"), label_ids)
print("✔ Label disimpan di labels.npy")

print("\n✅ Semua model selesai dilatih dan disimpan di folder Dataset/")
