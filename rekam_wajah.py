import cv2
import os

# Input nama pengguna
nama = input("Masukkan nama pengguna: ")

# Buat folder simpan wajah
folder_wajah = os.path.join("datawajah", nama)
os.makedirs(folder_wajah, exist_ok=True)

# Inisialisasi kamera
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 0
while True:
    ret, frame = cam.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = face_detector.detectMultiScale(abu, 1.3, 5)

    for (x, y, w, h) in wajah:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.putText(frame, "Tekan SPACE untuk ambil foto, Q/ESC untuk keluar", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Ambil Data Wajah", frame)

    k = cv2.waitKey(1) & 0xFF

    if k == 27 or k == ord('q'):
        # ESC atau Q untuk keluar
        break
    elif k == 32:  # tombol SPACE ditekan
        if len(wajah) == 0:
            print("Wajah tidak terdeteksi, coba lagi...")
            continue
        for (x, y, w, h) in wajah:
            count += 1
            wajah_crop = abu[y:y+h, x:x+w]
            file_path = os.path.join(folder_wajah, f"user.{nama}.{count}.jpg")
            cv2.imwrite(file_path, wajah_crop)
            print(f"Foto {count} tersimpan.")
        if count >= 20:
            print("Mencapai batas 20 foto, proses selesai.")
            break

cam.release()
cv2.destroyAllWindows()
print(f"Pengambilan data selesai untuk {nama}")
