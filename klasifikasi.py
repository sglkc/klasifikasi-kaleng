import sys, cv2
from analisis import analisis, tampilkan

# Fungsi klasifikasi dengan rasio optimal
def klasifikasi(file_path: str):

    ## Dapatkan rasio dari analisis piksel
    bg_px, citra, hasil = analisis(file_path, show=False, verbose=False)

    ## Konstanta kaleng optimal dengan tambahan error margin 0.5%
    optimal = 0.03767 + 0.005

    ## Jika rasio pada citra melebihi nilai optimal,
    ## maka dikatakan buruk, sebaliknya dikatakan baik
    if bg_px > optimal:
        klasifikasi = "BURUK"
    else:
        klasifikasi = "BAIK"

    cv2.putText(
        hasil, f"Klasifikasi: {klasifikasi}",
        (12, 52), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255)
    )

    cv2.putText(
        citra, f"Klasifikasi : {klasifikasi}",
        (12, 52), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255)
    )

    tampilkan((
        ('Citra', citra),
        ('Hasil', hasil)
    ))

    print(f"Bentuk kaleng: [{klasifikasi}]")
    print(end='', flush=True)

# Jika program dijalankan dengan perintah `python klasifikasi.py`
# jalankan fungsi klasifikasi dengan citra pada input lokasi
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input('Lokasi citra (dataset/dataset-1.jpg): ') \
            or 'dataset/dataset-1.jpg'

    klasifikasi(file_path)
