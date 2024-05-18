import cv2, time, sys
import numpy as np
from typing import Iterable, Sequence, Tuple
from cv2.typing import MatLike

__verbose = False

# Prosedur untuk menampilkan teks pada terminal
def log(text, force=False) -> None:
    global __verbose

    ## Jika verbose atau force = True, tampilkan pesan
    if __verbose or force:
        print(text, flush=True)
    else:
        print('.', end='', flush=True)

# Fungsi grayscaling dengan rumus fungsi yang dapat diubah
def grayscale(citra: MatLike,
              rumus=lambda r,g,b:(0.2126*r)+(0.7152*g)+(0.0722*b)
              ) -> MatLike:

    ## Dapatkan tinggi dan lebar citra lalu buat citra hasil kosong
    h, w = citra.shape[:2]
    hasil = np.zeros((h, w), np.uint8)

    ## Perulangan untuk setiap piksel pada citra
    for y in range(h):
        for x in range(w):

            ### Dapatkan nilai biru, hijau, dan merah lalu hitung
            ### dengan rumus
            b, g, r = citra[y, x]
            pixel = np.ceil(rumus(r, g, b))

            ### Clipping agar nilai sesuai rentang kedalaman 8 bit
            if pixel < 0: pixel = 0
            if pixel > 255: pixel = 255

            ### Piksel pada koordinat ini pada citra hasil
            ### adalah hasil kalkulasi
            hasil[y, x] = pixel

    ## Kembalikan citra hasil
    return hasil

# Fungsi invert thresholding dengan T = titik ambang
def invert_threshold(citra: MatLike, T: int) -> MatLike:

    ## Dapatkan tinggi dan lebar citra lalu buat citra hasil kosong
    h, w = citra.shape[:2]
    hasil = np.zeros((h, w), np.uint8)

    ## Perulangan untuk setiap piksel pada citra
    for y in range(h):
        for x in range(w):

            ### Jika nilai piksel lebih dari T, maka 0
            ### Jika nilai piksel kurang dari sama dengan T, maka 1
            hasil[y, x] = 0 if citra[y, x] > T else 255

    ## Kembalikan citra hasil
    return hasil

# Fungsi untuk membuat bingkai pada citra dengan ketebalan tertentu
def padding(citra: MatLike, thickness: int) -> MatLike:

    ## Dapatkan tinggi dan lebar citra lalu buat citra hasil kosong
    h, w = citra.shape[:2]

    ## Citra hasil memiliki tinggi dan lebar yang ditambahkan dengan
    ## ketebalan yang diinginkan
    hasil = np.zeros((h + thickness * 2, w + thickness * 2), np.uint8)

    ## Perulangan untuk setiap piksel pada citra
    for y in range(h):
        for x in range(w):

            ### Piksel citra hasil memakai offset agar objek
            ### berada di tengah
            hasil[y + thickness, x + thickness] = citra[y, x]

    ## Kembalikan citra hasil
    return hasil

def closing(citra: MatLike, strel: MatLike) -> MatLike:
    return cv2.morphologyEx(
        citra,
        cv2.MORPH_CLOSE,
        strel,
        iterations=4
    )

# Fungsi untuk memotong bingkai pada citra dengan ketebalan tertentu
def crop(citra: MatLike, thickness: int) -> MatLike:

    ## Dapatkan tinggi dan lebar citra
    h1, w1 = citra.shape[:2]

    ## Dapatkan tinggi dan lebar citra tanpa bingkai
    h2, w2 = h1 - thickness * 2, w1 - thickness * 2

    ## Buat citra hasil kosong dengan tinggi dan lebar setelah crop
    hasil = np.zeros((h2, w2), np.uint8)

    ## Perulangan untuk mendapatkan setiap piksel pada citra
    for y in range(h2):
        for x in range(w2):

            ### Piksel citra hasil memakai offset karena
            ### objek berada di tengah
            hasil[y, x] = citra[y + thickness, x + thickness]

    ## Kembalikan citra hasil
    return hasil

# Fungsi bounding box untuk mendapatkan titik terluar dari kontur
def bounding_box(citra: MatLike,
                 contours: Sequence[MatLike]
                 ) -> Tuple[int, int, int, int]:

    ## Atas & kiri adalah titik minimum, jadi mulai dari ukuran citra
    atas, kiri = citra.shape[:2]

    ## Bawah & kanan adalah titik maksimum, jadi mulai dari 0
    bawah, kanan = 0, 0

    ## Perulangan untuk setiap garis dari kontur
    for i in range(len(contours)):
        garis = contours[i]

        ### Perulangan untuk setiap titik dari garis kontur
        for j in range(len(garis)):

            #### Dapatkan titik x, y dari setiap titik garis
            x, y = garis[j][0]

            #### Percabangan untuk menentukan titik terjauh
            if bawah < y: bawah = y
            if kanan < x: kanan = x
            if atas  > y: atas = y
            if kiri  > x: kiri = x

    ## Kembalikan titik terjauh atas, kiri, bawah, dan kanan
    return atas, kiri, bawah, kanan

# Fungsi menggambar kontur pada citra, seperti fungsi pada OpenCV
def draw_contours(citra: MatLike, contours: Sequence[MatLike],
                  color: Tuple[int, int, int], thickness: int
                  ) -> MatLike:

    ## Buat citra hasil yang merupakan duplikat dari citra masukan
    hasil = citra.copy()

    ## Gambar kontur pada citra hasil
    cv2.drawContours(hasil, contours, -1, color, thickness)

    ## Kembalikan citra hasil
    return hasil

# Fungsi menggambar kotak pada citra dengan 4 titik
def draw_box(citra: MatLike, box: Tuple[int, int, int, int],
             color: Tuple[int, int, int], thickness: int) -> MatLike:

    ## Mendapatkan titik atas, kiri, bawah, dan kanan dari box
    atas, kiri, bawah, kanan = box

    ## Buat citra hasil yang merupakan duplikat dari citra masukan
    hasil = citra.copy()

    ## Menggambar 4 garis untuk membuat kotak
    cv2.line(hasil, (kiri, atas), (kanan, atas), color, thickness)
    cv2.line(hasil, (kanan, atas), (kanan, bawah), color, thickness)
    cv2.line(hasil, (kanan, bawah), (kiri, bawah), color, thickness)
    cv2.line(hasil, (kiri, bawah), (kiri, atas), color, thickness)

    ## Kembalikan citra hasil
    return hasil

# Fungsi menghitung piksel pada citra dengan bounding box
def hitung_piksel(citra: MatLike,
                  box: Tuple[int, int, int, int]
                  ) -> Tuple[int, int, int, float, float, float]:

    ## Dapatkan setiap titik dari kotak
    atas, kiri, bawah, kanan = box

    ## Definisikan variabel piksel, latar, dan objek
    px = 0
    bg = 0
    ob = 0

    ## Perulangan untuk setiap piksel citra di dalam bounding box
    for y in range(atas, bawah):
        for x in range(kiri, kanan):

            ### Menambahkan jumlah piksel
            px += 1

            ### Jika piksel putih, maka termasuk objek
            ### Jika piksel hitam, maka termasuk latar
            if citra[y, x]:
                ob += 1
            else:
                bg += 1

    ## Bulatkan perbandingan jika bilangan penyebut lebih dari 0
    bg_px = round(bg / px, 5) if px > 0 else 0
    ob_px = round(ob / px, 5) if px > 0 else 0
    bg_ob = round(bg / ob, 5) if ob > 0 else 0

    ## Kembalikan jumlah piksel, latar, objek, dan perbandingannya
    return px, bg, ob, bg_px, ob_px, bg_ob

# Fungsi menampilkan citra pada window yang berderet
def tampilkan(windows: Iterable[Tuple[str, MatLike]]):
    i = 0

    for title, citra in windows:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.moveWindow(title, i % 3 * 500 + 50, i // 3 * 450 + 50)
        cv2.resizeWindow(title, 400, 400)
        cv2.imshow(title, citra)
        i += 1

    cv2.waitKey()
    cv2.destroyAllWindows()

# Fungsi utama analisis yang menjalankan semua proses pada citra
def analisis(file_path: str, show=True, verbose=False):

    ## Variabel apakah output pada terminal mendetail
    global __verbose
    __verbose = verbose

    ## Mulai menghitung waktu mulai pengolahan
    start_time = time.process_time()

    ## Load
    log('Memuat citra')
    citra = cv2.imread(file_path)

    ## Preprocess
    ### Grayscaling
    log('Grayscaling...')
    hasil_grayscale = grayscale(citra)

    ### Thresholding invert
    log('Thresholding...')
    hasil_threshold = invert_threshold(hasil_grayscale, 225)

    ### Perbesar citra sebelum morfologi
    log('Perbesar citra...')
    hasil_perbesar = padding(hasil_threshold, 50)

    ### Morfologi closing
    log('Morfologi closing...')
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32, 32))
    hasil_closing = closing(hasil_perbesar, strel)

    ### Crop citra sesudah morfologi
    log('Cropping...')
    hasil_crop = crop(hasil_closing, 50)

    ## EKSTRAKSI FITUR
    ekstraksi_fitur = np.zeros_like(citra)

    ### KONTUR
    log('Mencari kontur...')
    HITAM = (255, 255, 255)
    contours, _ = cv2.findContours(
        hasil_crop,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    hasil_kontur = draw_contours(hasil_crop, contours, HITAM, 2)
    ekstraksi_fitur = draw_contours(ekstraksi_fitur, contours, HITAM, 2)

    ### MEMBUAT BOUNDING BOX DARI KONTUR
    log('Bounding box...')
    box = bounding_box(hasil_kontur, contours)
    hasil_bounding = draw_box(hasil_kontur, box, HITAM, 2)
    ekstraksi_fitur = draw_box(ekstraksi_fitur, box, HITAM, 2)

    ## Tampilkan waktu dibutuhkan untuk pengolahan
    elapsed = round(time.process_time() - start_time, 3)
    log(f'({elapsed}s)', force=True)

    ## MENGHITUNG RASIO ANTARA LATAR & OBJEK
    px, bg, ob, bg_px, ob_px, bg_ob = hitung_piksel(hasil_bounding, box)

    print("Jumlah Piksel:", px)
    print("Piksel Latar:", bg)
    print("Piksel Objek:", ob)
    print("Latar / Piksel:", bg_px)
    print("Objek / Piksel:", ob_px)
    print("Latar / Objek:", bg_ob)
    print(end='', flush=True)

    # Tampilkan rasio pada citra
    cv2.putText(
        ekstraksi_fitur, f"Rasio latar: {bg_px}",
        (12, 24), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255)
    )

    cv2.putText(
        citra, f"Rasio latar: {bg_px}",
        (12, 24), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255)
    )

    ## Tampilkan citra jika argumen show = True
    if show:
        tampilkan((
            ('Citra Awal', citra),
            ('Grayscale', hasil_grayscale),
            ('Threshold', hasil_threshold),
            ('Perbesar', hasil_perbesar),
            ('Closing', hasil_closing),
            ('Ekstraksi Fitur', ekstraksi_fitur),
        ))

    ## Kembalikan rasio background / piksel
    return bg_px, citra, ekstraksi_fitur

# Jika program dijalankan dengan perintah `python analisis.py`
# jalankan fungsi analisis dengan citra pada input lokasi
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input('Lokasi citra (dataset/dataset-1.jpg): ') \
            or 'dataset/dataset-1.jpg'

    analisis(file_path, show=False, verbose=True)
