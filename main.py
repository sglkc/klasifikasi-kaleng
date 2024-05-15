from typing import Sequence, Tuple
import cv2, time
import numpy as np
from cv2.typing import MatLike

__verbose = False

def log(text, force=False):
    global __verbose

    if __verbose or force:
        print(text, flush=True)
    else:
        print('.', end='', flush=True)

def showd(citra, title='window'):
    cv2.imshow(title, citra)
    cv2.waitKey()
    cv2.destroyAllWindows()
    exit()

def grayscale(citra: MatLike, rumus=lambda r,g,b:(0.2126*r)+(0.7152*g)+(0.0722*b)):
    h, w = citra.shape[:2]
    hasil = np.zeros((h, w), np.uint8)

    for y in range(h):
        for x in range(w):
            b, g, r = citra[y, x]
            pixel = np.ceil(rumus(r, g, b))

            ## Clipping
            if pixel < 0: pixel = 0
            if pixel > 255: pixel = 255

            hasil[y, x] = pixel

    return hasil

def threshold(citra: MatLike, T: int):
    h, w = citra.shape[:2]
    hasil = np.zeros((h, w), np.uint8)

    for y in range(h):
        for x in range(w):
            hasil[y, x] = 0 if citra[y, x] > T else 255

    return hasil

def padding(citra: MatLike, thickness: int):
    h, w = citra.shape[:2]
    hasil = np.zeros((h + thickness * 2, w + thickness * 2), np.uint8)

    for y in range(h):
        for x in range(w):
            hasil[y + thickness, x + thickness] = citra[y, x]

    return hasil

def crop(citra: MatLike, thickness: int):
    h1, w1 = citra.shape[:2]
    h2, w2 = h1 - thickness * 2, w1 - thickness * 2
    hasil = np.zeros((h2, w2), np.uint8)

    for y in range(h2):
        for x in range(w2):
            hasil[y, x] = citra[y + thickness, x + thickness]

    return hasil

def bounding_box(citra: MatLike, contours: Sequence[MatLike]):

    # atas & kanan adalah nilai minimum jadi mulai dari ukuran citra
    # bawah & kiri adalah nilai maksimum, jadi mulai dari 0
    atas, kiri = citra.shape[:2]
    bawah, kanan = 0, 0

    # looping nyari titik terjauh atas bawah kanan kiri
    for i in range(len(contours)):
        garis = contours[i]

        for j in range(len(garis)):
            x, y = garis[j][0]

            if bawah < y: bawah = y
            if kanan < x: kanan = x
            if atas  > y: atas = y
            if kiri  > x: kiri = x

    return atas, kiri, bawah, kanan

def draw_contours(citra: MatLike, contours: Sequence[MatLike],
                  color: Tuple[int, int, int], thickness: int):
    hasil = citra.copy()
    cv2.drawContours(hasil, contours, -1, color, thickness)
    return hasil

def draw_box(citra: MatLike, box: Tuple[int, int, int, int],
             color: Tuple[int, int, int], thickness: int):
    atas, kiri, bawah, kanan = box
    hasil = citra.copy()

    cv2.line(hasil, (kiri, atas), (kanan, atas), color, thickness)
    cv2.line(hasil, (kanan, atas), (kanan, bawah), color, thickness)
    cv2.line(hasil, (kanan, bawah), (kiri, bawah), color, thickness)
    cv2.line(hasil, (kiri, bawah), (kiri, atas), color, thickness)

    return hasil

def hitung_piksel(citra: MatLike, box: Tuple[int, int, int, int]):
    atas, kiri, bawah, kanan = box
    px = 0
    bg = 0
    ob = 0

    for y in range(atas, bawah):
        for x in range(kiri, kanan):
            px += 1
            if citra[y, x]:
                ob += 1
            else:
                bg += 1

    ## TAMPILKAN HASIL RASIO
    bg_px = round(bg / px, 5) if px > 0 else 0
    ob_px = round(ob / px, 5) if px > 0 else 0
    bg_ob = round(bg / ob, 5) if ob > 0 else 0

    return px, bg, ob, bg_px, ob_px, bg_ob

def tampilkan(title: str, citra: MatLike, col=0):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.moveWindow(title, col % 3 * 500 + 50, col // 3 * 450 + 50)
    cv2.resizeWindow(title, 400, 400)
    cv2.imshow(title, citra)

def klasifikasi(file_path: str, show=True, verbose=False):
    global __verbose
    __verbose = verbose
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
    hasil_threshold = threshold(hasil_grayscale, 225)

    ### Perbesar citra sebelum morfologi
    log('Perbesar citra...')
    hasil_perbesar = padding(hasil_threshold, 50)

    ### Morfologi closing
    log('Morfologi closing...')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32, 32))
    hasil_closing = cv2.morphologyEx(
        hasil_perbesar, cv2.MORPH_CLOSE, kernel, iterations=4
    )

    ### Crop citra sesudah morfologi
    log('Cropping...')
    hasil_crop = crop(hasil_closing, 50)

    ## EKSTRAKSI FITUR
    ekstraksi_fitur = np.zeros_like(citra)

    ### KONTUR
    log('Mencari kontur...')
    contours, _ = cv2.findContours(hasil_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hasil_kontur = draw_contours(hasil_crop, contours, (255, 255, 255), 2)
    ekstraksi_fitur = draw_contours(ekstraksi_fitur, contours, (255, 255, 255), 2)

    ### MEMBUAT BOUNDING BOX DARI KONTUR
    log('Bounding box...')
    box = bounding_box(hasil_kontur, contours)
    hasil_bounding = draw_box(hasil_kontur, box, (255, 255, 255), 2)
    ekstraksi_fitur = draw_box(ekstraksi_fitur, box, (255, 255, 255), 2)

    ## Tampilkan waktu dibutuhkan untuk pengolahan
    elapsed = round(time.process_time() - start_time, 3)
    log(f'({elapsed}s)', force=True)

    ## MENGHITUNG RASIO ANTARA LATAR & OBJEK
    log('Menghitung piksel')
    px, bg, ob, bg_px, ob_px, bg_ob = hitung_piksel(hasil_bounding, box)

    print("Jumlah Piksel:", px)
    print("Piksel Latar:", bg)
    print("Piksel Objek:", ob)
    print("Latar / Piksel:", bg_px)
    print("Objek / Piksel:", ob_px)
    print("Latar / Objek:", bg_ob)
    print(end='', flush=True)

    ## TAMPILKAN CITRA
    if show:
        tampilkan('Citra Awal', citra, 0)
        tampilkan('Grayscale', hasil_grayscale, 1)
        tampilkan('Threshold', hasil_threshold, 2)
        tampilkan('Perbesar', hasil_perbesar, 3)
        tampilkan('Closing', hasil_closing, 4)
        tampilkan('Ekstraksi Fitur', ekstraksi_fitur, 5)
        cv2.waitKey()
        cv2.destroyAllWindows()

    ## Return rasio background / piksel
    return bg_px

if __name__ == "__main__":
    file_path = input('Lokasi citra untuk diklasifikasi (dataset/dataset-1.jpg): ') \
    or 'dataset/dataset-1.jpg'
    klasifikasi(file_path, show=True, verbose=True)
