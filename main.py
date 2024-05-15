import cv2, time
import numpy as np

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

def klasifikasi(file_path: str, show=True, verbose=False):
    global __verbose
    __verbose = verbose
    start_time = time.process_time()

    ## Load
    log('Memuat citra')
    citra = cv2.imread(file_path)
    final_image = citra.copy()

    ## Preprocess
    ### Grayscaling
    log('Grayscaling...')
    h, w = citra.shape[:2]
    grayscale = np.zeros((h, w), np.uint8)

    for y in range(h):
        for x in range(w):
            b, g, r = citra[y, x]
            pixel = np.ceil((0.2126 * r) + (0.7152 * g) + (0.0722 * b))

            ## Clipping
            if pixel < 0: pixel = 0
            if pixel > 255: pixel = 255

            grayscale[y, x] = pixel

    ### Thresholding invert
    log('Thresholding...')
    threshold = grayscale.copy()
    thres = 225

    for y in range(h):
        for x in range(w):
            threshold[y, x] = 0 if grayscale[y, x] > thres else 255

    ### Perbesar citra sebelum morfologi
    log('Perbesar citra...')
    offset = 100
    half_offset = offset // 2
    morfologi = np.zeros((h + offset, w + offset), np.uint8)

    for y in range(h):
        for x in range(w):
            morfologi[y + half_offset, x + half_offset] = threshold[y, x]

    ### Morfologi closing
    log('Morfologi closing...')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32, 32))
    morfologi = cv2.morphologyEx(morfologi, cv2.MORPH_CLOSE, kernel, iterations=4)

    ### Crop citra sesudah morfologi
    log('Cropping...')
    citra = np.zeros((h, w), np.uint8)

    for y in range(h):
        for x in range(w):
            citra[y, x] = morfologi[y + half_offset, x + half_offset]

    ## KONTUR
    log('Mencari kontur...')
    contours, _ = cv2.findContours(citra, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(citra, contours, -1, (255, 255, 255), 2)
    cv2.drawContours(final_image, contours, -1, (0, 255, 255), 2)

    ## MEMBUAT BOUNDING BOX DARI KONTUR
    log('Bounding box...')

    # atas & kanan adalah nilai minimum jadi ambil dari ukuran citra
    # bawa & kiri adalah nilai maksimum dari 0
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

    # gambar box
    for img in (citra, final_image):
        cv2.line(img, (kiri, atas), (kanan, atas), (0, 0, 255), 2)
        cv2.line(img, (kanan, atas), (kanan, bawah), (0, 0, 255), 2)
        cv2.line(img, (kanan, bawah), (kiri, bawah), (0, 0, 255), 2)
        cv2.line(img, (kiri, bawah), (kiri, atas), (0, 0, 255), 2)

    ## MENGHITUNG RASIO ANTARA LATAR & OBJEK
    log('Menghitung piksel')
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

    ## Tampilkan waktu dibutuhkan untuk pengolahan
    elapsed = round(time.process_time() - start_time, 3)
    log(f'({elapsed}s)', force=True)

    ## TAMPILKAN HASIL RASIO
    bg_px = round(bg / px, 5) if px > 0 else 0
    ob_px = round(ob / px, 5) if px > 0 else 0
    bg_ob = round(bg / ob, 5) if ob > 0 else 0

    print("Piksel Latar:", bg)
    print("Piksel Objek:", ob)
    print("Latar / Piksel:", bg_px)
    print("Objek / Piksel:", ob_px)
    print("Latar / Objek:", bg_ob)
    print(flush=True)

    ## TAMPILKAN CITRA
    if show:
        i = 1
        windows = [
            ('Preprocess', citra),
            ('Hasil', final_image),
        ]

        for title, img in windows:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.moveWindow(title, i * 500, 300)
            cv2.resizeWindow(title, 450, 450)
            cv2.imshow(title, img)
            i += 1

        cv2.waitKey()
        cv2.destroyAllWindows()

    ## Return rasio background / piksel
    return bg_px

if __name__ == "__main__":
    file_path = input('Lokasi citra untuk diklasifikasi (dataset/dataset-1.jpg): ') \
    or 'dataset/dataset-1.jpg'
    klasifikasi(file_path, verbose=True)
