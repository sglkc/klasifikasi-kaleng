import cv2
import numpy as np

def showd(citra, title='window'):
    cv2.imshow(title, citra)
    cv2.waitKey()
    cv2.destroyAllWindows()
    exit()

def klasifikasi(file_path: str, show=True):

    ## Load
    citra = cv2.imread(file_path)
    final_image = citra.copy()

    ## Preprocess
    ### Grayscaling
    citra = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)

    ### Thresholding invert
    _, citra = cv2.threshold(citra, 225, 255, cv2.THRESH_BINARY_INV)

    ### Perbesar citra sebelum morfologi
    offset = 100
    half_offset = offset // 2
    h, w = citra.shape[:2]
    mask = np.zeros((h + offset, w + offset))

    for y in range(h):
        for x in range(w):
            mask[y + half_offset, x + half_offset] = citra[y, x]

    ### Morfologi closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32, 32))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    ### Crop citra sesudah morfologi
    for y in range(h):
        for x in range(w):
            citra[y, x] = mask[y + half_offset, x + half_offset]

    ## KONTUR
    contours, _ = cv2.findContours(citra, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(citra, contours, -1, (255, 255, 255), 2)
    cv2.drawContours(final_image, contours, -1, (0, 255, 255), 2)

    ## MEMBUAT BOUNDING BOX DARI KONTUR

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
    bg_px = bg / px if px > 0 else 0
    ob_px = ob / px if px > 0 else 0
    bg_ob = bg / ob if ob > 0 else 0

    print("Pixel Latar:", bg)
    print("Pixel Objek:", ob)
    print("Latar / Pixels:", bg_px)
    print("Objek / Pixels:", ob_px)
    print("Latar / Objek:", bg_ob)
    print(end='', flush=True)

    ## TAMPILKAN CITRA
    if show:
        i = 1
        windows = [
            ('Preprocess', citra,),
            # ('Kontur', contour_only,),
            ('Hasil', final_image,),
        ]

        for title, img in windows:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.moveWindow(title, i * 500, 300)
            cv2.resizeWindow(title, 450, 450)
            cv2.imshow(title, img)
            i += 1

        cv2.waitKey()
        cv2.destroyAllWindows()

    ## Return rasio background / pixels
    return bg_px

if __name__ == "__main__":
    file_path = input('Lokasi file untuk diklasifikasi (dataset/image.jpg): ')
    klasifikasi(file_path)
