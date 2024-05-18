from analisis import analisis

# Fungsi klasifikasi dengan rasio optimal
def klasifikasi(file_path: str):

    ## Dapatkan rasio dari analisis piksel
    bg_px = analisis(file_path, show=False, verbose=False)

    ## Konstanta kaleng optimal dengan tambahan error margin 0.5%
    optimal = 0.03767 + 0.005

    ## Jika rasio pada citra melebihi nilai optimal,
    ## maka dikatakan buruk, sebaliknya dikatakan baik
    if bg_px > optimal:
        print('Bentuk kaleng: [BURUK]')
    else:
        print('Bentuk kaleng: [BAIK]')

    print(end='', flush=True)

# Jika program dijalankan dengan perintah `python klasifikasi.py`
# jalankan fungsi klasifikasi dengan citra pada input lokasi
if __name__ == "__main__":
    file_path = input('Lokasi citra (dataset/dataset-1.jpg): ') \
        or 'dataset/dataset-1.jpg'
    klasifikasi(file_path)
