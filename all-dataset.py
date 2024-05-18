import os
import numpy as np
from analisis import analisis

# Fungsi untuk mengurutkan file sesuai angka pada nama
sort = lambda f: int(''.join(filter(str.isdigit, f)))

# Lokasi seluruh citra dataset
dataset_dir = './dataset'

# Dapatkan semua citra dataset yang telah diurutkan
files = sorted(os.listdir(dataset_dir), key=sort)

# Definisikan array untuk menyimpan rasio dari setiap dataset
rasio_objek = []

# Perulangan untuk setiap citra dataset pada folder
for file in files:
    print('\n', file, sep='', end='', flush=True)

    ## Dapatkan lokasi file
    file_path = os.path.join(dataset_dir, file)

    ## Analisis piksel pada citra dataset yang telah diolah
    rasio = analisis(file_path, show=False, verbose=False)
    rasio_objek.append(rasio)

# Tampilkan hasil rata-rata dan median dari seluruh dataset
print("\n\nSELESAI")
print("Hasil dataset rasio background per piksel")
print("Mean:", round(np.mean(rasio_objek), 5))
print("Median:", round(np.median(rasio_objek), 5))
