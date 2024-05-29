import os
from klasifikasi import klasifikasi

# Fungsi untuk mengurutkan file sesuai angka pada nama
sort = lambda f: int(''.join(filter(str.isdigit, f)))

# Lokasi seluruh citra uji
dataset_dir = './test'

# Dapatkan semua citra dataset yang telah diurutkan
files = sorted(os.listdir(dataset_dir), key=sort)

# Perulangan untuk setiap citra dataset pada folder
for file in files:
    print('\n', file, sep='', end='', flush=True)

    ## Dapatkan lokasi file
    file_path = os.path.join(dataset_dir, file)

    ## Analisis piksel pada citra dataset yang telah diolah
    klasifikasi(file_path, show=False)
