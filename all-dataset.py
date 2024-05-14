import os
import numpy as np
from main import klasifikasi

dataset_dir = './dataset'
sort = lambda f: int(''.join(filter(str.isdigit, f)))
files = sorted(os.listdir(dataset_dir), key=sort)
rasio_objek = []

for file in files:
    print('\n', file, flush=True)
    rasio = klasifikasi(os.path.join(dataset_dir, file), True)
    rasio_objek.append(rasio)

print("\n\nSELESAI")
print("Hasil rasio background per pixels dataset")
print("Mean:", round(np.mean(rasio_objek), 5))
print("Median:", round(np.median(rasio_objek), 5))
