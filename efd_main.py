#!/usr/bin/env python3
import multiprocessing
import os

from awds.efd import EFD
from awds.persistence import StoreEFD
from awds.awds_with_persistence import AWDS
from joblib import Parallel, delayed
from tqdm import tqdm

# Use the same base folders as visualizer.py
H5_FOLDER = os.path.join(os.getcwd(), "fileH5")
PKL_FOLDER = os.path.join(os.getcwd(), "filePKL")

def find_whistlers(file_tuple):
    year, month, file_name = file_tuple
    folder_path = H5_FOLDER

    data_analysis_path = PKL_FOLDER
    if not os.path.exists(data_analysis_path):
        os.makedirs(data_analysis_path)

    target_pkl = file_name.replace(".h5", ".pkl")
    if target_pkl in os.listdir(data_analysis_path):
        return

    awds.main(reader, store, folder_path, file_name)


if __name__ == "__main__":
    # Enumerate files under H5_FOLDER
    years_list = [d for d in os.listdir(H5_FOLDER) if os.path.isdir(os.path.join(H5_FOLDER, d))]
    list_files = []
    for y in years_list:
        months = [m for m in os.listdir(os.path.join(H5_FOLDER, y)) if os.path.isdir(os.path.join(H5_FOLDER, y, m))]
        for m in months:
            dir_path = os.path.join(H5_FOLDER, y, m)
            for fn in os.listdir(dir_path):
                if fn.endswith('.h5'):
                    list_files.append((y, m, fn))

    awds = AWDS()
    store = StoreEFD()
    reader = EFD()

    num_cores = multiprocessing.cpu_count()
    inputs = tqdm(list_files, position=0, leave=True)
    Parallel(n_jobs=num_cores)(delayed(find_whistlers)(i) for i in inputs)