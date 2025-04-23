import h5py
import numpy as np
from datetime import datetime

def analyze_h5_file(file_path, sampling_frequency=51200):
    with h5py.File(file_path, 'r') as f:
        print(f"Analisi del file: {file_path}\n")

        # Elenco dei dataset presenti nel file
        print("Chiavi presenti nel file:")
        for key in f.keys():
            print(f" - {key}")
        print()

        # Verifica la presenza del dataset 'A131_W'
        if 'A131_W' in f:
            dataset = f['A131_W']
            shape = dataset.shape
            print(f"Dataset 'A131_W' trovato con shape: {shape}")

            # Calcolo della durata totale del segnale
            num_samples = shape[0] * shape[1]
            duration_seconds = num_samples / sampling_frequency
            print(f"Durata totale del segnale: {duration_seconds:.2f} secondi\n")
        else:
            print("Dataset 'A131_W' non trovato nel file.\n")

        # Verifica la presenza del dataset 'UTC_TIME'
        if 'UTC_TIME' in f:
            utc_time = f['UTC_TIME'][:]
            if utc_time.size > 0:
                # Assumendo che 'UTC_TIME' sia un array di interi con formato 'YYYYMMDDHHMMSSfff'
                utc_str = str(utc_time[0][0]) if utc_time.ndim > 1 else str(utc_time[0])
                try:
                    start_time = datetime.strptime(utc_str, "%Y%m%d%H%M%S%f")
                    print(f"Tempo di inizio (UTC_TIME): {start_time}")
                except ValueError as e:
                    print(f"Errore nella conversione di UTC_TIME: {e}")
            else:
                print("Il dataset 'UTC_TIME' è vuoto.")
        else:
            print("Dataset 'UTC_TIME' non trovato nel file.")

# Esempio di utilizzo
file_path = 'C:/Users/Ivan/PycharmProjects/visualizer-Donato-Ivan/visualizer-Ivan/visualizer/ENV_TEST/fileH5/CSES_01_EFD_3_L02_A1_267941_20221201_003334_20221201_011048_000.h5'  
analyze_h5_file(file_path)
