#!/usr/bin/env python3
import os
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from awds.spectra import Spectra
from whistler_detection_visualizer import WhistlerDetectionVisualizer

# Parametri di campionamento e file di input (modifica secondo le tue cartelle)
SAMPLING_FREQUENCY = 51200
file_name = "CSES_01_EFD_3_L02_A1_059341_20190227_151534_20190227_155009_000"
# Modifica questi percorsi secondo la tua struttura
H5_FILE = os.path.join("C:/Users/Ivan/PycharmProjects/visualizer-Donato-Ivan/visualizer-Donato/fileH5_test", file_name + ".h5")
PKL_FILE = os.path.join("C:/Users/Ivan/PycharmProjects/visualizer-Donato-Ivan/visualizer-Donato/filePKL_test", file_name + ".pkl")
OUTPUT_FOLDER = "C:/Users/Ivan/PycharmProjects/visualizer-Donato-Ivan/visualizer-Donato/output"

def load_signal_from_h5(h5_file, sampling_frequency=SAMPLING_FREQUENCY):
    """
    Legge il file H5 e restituisce:
      - signal: il segnale completo (dal dataset 'A131_W')
      - signal_start_time: il timestamp del primo campione (dal dataset 'UTC_TIME')
    """
    with h5py.File(h5_file, "r") as f:
        # Leggi il segnale dal dataset 'A131_W'
        if 'A131_W' in f.keys():
            data = np.array(f['A131_W'])
            signal = np.concatenate(data.tolist())
        else:
            signal = None

        # Leggi il timestamp dal dataset 'UTC_TIME'
        if 'UTC_TIME' in f.keys():
            utc_times = f['UTC_TIME'][()]
            first_time_str = str(utc_times[0][0])
            try:
                signal_start_time = datetime.strptime(first_time_str, "%Y%m%d%H%M%S%f")
            except Exception as e:
                print(f"Errore nella conversione del timestamp: {e}")
                signal_start_time = None
        else:
            signal_start_time = None

    return signal, signal_start_time

def load_detections_from_pkl(pkl_file, signal_start_time):
    """
    Carica il file PKL e converte le colonne di rilevamento in una lista di detection boxes.
    Si assume che il PKL contenga le colonne:
      'Start_Time', 'End_Time', 'Start_Freq', 'End_Freq', 'D0'
    I tempi vengono convertiti in secondi relativi a signal_start_time;
    le frequenze vengono convertite da Hz a kHz.
    Restituisce una lista di tuple: (x1, x2, y1, y2, D0)
    """
    df = pd.read_pickle(pkl_file)
    if isinstance(df, list):
        df = pd.DataFrame(df)
    # Converti i timestamp in secondi relativi a signal_start_time
    df['x1'] = pd.to_datetime(df['Start_Time']).apply(lambda x: (x - signal_start_time).total_seconds())
    df['x2'] = pd.to_datetime(df['End_Time']).apply(lambda x: (x - signal_start_time).total_seconds())
    # Converti le frequenze in kHz
    df['y1'] = df['Start_Freq'] / 1000.0
    df['y2'] = df['End_Freq'] / 1000.0
    # Seleziona le colonne necessarie
    detections = df[['x1', 'x2', 'y1', 'y2', 'D0']].to_records(index=False)
    detection_list = [tuple(rec) for rec in detections]
    return detection_list

def main():
    # 1. Carica il segnale dal file H5
    signal, signal_start_time = load_signal_from_h5(H5_FILE, SAMPLING_FREQUENCY)
    if signal is None or signal_start_time is None:
        print("Errore nel caricamento del file H5.")
        return
    print(f"Signal length: {len(signal)} campioni")
    durata = len(signal) / SAMPLING_FREQUENCY
    print(f"Durata del segnale: {durata:.3f} s")
    print(f"Signal start time: {signal_start_time}")

    # 2. Calcola lo spettrogramma usando la classe Spectra (AWDS)
    spectra_obj = Spectra()
    freqs, times, spec = spectra_obj.spectrogram(signal, SAMPLING_FREQUENCY)
    print(f"Spettrogramma shape: {spec.shape}")
    print(f"Time range: {times[0]:.3f} s - {times[-1]:.3f} s")
    print(f"Freq range: {freqs[0]:.3f} kHz - {freqs[-1]:.3f} kHz")

    # 3. Carica le detections dal file PKL
    detections = load_detections_from_pkl(PKL_FILE, signal_start_time)
    if not detections:
        print("Nessun rilevamento trovato nel file PKL.")
        return
    print(f"Trovati {len(detections)} rilevamenti.")

    # 4. Crea un'istanza di WhistlerDetectionVisualizer
    visualizer = WhistlerDetectionVisualizer(
        plot_obj=None,
        padding_factor=0.2,
        kernel_alpha=0.7,
        kernel_threshold=0.1,
        output_dir=OUTPUT_FOLDER
    )

    # 5. Processa tutte le detections e genera le visualizzazioni
    saved_figs = visualizer.process_all_detections(
        spectrogram=spec,
        time=times,
        freqs=freqs,
        detections=detections,
        t_res=0.01,      # Risoluzione temporale in secondi
        f_res=0.1,       # Risoluzione in frequenza in kHz
        low_f=4000,      # Limite inferiore in Hz (modifica se necessario)
        high_f=12000,    # Limite superiore in Hz
        fn=25000,        # Nominal frequency in Hz
        spectrogram_cmap='jet',
        kernel_cmap='transparent_red',
        show_kernel=True
    )
    if saved_figs:
        print("Visualizzazioni salvate:")
        for fig_path in saved_figs:
            print(f" - {fig_path}")
    else:
        print("Nessuna visualizzazione generata.")

if __name__ == "__main__":
    main()