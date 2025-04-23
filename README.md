# Visualizer Spectrogram AWDS

Questo progetto fornisce una **applicazione GUI** in Python per visualizzare spettrogrammi di segnali VLF estratti da file HDF5 e per sovrapporre le detection di whistler generate dal tool **AWDS**. Include inoltre un modulo per la rappresentazione istogrammi delle detections.

## Caratteristiche

- Calcolo e plot interattivo dello spettrogramma di un file HDF5
- Visualizzazione dei rettangoli di detection (Start_Time, End_Time, Start_Freq, End_Freq, D0)
- Possibilità di filtrare detections in base a range temporale e frequenziale
- Esplorazione grafica di intervalli personalizzati
- Istogramma delle detections nel tempo

## Prerequisiti

- Python 3.8+
- Librerie Python:
  - `h5py`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `tkinter` (incluso nella distribuzione standard di Python su Windows/Mac)
  - `awds` (il modulo contenente `Spectra`)

## Struttura del Progetto
├── visualizer_spectrogram_range_10.py  # Applicazione principale GUI

├── histogram.py                        # Modulo per plot istogrammi

├── utils/                              # Eventuali script di utilità

├── data/

│   ├── fileH5/     # Folder contenente file .h5 di input

│   └── filePKL/    # Folder contenente file .pkl con le detection

└── README.md                           # Questo file


## Configurazione

All’inizio di `visualizer_spectrogram_range_10.py` definire:

- `fileH5`: percorso alla cartella dei file `.h5`
- `filePKL`: percorso alla cartella dei file `.pkl`
- `SAMPLING_FREQUENCY`: frequenza di campionamento (es. 51200 Hz)
- `COLORMAPS`: lista di colormap disponibili per il plot

## Utilizzo

1. Assicurarsi di avere i file `.h5` nella cartella `fileH5` e i corrispondenti `.pkl` in `filePKL`.
2. Installare i prerequisiti:
   ```bash
   pip install h5py numpy pandas matplotlib
python visualizer.py

4. Nella finestra GUI:
   - Selezionare il file H5 dalla tendina.
   - Abilitare/disabilitare le detections con la checkbox.
   - Selezionare intervalli di inizio/fine per analisi puntuali.
   - Scegliere la colormap.
   - Premere **Plot** per disegnare il grafico nella scheda "Intervallo".
   - Usare il tab **Istogramma** per visualizzare la distribuzione temporale delle detections.

## Esempio di Interfaccia

- **Tab “Visualizzazione”**: spettrogramma full-chunk con overlay dei rettangoli di detection.
- **Tab “Intervallo”**: spettrogramma di un sotto-intervallo selezionato.
- **Tab “Istogramma”**: grafico a barre del numero di events in base all’istante di inizio.

## Moduli Principali

- `load_signal_and_time(h5_file)`: legge il segnale da `A131_W` e il timestamp iniziale da `UTC_TIME`.
- `plot_spectrogram_section(...)`: calcola e visualizza spettrogramma + detections.
- `plot_spectrogram_basic_section(...)`: spettrogramma di un intervallo senza overlay.
- `HistogramPlotter`: classe per generare l’istogramma delle detection (definita in `histogram.py`).

## Personalizzazioni

- Modificare `COLORMAPS` per aggiungere nuove mappe.
- In `plot_spectrogram_section` è possibile filtrare ulteriormente le detections modificando i criteri su `df` (ad es. range di `D0` o di frequenza).


