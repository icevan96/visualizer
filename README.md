# AWDS Visualizer

A desktop application per esplorare l’output del **Automatic Whistler Detection System (AWDS)**.  
Carica file `.h5`, genera detections (PKL) e naviga interattivamente tra spettrogrammi, anteprime e istogrammi.

---

## Features

- **Apri & Ispeziona**  
  - Modale con metadata H5: intervallo temporale, lat/lon, luogo geocodificato.  
- **Generazione PKL integrata**  
  - AWDS in-process.  
- **Navigazione Spettrogrammi**  
  - Paginazione, zoom scalabile, filtro interattivo su soglia `D0`.  
- **Browser Detections**  
  - Anteprime dei kernel di whistler, navigazione Next/Prev.  
- **Tab Istogramma**  
  - Distribuzione dei parametri di detection (`D0`, frequenza, durata).  

---

## Requirements

- **Python** ≥ 3.8  
- **Tkinter** (incluso con Python)  
- **AWDS library** (nel `PYTHONPATH`)  
- **Dipendenze Python**:  
  ```bash
  pip install pandas numpy h5py pillow matplotlib reverse_geocoder pycountry_convert tqdm joblib
  ```

---

## Installation

1. Clona il repository:
   ```bash
   git clone https://github.com/your-org/awds-visualizer.git
   cd awds-visualizer
   ```
2. _(Opzionale)_ Crea e attiva un virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate.bat      # Windows
   ```
3. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

```bash
python visualizer.py
```

1. Clicca **Apri** e seleziona un file `.h5`.  
2. Nella finestra **Dettagli file H5** clicca **Finish**.  
3. Se il `.pkl` non esiste, premi **Genera PKL**.  
4. Naviga tra le tab:  
   - **Spettrogramma**: paginazione, zoom e filtro `D0`.  
   - **Detections**: scorrimento anteprime.  
   - **Istogramma**: distribuzione dei parametri.  
   - **Dettagli**: metadata H5.

---

## Project Structure

```
├── visualizer.py
├── run_awds.py
├── awds_visualizer/
│   ├── plotting.py
│   └── whistler_detection_visualizer.py
├── histogram.py
├── requirements.txt
└── README.md
```

---

## Contributing

1. **Fork** del repo  
2. **Branch** di feature:  
   ```bash
   git checkout -b feature/tuo-branch
   ```
3. **Commit** delle modifiche:  
   ```bash
   git commit -m "Descrizione delle modifiche"
   ```
4. **Push** e apri **Pull Request**.
