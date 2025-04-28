#!/usr/bin/env python3
import os
import sys
import threading
import subprocess
import pandas as pd
import numpy as np
import h5py
from itertools import chain
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

import subprocess
from subprocess import CREATE_NO_WINDOW, STARTUPINFO, STARTF_USESHOWWINDOW
import runpy

# Prevent interactive matplotlib windows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# AWDS imports
from awds.spectra import Spectra
from awds.efd import EFD
import awds.awds_with_persistence as awds_util
from awds.awds_with_persistence import AWDS
from awds.persistence import StoreEFD

# Local modules
import src.plotting as plotting
from src.plotting import SpectrogramPlot
from src.whistler_detection_visualizer import WhistlerDetectionVisualizer
from histogram import HistogramPlotter

# Libreria reverse_geocode (geolocalizzazione)
import reverse_geocoder as rg
import pycountry_convert as pc

# Costanti
DISPLAY_SCALE = 0.5
PAGE_SIZE_SECONDS = 10
PKL_FOLDER = os.path.join(os.getcwd(), "filePKL")
FIG_FOLDER = os.path.join(os.getcwd(), "filePKL", "figures")

# Larghezza minima per immagini Detections
MIN_DET_WIDTH = 300


# Helper per centratura
def center_window(win, width, height):
    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    x = (screen_w - width) // 2
    y = (screen_h - height) // 2
    win.geometry(f"{width}x{height}+{x}+{y}")

# Helper per avvio tool AWDS
def run_awds(filepath, debug=False):
        """
        Crea il .pkl per il file H5 dato, replicando run_awds.py --path.
        """
        awds = AWDS()
        reader = EFD()
        store  = StoreEFD()

        directory = os.path.dirname(filepath)
        filename  = os.path.basename(filepath)

        awds.main(reader, store, directory, filename, debug_enabled=debug)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Spettrogramma & Detections & Istogramma")
        root.geometry("1000x700")
        root.minsize(600, 400)

        # Stato selezione
        self.selected_filepath = None
        self.selected_base     = None
        self.display_scale     = DISPLAY_SCALE
        self.overlay           = None
        self.d0_threshold      = 0.0

        # Crea cartelle se non esistono
        os.makedirs(PKL_FOLDER, exist_ok=True)
        os.makedirs(FIG_FOLDER, exist_ok=True)

        # Top: pulsante Apri + label
        top = tk.Frame(root)
        top.pack(fill='x', pady=5)
        btn_open = ttk.Button(top, text="Apri", command=self.open_file)
        btn_open.pack(side='left', padx=5)
        self.lbl_file = ttk.Label(top, text="Nessun file selezionato")
        self.lbl_file.pack(side='left', padx=5)

        # Scala immagine
        scale_row = tk.Frame(root)
        scale_row.pack(fill='x', padx=5, pady=2)
        tk.Label(scale_row, text="Scala immagine (da 0.1 a 1.0):").pack(side='left', padx=5)
        self.scale_entry = ttk.Entry(scale_row, width=5)
        self.scale_entry.pack(side='left')
        self.scale_entry.insert(0, str(DISPLAY_SCALE))
        self.scale_entry.bind('<Return>', lambda e: self.update_display_scale())

        # Notebook con tab
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')
        self._build_spectrogram_tab()
        self._build_detections_tab()
        self._build_histogram_tab()
        self._build_details_tab()

        # Contenitori dati
        self.splits      = []
        self.page_index  = 0
        self.spec_pages  = []
        self.t0_global   = None
        self.dets_global = []
        self.images      = []
        self.index       = 0


    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
        if not path:
            return
        self.show_file_details(path)

    # Finestra Dettagli
    def show_file_details(self, filepath):
        # Leggi metadati
        try:
            with h5py.File(filepath, 'r') as f:
                utc_raw  = f['UTC_TIME'][:].flatten()
                utc_strs = [f"{int(x):017d}" for x in utc_raw]
                times    = pd.to_datetime(utc_strs, format='%Y%m%d%H%M%S%f')
                start_ts = times[0]
                end_ts   = times[-1]
                duration = (end_ts - start_ts).total_seconds()
                lat = float(np.array(f['GEO_LAT'])[:].flatten()[0])
                lon = float(np.array(f['GEO_LON'])[:].flatten()[0])
        except KeyError as e:
            messagebox.showerror("Errore metadati H5", f"Dataset mancante: {e}")
            return
        except Exception as e:
            messagebox.showerror("Errore file", f"Impossibile leggere i metadati H5:\n{e}")
            return

        # reverse-geocoding (UnicodeDecodeError gestito)
        coord = (lat, lon)
        try:
            res = rg.search(coord, mode=1)[0]
            country_code = res.get('cc', '')
            region_name  = res.get('admin1', '')
        except UnicodeDecodeError:
            country_code = ''
            region_name  = ''
        try:
            cont_code = pc.country_alpha2_to_continent_code(country_code)
            continent = pc.convert_continent_code_to_continent_name(cont_code)
        except:
            continent = ''
        import pycountry
        country_obj = pycountry.countries.get(alpha_2=country_code)
        country = country_obj.name if country_obj else ''
        location_str = ", ".join(filter(None, [continent, region_name, country]))

        # Crea finestra di dettaglio in modal
        self.details_win = tk.Toplevel(self.root)
        self.details_win.title("Dettagli file H5")
        self.details_win.transient(self.root)      
        self.details_win.grab_set()                
        center_window(self.details_win, 500, 300)

        frm = ttk.Frame(self.details_win, padding=10)
        frm.pack(fill='both', expand=True)

        ttk.Label(frm, text=f"File: {os.path.basename(filepath)}").pack(anchor='w', pady=2)
        ttk.Label(frm, text=f"Durata (s): {duration:.2f}").pack(anchor='w', pady=2)
        ttk.Label(frm, text=f"Inizio: {start_ts}").pack(anchor='w', pady=2)
        ttk.Label(frm, text=f"Fine:   {end_ts}").pack(anchor='w', pady=2)
        ttk.Label(frm, text=f"Lat/Lon: {lat:.4f}, {lon:.4f}").pack(anchor='w', pady=2)
        ttk.Label(frm, text=f"Luogo: {location_str}").pack(anchor='w', pady=2)

        btn_frame = ttk.Frame(frm)
        btn_frame.pack(fill='x', pady=10)
        ttk.Button(btn_frame, text="Back",
                   command=lambda: (self.details_win.destroy(), self.open_file())
        ).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Finish",
                   command=lambda: self._finish_selection(filepath)
        ).pack(side='left', padx=5)

        # Fermo l'esecuzione qui fino a che details_win non viene chiusa
        self.root.wait_window(self.details_win)
    
    # Finestra Selection 
    def _finish_selection(self, filepath):
        # Salva stato precedente per rollback
        self._prev_selected_base     = self.selected_base
        self._prev_selected_filepath = self.selected_filepath

        # Imposta nuova selezione (ma non toccare UI finché non validato)
        base     = os.path.splitext(os.path.basename(filepath))[0]
        pkl_path = os.path.join(PKL_FOLDER, base + '.pkl')
        self.selected_filepath = filepath
        self.selected_base     = base

        # Chiudi finestra dettaglio
        if hasattr(self, 'details_win') and self.details_win.winfo_exists():
            self.details_win.destroy()

        if os.path.exists(pkl_path):
            # PKL già presente: procedi a process_and_load senza toccare UI
            self.show_loading("Caricamento spettrogramma…")
            try:
                self.threaded_task(self.process_and_load)
            except Exception as e:
                self.hide_overlay()
                self.root.after(0, lambda err=e: messagebox.showerror("Errore caricamento", str(err)))
                return
        else:
            # PKL mancante: dialog modale
            self.alert_win = tk.Toplevel(self.root)
            self.alert_win.title("PKL non trovato")
            # la rendiamo modale
            self.alert_win.transient(self.root)
            self.alert_win.grab_set()
            center_window(self.alert_win, 400, 200)
            frm = ttk.Frame(self.alert_win, padding=10)
            frm.pack(fill='both', expand=True)
            ttk.Label(frm, text="File .pkl non trovato.").pack(pady=10)
            btns = ttk.Frame(frm)
            btns.pack(pady=5)
            ttk.Button(btns, text="Exit", command=self._cancel_selection).pack(side='left', padx=5)
            ttk.Button(btns, text="Genera PKL", command=lambda: self._generate_pkl(filepath)).pack(side='left', padx=5)
            # blocca l’esecuzione fino a che l’utente non chiude il dialog
            self.root.wait_window(self.alert_win)

    def _cancel_selection(self):
        # Ripristina selezione precedente
        if hasattr(self, 'alert_win') and self.alert_win.winfo_exists():
            self.alert_win.destroy()
        self.selected_base     = getattr(self, '_prev_selected_base', None)
        self.selected_filepath = getattr(self, '_prev_selected_filepath', None)
        if self.selected_base:
            self.lbl_file.config(text=self.selected_base)


    def _generate_pkl(self, filepath):
        # Chiudi dialog PKL se aperto
        if hasattr(self, 'alert_win') and self.alert_win.winfo_exists():
            self.alert_win.destroy()

        def worker():
            # 1) fase PKL
            self.show_loading("Generazione file PKL…")
            try:
                run_awds(filepath, debug=False)
            except Exception as e:
                self.hide_overlay()
                self.root.after(0, lambda err=e: messagebox.showerror("Errore PKL", str(err)))
                return

            # 2) fase loading spettrogramma
            self.show_loading("Caricamento spettrogramma…")
            try:
                self.process_and_load()
            except Exception as e:
                self.hide_overlay()
                self.root.after(0, lambda err=e: messagebox.showerror("Errore caricamento", str(err)))
                return

            # 3) fine, rimuovi overlay
            self.hide_overlay()

        # Esegui tutto in thread per non bloccare la UI
        threading.Thread(target=worker, daemon=True).start()


    def _on_pkl_generated(self, filepath):
        self.hide_overlay()
        # Dopo generazione, process_and_load si occuperà di aggiornare o fare rollback
        self.threaded_task(self.process_and_load)


    def process_and_load(self):
        if not getattr(self, 'selected_filepath', None):
            return

        filepath = self.selected_filepath
        base     = self.selected_base

        # Prepara cartelle di output
        fig_dir = os.path.join(FIG_FOLDER, base)
        det_dir = os.path.join(fig_dir, 'detections_with_kernels')
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(det_dir, exist_ok=True)
        self.current_fig_dir = fig_dir
        self.current_det_dir = det_dir
        self.current_pkl     = os.path.join(PKL_FOLDER, base + '.pkl')

        # Salva stato per rollback
        prev_base     = getattr(self, '_prev_selected_base', None)
        prev_filepath = getattr(self, '_prev_selected_filepath', None)

        # Carica dati VLF
        reader = EFD()
        vlf    = reader.read(os.path.dirname(filepath), os.path.basename(filepath))

        # Se non ci sono burst -> rollback e avviso
        if not vlf.split:
            self.selected_base     = prev_base
            self.selected_filepath = prev_filepath
            if prev_base:
                self.lbl_file.config(text=prev_base)
            messagebox.showwarning(
                "Nessun burst",
                f"Non ci sono segmenti di segnale validi in {os.path.basename(filepath)}."
            )
            return

        # Burst validi: ora aggiorna UI
        self.lbl_file.config(text=base)
        try:
            self.update_details_tab(filepath)
        except AttributeError:
            pass

        # Carica DataFrame detections
        df = pd.read_pickle(self.current_pkl)
        df['Start_Time'] = pd.to_datetime(df['Start_Time'])
        df['End_Time']   = pd.to_datetime(df['End_Time'])

        # Prepara detections globali
        data0 = vlf.split[0]
        data0['DateTime'] = pd.to_datetime(data0['DateTime'])
        self.t0_global = data0['DateTime'].min()
        self.dets_global = [
            [
                (r['Start_Time'] - self.t0_global).total_seconds(),
                (r['End_Time']   - self.t0_global).total_seconds(),
                r['Start_Freq'] / 1e3,
                r['End_Freq']   / 1e3,
                r['D0']
            ]
            for _, r in df[df['D0'] > 0].iterrows()
        ]

        # Costruisci e mostra spettrogrammi
        self.splits = vlf.split
        self.update_spec_pages()
        self.show_spec_page()

        # Pulisci frame detections e aggiorna istogramma
        self._clear_detections_frame()
        self.update_histogram()

        # Torna al tab spettrogramma
        self.notebook.select(self.spec_tab)


    def update_display_scale(self):
        try:
            new_scale = float(self.scale_entry.get())
            if not (0.1 <= new_scale <= 1.0):
                raise ValueError
            self.display_scale = new_scale
            self.show_spec_page()
            if self.images:
                self.show_image()
        except ValueError:
            messagebox.showerror(
                "Errore scala",
                "Inserire un valore numerico tra 0.1 e 1.0"
            )
            self.scale_entry.delete(0, tk.END)
            self.scale_entry.insert(0, str(self.display_scale))

    def show_overlay(self):
        if not self.overlay:
            self.overlay = tk.Toplevel(self.root)
            self.overlay.overrideredirect(True)
            self.overlay.attributes('-alpha', 0.7)
            self.overlay.configure(bg='white')
            x = self.root.winfo_rootx()
            y = self.root.winfo_rooty()
            w = self.root.winfo_width()
            h = self.root.winfo_height()
            self.overlay.geometry(f"{w}x{h}+{x}+{y}")
            spinner = ttk.Progressbar(self.overlay, mode='indeterminate')
            spinner.place(relx=0.5, rely=0.5, anchor='center')
            spinner.start()
            self.overlay.update()

    def hide_overlay(self):
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None

    def show_loading(self, message):
        """
        Mostra un overlay con spinner e messaggio personalizzato.
        """
        # Rimuovi overlay esistente
        if self.overlay:
            self.overlay.destroy()
        # Crea nuovo overlay
        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True)
        self.overlay.attributes('-alpha', 0.7)
        self.overlay.configure(bg='white')
        # Posiziona full-window
        x = self.root.winfo_rootx()
        y = self.root.winfo_rooty()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        self.overlay.geometry(f"{w}x{h}+{x}+{y}")
        # Spinner
        spinner = ttk.Progressbar(self.overlay, mode='indeterminate')
        spinner.place(relx=0.5, rely=0.45, anchor='center')
        spinner.start()
        # Messaggio
        lbl = ttk.Label(self.overlay, text=message)
        lbl.place(relx=0.5, rely=0.55, anchor='center')
        self.overlay.update()


    def threaded_task(self, func):
        self.show_overlay()
        def worker():
            try:
                func()
            finally:
                self.root.after(0, self.hide_overlay)
        threading.Thread(target=worker, daemon=True).start()


    def _build_spectrogram_tab(self):
        self.spec_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.spec_tab, text='Spettrogramma')

        ctrl = tk.Frame(self.spec_tab)
        ctrl.pack(fill='x', pady=5, padx=5)

        self.btn_gen = ttk.Button(
            ctrl,
            text='Generate Detections',
            command=lambda: self.threaded_task(self.generate_detections)
        )
        self.btn_gen.pack(side='left', padx=5)

        tk.Label(ctrl, text='D0 ≥').pack(side='left', padx=(10,2))
        self.d0_entry = ttk.Entry(ctrl, width=5)
        self.d0_entry.pack(side='left')
        self.d0_entry.insert(0, str(self.d0_threshold))
        self.d0_entry.bind('<Return>', lambda e: self.threaded_task(self.update_threshold))

        container = tk.Frame(self.spec_tab)
        container.pack(fill='both', expand=True)
        self.spec_h = tk.Scrollbar(container, orient='horizontal')
        self.spec_h.pack(side='bottom', fill='x')
        self.spec_v = tk.Scrollbar(container, orient='vertical')
        self.spec_v.pack(side='right', fill='y')
        self.spec_c = tk.Canvas(
            container,
            xscrollcommand=self.spec_h.set,
            yscrollcommand=self.spec_v.set
        )
        self.spec_c.pack(side='left', fill='both', expand=True)
        self.spec_h.config(command=self.spec_c.xview)
        self.spec_v.config(command=self.spec_c.yview)
        self.spec_frame = tk.Frame(self.spec_c)
        self.spec_c.create_window((0,0), window=self.spec_frame, anchor='nw')
        self.spec_frame.bind(
            '<Configure>',
            lambda e: self.spec_c.config(scrollregion=self.spec_c.bbox('all'))
        )

        bottom = tk.Frame(self.spec_tab)
        bottom.pack(fill='x', pady=5)
        inner = tk.Frame(bottom)
        inner.pack(expand=True)
        self.prev_spec = ttk.Button(inner, text='〈 Prev', command=lambda: self.threaded_task(self.prev_spec_page))
        self.prev_spec.pack(side='left', padx=5)
        self.spec_counter = ttk.Label(inner, text='Pagina 0 di 0')
        self.spec_counter.pack(side='left', padx=5)
        self.next_spec = ttk.Button(inner, text='Next 〉', command=lambda: self.threaded_task(self.next_spec_page))
        self.next_spec.pack(side='left', padx=5)
        self.spec_page_var = tk.StringVar()
        self.spec_entry = ttk.Entry(inner, textvariable=self.spec_page_var, width=5)
        self.spec_entry.pack(side='left', padx=5)
        self.spec_entry.bind('<Return>', lambda e: self.threaded_task(self.go_to_spec_page))
        self.spec_go = ttk.Button(inner, text='Vai', command=lambda: self.threaded_task(self.go_to_spec_page))
        self.spec_go.pack(side='left', padx=5)


    def _build_detections_tab(self):
        self.det_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.det_tab, text='Detections')
        container = tk.Frame(self.det_tab)
        container.pack(fill='both', expand=True)
        self.det_h = tk.Scrollbar(container, orient='horizontal')
        self.det_h.pack(side='bottom', fill='x')
        self.det_v = tk.Scrollbar(container, orient='vertical')
        self.det_v.pack(side='right', fill='y')
        self.det_c = tk.Canvas(
            container,
            xscrollcommand=self.det_h.set,
            yscrollcommand=self.det_v.set
        )
        self.det_c.pack(side='left', fill='both', expand=True)
        self.det_h.config(command=self.det_c.xview)
        self.det_v.config(command=self.det_c.yview)
        self.det_frame = tk.Frame(self.det_c)
        self.det_c.create_window((0,0), window=self.det_frame, anchor='nw')
        self.det_frame.bind(
            '<Configure>',
            lambda e: self.det_c.config(scrollregion=self.det_c.bbox('all'))
        )
        bottom = tk.Frame(self.det_tab)
        bottom.pack(fill='x', pady=5)
        inner = tk.Frame(bottom)
        inner.pack(expand=True)
        self.prev_btn = ttk.Button(inner, text='〈 Prev', command=self.prev_image)
        self.prev_btn.pack(side='left', padx=5)
        self.counter  = ttk.Label(inner, text='Immagine 0 di 0')
        self.counter.pack(side='left', padx=5)
        self.next_btn = ttk.Button(inner, text='Next 〉', command=self.next_image)
        self.next_btn.pack(side='left', padx=5)


    def _build_histogram_tab(self):
        self.hist_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.hist_tab, text='Istogramma')


    def _build_details_tab(self):
        self.details_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.details_tab, text='Dettagli')
        
        text = tk.Text(self.details_tab, wrap='none')
        vsb  = ttk.Scrollbar(self.details_tab, orient='vertical',   command=text.yview)
        hsb  = ttk.Scrollbar(self.details_tab, orient='horizontal', command=text.xview)
        text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        text.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        self.details_tab.rowconfigure(0, weight=1)
        self.details_tab.columnconfigure(0, weight=1)
        self.details_text = text
        self.details_text.config(state='disabled')


    def update_details_tab(self, filepath):
        # sblocco la Text per aggiornare
        self.details_text.config(state='normal')
        # svuota
        self.details_text.delete('1.0', tk.END)

        # leggi metadati
        try:
            with h5py.File(filepath, 'r') as f:
                utc_raw  = f['UTC_TIME'][:].flatten()
                utc_strs = [f"{int(x):017d}" for x in utc_raw]
                times    = pd.to_datetime(utc_strs, format='%Y%m%d%H%M%S%f')
                start_ts = times[0]
                end_ts   = times[-1]
                duration = (end_ts - start_ts).total_seconds()
                lat = float(np.array(f['GEO_LAT'])[:].flatten()[0])
                lon = float(np.array(f['GEO_LON'])[:].flatten()[0])
        except Exception as e:
            self.details_text.insert('1.0', f"Errore lettura H5: {e}")
            return

        # reverse-geocoding (fallback se UnicodeDecodeError)
        coord = (lat, lon)
        try:
            res = rg.search(coord, mode=1)[0]
            country_code = res.get('cc', '')
            region_name  = res.get('admin1', '')
        except UnicodeDecodeError:
            country_code = ''
            region_name  = ''

        # ricavo anche il continente
        try:
            cont_code = pc.country_alpha2_to_continent_code(country_code)
            continent = pc.convert_continent_code_to_continent_name(cont_code)
        except Exception:
            continent = ''
        
        import pycountry
        country_obj = pycountry.countries.get(alpha_2=country_code)
        country = country_obj.name if country_obj else ''
        location_str = ", ".join(filter(None, [continent, region_name, country]))

        # formatta e inserisci
        lines = [
            f"File:      {os.path.basename(filepath)}",
            f"Durata:    {duration:.2f} s",
            f"Inizio:    {start_ts}",
            f"Fine:      {end_ts}",
            f"Lat/Lon:   {lat:.4f}, {lon:.4f}",
            f"Luogo:     {location_str}",
            ""
        ]
        self.details_text.insert('1.0', "\n".join(lines))
        # per rendere il testo non modificabile
        self.details_text.config(state='disabled')


    def process_and_load(self):
        # Se non abbiamo selezionato nulla, esci
        if not getattr(self, 'selected_filepath', None):
            return

        # Salvo lo stato corrente per eventuale rollback
        prev_base     = self.selected_base
        prev_filepath = self.selected_filepath

        # Percorsi e nomi
        filepath = self.selected_filepath
        dirpath  = os.path.dirname(filepath)
        h5_name  = os.path.basename(filepath)
        base     = self.selected_base

        # Prepara cartelle di output
        fig_dir = os.path.join(FIG_FOLDER, base)
        det_dir = os.path.join(fig_dir, 'detections_with_kernels')
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(det_dir, exist_ok=True)
        self.current_fig_dir = fig_dir
        self.current_det_dir = det_dir
        self.current_pkl     = os.path.join(PKL_FOLDER, base + '.pkl')

        # Carica con EFD
        reader = EFD()
        vlf    = reader.read(dirpath, h5_name)

        # Se split è None o lista vuota -> rollback e avviso
        if not vlf.split:
            # rollback della selezione
            prev_base = getattr(self, '_prev_selected_base', None)
            prev_fp   = getattr(self, '_prev_selected_filepath', None)
            self.selected_base     = prev_base
            self.selected_filepath = prev_fp
            if prev_base:
                self.lbl_file.config(text=prev_base)
            messagebox.showwarning(
                "Nessun burst",
                f"Non ci sono segmenti di segnale validi in {h5_name}."
            )
            return

        # Solo su split valido: aggiorna UI
        self.lbl_file.config(text=base)
        try:
            self.update_details_tab(filepath)
        except AttributeError:
            pass  # se non esiste la tab Dettagli

        # Carica il DataFrame delle detections
        df = pd.read_pickle(self.current_pkl)
        df['Start_Time'] = pd.to_datetime(df['Start_Time'])
        df['End_Time']   = pd.to_datetime(df['End_Time'])

        # Prepara detections globali
        data0 = vlf.split[0]
        data0['DateTime'] = pd.to_datetime(data0['DateTime'])
        self.t0_global = data0['DateTime'].min()
        self.dets_global = [
            [
                (r['Start_Time'] - self.t0_global).total_seconds(),
                (r['End_Time']   - self.t0_global).total_seconds(),
                r['Start_Freq'] / 1e3,
                r['End_Freq']   / 1e3,
                r['D0']
            ]
            for _, r in df[df['D0'] > 0].iterrows()
        ]

        # Popola le pagine di spettrogramma
        self.splits = vlf.split
        self.update_spec_pages()
        self.show_spec_page()

        # Prepara e mostra detections
        self._clear_detections_frame()

        # Aggiorna istogramma
        self.update_histogram()

        # Torna alla scheda spettrogramma
        self.notebook.select(self.spec_tab)


    def update_spec_pages(self):
        spectra = Spectra()
        self.spec_pages = []
        for idx, block in enumerate(self.splits):
            block['DateTime'] = pd.to_datetime(block['DateTime'])
            sig = np.asarray(list(chain.from_iterable(block.Signal.values)))
            freq = block.Frequency.values[0]
            freqs, times, spec = spectra.spectrogram(sig, freq)
            times *= 2
            blk_start = (block['DateTime'].min() - self.t0_global).total_seconds()
            blk_end   = blk_start + PAGE_SIZE_SECONDS
            dets_blk = [
                [d[0]-blk_start, d[1]-blk_start, d[2], d[3], d[4]]
                for d in self.dets_global
                if d[0] >= blk_start and d[1] <= blk_end and d[4] >= self.d0_threshold
            ]
            plotter = SpectrogramPlot()
            fig_sz = plotter.calculate_figure_size(
                spectra.get_time_freq_ratio(times, freqs)
            )
            fname = os.path.join(self.current_fig_dir, f'spectrogram_page_{idx}.png')
            plotter.plot(
                spec, times, freqs,
                figsize=fig_sz,
                detections=dets_blk,
                save=True,
                file_name=fname
            )
            self.spec_pages.append(fname)


    def show_spec_page(self):
        for w in self.spec_frame.winfo_children():
            w.destroy()
        if not self.spec_pages:
            self.spec_counter.config(text='Pagina 0 di 0')
            return
        img = Image.open(self.spec_pages[self.page_index])
        if self.display_scale != 1.0:
            w, h = img.size
            img = img.resize((int(w*self.display_scale), int(h*self.display_scale)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(self.spec_frame, image=photo)
        lbl.image = photo
        lbl.pack()
        self.spec_c.update_idletasks()
        self.spec_c.config(scrollregion=self.spec_c.bbox('all'))
        self.spec_counter.config(
            text=f'Pagina {self.page_index+1} di {len(self.spec_pages)}'
        )
        self.prev_spec.config(
            state='normal' if self.page_index>0 else 'disabled'
        )
        self.next_spec.config(
            state='normal' if self.page_index<len(self.spec_pages)-1 else 'disabled'
        )


    def prev_spec_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self.show_spec_page()


    def next_spec_page(self):
        if self.page_index < len(self.spec_pages)-1:
            self.page_index += 1
            self.show_spec_page()


    def go_to_spec_page(self):
        try:
            page = int(self.spec_page_var.get()) - 1
            if 0 <= page < len(self.spec_pages):
                self.page_index = page
                self.show_spec_page()
            else:
                messagebox.showerror(
                    'Errore pagina',
                    f'Inserisci un numero tra 1 e {len(self.spec_pages)}'
                )
        except ValueError:
            messagebox.showerror('Errore pagina', 'Inserisci un numero intero valido')


    def update_threshold(self):
        try:
            val = float(self.d0_entry.get())
            if val < 0:
                raise ValueError
            self.d0_threshold = val
            self.update_spec_pages()
            self.show_spec_page()
        except ValueError:
            messagebox.showerror('Errore soglia', 'Inserisci un valore numerico ≥ 0')


    def generate_detections(self):
        self.show_loading("Generazione detections…")
        try:
            self._clear_detections_dir()
            self._clear_detections_frame()
            block = self.splits[self.page_index]
            block['DateTime'] = pd.to_datetime(block['DateTime'])
            sig = np.asarray(list(chain.from_iterable(block.Signal.values)))
            freq = block.Frequency.values[0]
            spec_reader = Spectra()
            freqs, times, spec = spec_reader.spectrogram(sig, freq)
            times *= 2
            blk_start = (block['DateTime'].min() - self.t0_global).total_seconds()
            dets_blk = [
                [d[0]-blk_start, d[1]-blk_start, d[2], d[3], d[4]]
                for d in self.dets_global
                if d[0] >= blk_start and d[1] <= blk_start+PAGE_SIZE_SECONDS and d[4] >= self.d0_threshold
            ]
            t_res = spec_reader.get_time_res(times)
            f_res = spec_reader.get_freq_res(freqs)
            low_f, high_f, fn, d0, d0min, d0max = awds_util.get_value_base_on_l(block.L.values[0])
            viz = WhistlerDetectionVisualizer(
                plot_obj=SpectrogramPlot(),
                padding_factor=0.3,
                kernel_alpha=0.4,
                output_dir=self.current_det_dir
            )
            viz.process_all_detections(
                spectrogram=spec,
                time=times,
                freqs=freqs,
                detections=dets_blk,
                t_res=t_res,
                f_res=f_res,
                low_f=low_f,
                high_f=high_f,
                fn=fn,
                kernel_cmap='transparent_blue'
            )
            viz.process_all_detections(
                spectrogram=spec,
                time=times,
                freqs=freqs,
                detections=dets_blk,
                t_res=t_res,
                f_res=f_res,
                low_f=low_f,
                high_f=high_f,
                fn=fn,
                kernel_cmap='transparent_blue',
                show_kernel=False
            )
            self.images = sorted(
                [f for f in os.listdir(self.current_det_dir)
                 if f.lower().endswith(('.png','.jpg','.jpeg'))]
            )
        except Exception as e:
            self.hide_overlay()
            self.root.after(0, lambda err=e: messagebox.showerror("Errore generazione", str(err)))
            return

        self.hide_overlay()
        self.index = 0
        self.show_image()



    def _clear_detections_dir(self):
        for f in os.listdir(self.current_det_dir):
            path = os.path.join(self.current_det_dir, f)
            if os.path.isfile(path):
                os.remove(path)


    def _clear_detections_frame(self):
        for w in self.det_frame.winfo_children():
            w.destroy()
        self.counter.config(text='Immagine 0 di 0')


    def show_image(self):
        # Pulisci il frame delle detections
        for w in self.det_frame.winfo_children():
            w.destroy()
        # Se non ci sono immagini, mostra contatore a zero
        if not self.images:
            self.counter.config(text='Immagine 0 di 0')
            return

        # Percorso file immagine
        fp = os.path.join(self.current_det_dir, self.images[self.index])
        img = Image.open(fp)

        # Applica scala se diversa da 1.0
        if self.display_scale != 1.0:
            w, h = img.size
            img = img.resize(
                (int(w * self.display_scale), int(h * self.display_scale)),
                Image.LANCZOS
            )

        # Padding per larghezza minima (dopo il resize)
        w0, h0 = img.size
        if w0 < MIN_DET_WIDTH:
            # Creo un'immagine bianca della larghezza minima
            new_img = Image.new(img.mode, (MIN_DET_WIDTH, h0), 'white')
            # Centro l'immagine originale orizzontalmente
            x_off = (MIN_DET_WIDTH - w0) // 2
            new_img.paste(img, (x_off, 0))
            img = new_img

        # Converte in PhotoImage e mostra
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(self.det_frame, image=photo)
        lbl.image = photo
        lbl.pack()

        # Aggiorna contatore e seleziona tab Detections
        self.counter.config(text=f'Immagine {self.index+1} di {len(self.images)}')
        self.notebook.select(self.det_tab)


    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()


    def next_image(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self.show_image()


    def update_histogram(self):
        for w in self.hist_tab.winfo_children():
            w.destroy()
        base = self.selected_base
        pkl = os.path.join(PKL_FOLDER, base + '.pkl')
        HistogramPlotter(self.hist_tab, pkl).plot_histogram()
        self.notebook.select(self.spec_tab)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == '__main__':
    main()
