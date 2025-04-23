import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class HistogramPlotter:
    def __init__(self, frame, pkl_path):
        self.frame = frame
        self.pkl_path = pkl_path

    def clear_frame(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

    def plot_histogram(self):
        self.clear_frame()
        try:
            df = pd.read_pickle(self.pkl_path)
            if 'Driver_corr' not in df.columns:
                raise ValueError("La colonna 'Driver_corr' non è presente nel file.")

            values = df['Driver_corr'].dropna()
            bins = np.arange(1, 101, 5)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(values, bins=bins, edgecolor='black')
            ax.set_xlabel("Valore di Driver_corr")
            ax.set_ylabel("Conteggio eventi")
            ax.set_title("Istogramma Driver_corr")
            ax.grid(True)

            canvas = FigureCanvasTkAgg(fig, master=self.frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

        except Exception as e:
            label = tk.Label(self.frame, text=f"Errore nel caricamento dell'istogramma:\n{e}", fg="red")
            label.pack(fill='both', expand=True)
