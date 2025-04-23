import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import messagebox, ttk
from awds.spectra import Spectra
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from histogram import HistogramPlotter

# === Configurazione ===
H5_FOLDER = r"C:\\Users\\Ivan\\PycharmProjects\\visualizer-Donato-Ivan\\visualizer-Ivan\\fileH5"
PKL_FOLDER = r"C:\\Users\\Ivan\\PycharmProjects\\visualizer-Donato-Ivan\\visualizer-Ivan\\filePKL"
SAMPLING_FREQUENCY = 51200
COLORMAPS = ['jet','viridis','plasma','inferno','cividis','magma','turbo','nipy_spectral']

def load_signal_and_time(h5_file):
    with h5py.File(h5_file, 'r') as f:
        signal = np.concatenate(f['A131_W'][:].tolist())
        utc = f['UTC_TIME'][()].flatten()
    t0 = datetime.strptime(str(utc[0]), "%Y%m%d%H%M%S%f")
    return signal, t0

def plot_spectrogram_section(signal, t0, t_start, t_end, cmap, name, show_detections=True):
    spec_obj = Spectra()
    freqs, times, spec = spec_obj.spectrogram(signal, 51200)
    start_sec = (t_start - t0).total_seconds()
    end_sec   = (t_end - t0).total_seconds()
    idx = np.where((times >= start_sec) & (times <= end_sec))[0]
    if len(idx) == 0:
        return None
    t1, t2 = idx[0], idx[-1] + 1
    tr = times[t1:t2]
    sr = spec[:, t1:t2]
    dtimes = [t0 + timedelta(seconds=s) for s in tr]
    duration_s = tr[-1] - tr[0]
    band_khz   = freqs[-1] - freqs[0]
    fig, ax = plt.subplots(figsize=(duration_s/2.54, band_khz/2.54))
    pcm = ax.pcolormesh(dtimes, freqs, sr, cmap=cmap)
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Frequenza [kHz]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.colorbar(pcm, ax=ax, label="Magnitudo [dB]")
    if show_detections:
        pkl_file = os.path.join(PKL_FOLDER, name + ".pkl")
        df = pd.read_pickle(pkl_file)
        df['Start_Time'] = pd.to_datetime(df['Start_Time'])
        df['End_Time'] = pd.to_datetime(df['End_Time'])
        df['x1'] = (df['Start_Time'] - t0).dt.total_seconds()
        df['x2'] = (df['End_Time'] - t0).dt.total_seconds()
        df['y1'] = df['Start_Freq'] / 1000.0
        df['y2'] = df['End_Freq'] / 1000.0
        df = df[(df['x1'] >= start_sec) & (df['x2'] <= end_sec) & (df['D0'] > 0) & ((df['y2'] - df['y1']) <= 6)]
        for _, row in df.iterrows():
            dt1 = t0 + timedelta(seconds=row['x1'])
            dt2 = t0 + timedelta(seconds=row['x2'])
            x1n = mdates.date2num(dt1)
            w = mdates.date2num(dt2) - x1n
            rect = patches.Rectangle((x1n, row['y1']), w, row['y2'] - row['y1'],
                                     edgecolor='white', facecolor='none', linewidth=1)
            ax.add_patch(rect)
            ax.text(x1n, row['y2'], f"{row['D0']:.0f}", color='red', va='bottom', fontsize=9)
    plt.tight_layout()
    return fig

def plot_spectrogram_basic_section(signal, signal_start_time, t_start, t_end, cmap):
    spectra_obj = Spectra()
    freqs, times, spec = spectra_obj.spectrogram(signal, SAMPLING_FREQUENCY)
    start_sec = (t_start - signal_start_time).total_seconds()
    end_sec = (t_end - signal_start_time).total_seconds()
    time_indices = np.where((times >= start_sec) & (times <= end_sec))[0]
    if len(time_indices) == 0:
        messagebox.showinfo("Intervallo non valido", "Nessun dato nell'intervallo richiesto.")
        return None
    t1, t2 = time_indices[0], time_indices[-1] + 1
    time_range = times[t1:t2]
    spec_range = spec[:, t1:t2]
    time_datetimes = [signal_start_time + timedelta(seconds=t) for t in time_range]
    fig_size = (len(time_range)/20, (freqs[-1]/1000)/2)
    fig, ax = plt.subplots(figsize=fig_size)
    pcm = ax.pcolormesh(time_datetimes, freqs / 1e3, spec_range, cmap=cmap)
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Frequenza [kHz]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    ax.set_title("Spettrogramma intervallo selezionato")
    fig.colorbar(pcm, ax=ax, label="Magnitudo [dB]")
    plt.tight_layout()
    return fig


def run_interface():
    root = tk.Tk()
    root.title("Visualizer")
    root.geometry("1000x700")

    files = [f[:-3] for f in os.listdir(H5_FOLDER) if f.endswith('.h5')]
    file_frame = tk.Frame(root)
    file_frame.pack(fill='x', pady=5)
    tk.Label(file_frame, text="File H5:").pack(side='left', padx=5)
    file_cb = ttk.Combobox(file_frame, values=files, width=80)
    file_cb.pack(side='left', fill='x', expand=True, padx=5)
    if files:
        file_cb.set(files[0])

    show_det_var = tk.BooleanVar(master=root, value=False)
    checkbox_frame = tk.Frame(root)
    checkbox_frame.pack(fill='x', pady=5)
    tk.Checkbutton(checkbox_frame, text="Detections", variable=show_det_var, command=lambda: auto_plot_full()).pack(side='left', padx=5)

    ctrl_frame = tk.Frame(root)
    ctrl_frame.pack(fill='x', pady=5)
    tk.Label(ctrl_frame, text="Start:").grid(row=0, column=0, padx=5, sticky='e')
    start_cb = ttk.Combobox(ctrl_frame, width=30, state='readonly')
    start_cb.grid(row=0, column=1, padx=5)
    tk.Label(ctrl_frame, text="End:").grid(row=0, column=2, padx=5, sticky='e')
    end_cb = ttk.Combobox(ctrl_frame, width=30, state='readonly')
    end_cb.grid(row=0, column=3, padx=5)
    tk.Label(ctrl_frame, text="Colormap:").grid(row=0, column=4, padx=5, sticky='e')
    cmap_cb = ttk.Combobox(ctrl_frame, values=COLORMAPS, width=30)
    cmap_cb.grid(row=0, column=5, padx=5)
    cmap_cb.set(COLORMAPS[0])
    plot_btn = ttk.Button(ctrl_frame, text="Plot", command=lambda: plot_interval())
    plot_btn.grid(row=0, column=6, padx=10)

    notebook = ttk.Notebook(root)
    frame_plot_full = ttk.Frame(notebook)
    frame_plot_interval = ttk.Frame(notebook)
    frame_histogram = ttk.Frame(notebook)
    notebook.add(frame_plot_full, text='Visualizzazione')
    notebook.add(frame_plot_interval, text='Intervallo')
    notebook.add(frame_histogram, text='Istogramma')
    notebook.pack(expand=True, fill='both')

    def display_figure_in_canvas(fig, frame, scrollable=True):
        for widget in frame.winfo_children():
            widget.destroy()

        if scrollable:
            scroll_ct = tk.Frame(frame)
            scroll_ct.pack(fill='both', expand=True)
            hbar = tk.Scrollbar(scroll_ct, orient='horizontal')
            hbar.pack(side='bottom', fill='x')
            vbar = tk.Scrollbar(scroll_ct, orient='vertical')
            vbar.pack(side='right', fill='y')
            canvas = tk.Canvas(scroll_ct, xscrollcommand=hbar.set, yscrollcommand=vbar.set)
            canvas.pack(side='left', fill='both', expand=True)
            hbar.config(command=canvas.xview)
            vbar.config(command=canvas.yview)

            plot_container = tk.Frame(canvas)
            canvas.create_window((0, 0), window=plot_container, anchor='nw')
            plot_container.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))

            canvas_tk = FigureCanvasTkAgg(fig, master=plot_container)
            canvas_tk.draw()
            canvas_tk.get_tk_widget().pack(fill='both', expand=True)

            toolbar_frame = tk.Frame(frame)
            toolbar_frame.pack(side='bottom', fill='x')
            NavigationToolbar2Tk(canvas_tk, toolbar_frame).update()
        else:
            canvas_tk = FigureCanvasTkAgg(fig, master=frame)
            canvas_tk.draw()
            canvas_tk.get_tk_widget().pack(fill='both', expand=True)
        notebook.select(frame)

    def auto_plot_full():
        name = file_cb.get().strip()
        h5_path = os.path.join(H5_FOLDER, name + ".h5")
        signal, t0 = load_signal_and_time(h5_path)
        dur = len(signal) / SAMPLING_FREQUENCY
        t_start, t_end = t0, t0 + timedelta(seconds=dur)
        fig = plot_spectrogram_section(signal, t0, t_start, t_end, cmap_cb.get(), name, show_detections=show_det_var.get())
        if fig:
            display_figure_in_canvas(fig, frame_plot_full, scrollable=True)
        update_histogram_tab(name)

    def plot_interval():
        name = file_cb.get().strip()
        h5_path = os.path.join(H5_FOLDER, name + ".h5")
        signal, t0 = load_signal_and_time(h5_path)
        s = start_cb.get()
        e = end_cb.get()
        fmt = "%Y-%m-%d %H:%M:%S.%f" if '.' in s else "%Y-%m-%d %H:%M:%S"
        t_start = datetime.strptime(s, fmt)
        t_end = datetime.strptime(e, fmt)
        fig = plot_spectrogram_basic_section(signal, t0, t_start, t_end, cmap_cb.get())
        if fig:
            display_figure_in_canvas(fig, frame_plot_interval, scrollable=False)

    def update_histogram_tab(name):
        pkl_path = os.path.join(PKL_FOLDER, name + ".pkl")
        plotter = HistogramPlotter(frame_histogram, pkl_path)
        plotter.plot_histogram()

    def on_file_change(event=None):
        name = file_cb.get()
        pkl = os.path.join(PKL_FOLDER, name + ".pkl")
        try:
            df = pd.read_pickle(pkl)
            df['Start_Time'] = pd.to_datetime(df['Start_Time'])
            df['End_Time'] = pd.to_datetime(df['End_Time'])
            starts = df['Start_Time'].dt.strftime("%Y-%m-%d %H:%M:%S.%f").tolist()
            start_cb['values'] = sorted(set(starts))
            if starts:
                start_cb.set(starts[0])
                on_start_change()
        except Exception as e:
            messagebox.showerror("Errore PKL", str(e))
        auto_plot_full()

    def on_start_change(event=None):
        sel = start_cb.get()
        if not sel:
            return
        fmt = "%Y-%m-%d %H:%M:%S.%f" if '.' in sel else "%Y-%m-%d %H:%M:%S"
        t0 = datetime.strptime(sel, fmt)
        df = pd.read_pickle(os.path.join(PKL_FOLDER, file_cb.get() + ".pkl"))
        df['Start_Time'] = pd.to_datetime(df['Start_Time'])
        df['End_Time'] = pd.to_datetime(df['End_Time'])
        ends = df[df['Start_Time'] > t0]['End_Time'].dt.strftime("%Y-%m-%d %H:%M:%S.%f").tolist()
        end_cb['values'] = sorted(set(ends))
        if ends:
            end_cb.set(ends[0])

    file_cb.bind("<<ComboboxSelected>>", on_file_change)
    start_cb.bind("<<ComboboxSelected>>", on_start_change)
    on_file_change()
    root.mainloop()

if __name__ == '__main__':
    run_interface()
