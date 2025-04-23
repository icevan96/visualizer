import sys
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
sys_base = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
H5_FOLDER  = os.path.join(sys_base, "fileH5")
PKL_FOLDER = os.path.join(sys_base, "filePKL")

SAMPLING_FREQUENCY = 51200
COLORMAPS = ['jet', 'viridis', 'plasma', 'inferno', 'cividis', 'magma', 'turbo', 'nipy_spectral']


def load_signal_and_time(h5_file):
    with h5py.File(h5_file, 'r') as f:
        signal = np.concatenate(f['A131_W'][:].tolist())
        utc = f['UTC_TIME'][()].flatten()
    t0 = datetime.strptime(str(utc[0]), "%Y%m%d%H%M%S%f")
    return signal, t0


def plot_spectrogram_section(signal, t0, t_start, t_end, cmap, name, show_detections=True):
    spec_obj = Spectra()
    freqs, times, spec = spec_obj.spectrogram(signal, SAMPLING_FREQUENCY)
    start_sec = (t_start - t0).total_seconds()
    end_sec = (t_end - t0).total_seconds()
    idx = np.where((times >= start_sec) & (times <= end_sec))[0]
    if len(idx) == 0:
        return None
    t1, t2 = idx[0], idx[-1] + 1
    tr = times[t1:t2]
    sr = spec[:, t1:t2]
    dtimes = [t0 + timedelta(seconds=s) for s in tr]
    duration_s = tr[-1] - tr[0] if len(tr) > 1 else 1
    band_khz = freqs[-1] - freqs[0] if len(freqs) > 1 else freqs[-1]
    fig, ax = plt.subplots(figsize=(duration_s / 2.54, band_khz / 2.54))
    pcm = ax.pcolormesh(dtimes, freqs, sr, cmap=cmap)
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Frequenza [kHz]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.colorbar(pcm, ax=ax, label="Magnitudo [dB]")

    if show_detections:
        pkl_file = os.path.join(PKL_FOLDER, name + ".pkl")
        try:
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
                rect = patches.Rectangle((x1n, row['y1']), w, row['y2'] - row['y1'], edgecolor='white', facecolor='none', linewidth=1)
                ax.add_patch(rect)
                ax.text(x1n, row['y2'], f"{row['D0']:.0f}", color='red', va='bottom', fontsize=9)
        except FileNotFoundError:
            pass

    plt.tight_layout()
    return fig


def plot_spectrogram_basic_section(signal, signal_start_time, t_start, t_end, cmap):
    spec_obj = Spectra()
    freqs, times, spec = spec_obj.spectrogram(signal, SAMPLING_FREQUENCY)
    start_sec = (t_start - signal_start_time).total_seconds()
    end_sec = (t_end - signal_start_time).total_seconds()
    idx = np.where((times >= start_sec) & (times <= end_sec))[0]
    if len(idx) == 0:
        messagebox.showinfo("Intervallo non valido", "Nessun dato nell'intervallo richiesto.")
        return None
    t1, t2 = idx[0], idx[-1] + 1
    tr = times[t1:t2]
    sr = spec[:, t1:t2]
    dtimes = [signal_start_time + timedelta(seconds=s) for s in tr]
    fig, ax = plt.subplots(figsize=(len(tr)/20, (freqs[-1]/1000)/2))
    pcm = ax.pcolormesh(dtimes, freqs/1e3, sr, cmap=cmap)
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

    # overlay spinner and dimming
    overlay = None
    def show_overlay():
        nonlocal overlay
        if overlay is None:
            overlay = tk.Toplevel(root)
            overlay.overrideredirect(True)
            overlay.attributes('-alpha', 0.3)
            overlay.configure(bg='black')
            overlay.geometry(f"{root.winfo_width()}x{root.winfo_height()}+{root.winfo_rootx()}+{root.winfo_rooty()}")
            spinner = ttk.Progressbar(overlay, mode='indeterminate')
            spinner.place(relx=0.5, rely=0.5, anchor='center')
            spinner.start()
            overlay.update()

    def hide_overlay():
        nonlocal overlay
        if overlay is not None:
            overlay.destroy()
            overlay = None

    # file selection
    files = [f[:-3] for f in os.listdir(H5_FOLDER) if f.endswith('.h5')]
    file_frame = tk.Frame(root)
    file_frame.pack(fill='x', pady=5)
    tk.Label(file_frame, text="File H5:").pack(side='left', padx=5)
    file_cb = ttk.Combobox(file_frame, values=files, width=80, state='readonly')
    file_cb.pack(side='left', fill='x', expand=True, padx=5)
    if files:
        file_cb.set(files[0])

    # controls: detections + colormap
    ctrl_frame = tk.Frame(root)
    ctrl_frame.pack(fill='x', pady=5)
    show_det_var = tk.BooleanVar(value=False)
    det_cb = ttk.Checkbutton(ctrl_frame, text="Detections", variable=show_det_var, command=lambda: threaded_task(auto_plot_full))
    det_cb.pack(side='left', padx=5)
    tk.Label(ctrl_frame, text="Colormap:").pack(side='left', padx=5)
    cmap_cb = ttk.Combobox(ctrl_frame, values=COLORMAPS, width=20, state='readonly')
    cmap_cb.pack(side='left', padx=5)
    cmap_cb.set(COLORMAPS[0])
    cmap_cb.bind("<<ComboboxSelected>>", lambda e: threaded_task(auto_plot_full))

    # interval controls
    interval_frame = tk.Frame(root)
    interval_frame.pack(fill='x', pady=5)
    tk.Label(interval_frame, text="Start:").grid(row=0, column=0, padx=5, sticky='e')
    start_cb = ttk.Combobox(interval_frame, width=30, state='readonly')
    start_cb.grid(row=0, column=1, padx=5)
    tk.Label(interval_frame, text="End:").grid(row=0, column=2, padx=5, sticky='e')
    end_cb = ttk.Combobox(interval_frame, width=30, state='readonly')
    end_cb.grid(row=0, column=3, padx=5)
    plot_btn = ttk.Button(interval_frame, text="Plot", command=lambda: threaded_task(plot_interval))
    plot_btn.grid(row=0, column=4, padx=10)

    # notebook
    notebook = ttk.Notebook(root)
    frame_full = ttk.Frame(notebook)
    frame_int = ttk.Frame(notebook)
    frame_hist = ttk.Frame(notebook)
    notebook.add(frame_full, text='Visualizzazione')
    notebook.add(frame_int, text='Intervallo')
    notebook.add(frame_hist, text='Istogramma')
    notebook.pack(expand=True, fill='both')

    def clear_interval():
        for w in frame_int.winfo_children():
            w.destroy()

    def auto_plot_full():
        name = file_cb.get().strip()
        h5_path = os.path.join(H5_FOLDER, name + ".h5")
        signal, t0 = load_signal_and_time(h5_path)
        dur = len(signal) / SAMPLING_FREQUENCY
        fig = plot_spectrogram_section(signal, t0, t0, t0 + timedelta(seconds=dur), cmap_cb.get(),
                                       name, show_detections=show_det_var.get())
        if fig:
            display_figure_in_canvas(fig, frame_full, scrollable=True)
        update_start_end()

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
            display_figure_in_canvas(fig, frame_int, scrollable=False)

    def display_figure_in_canvas(fig, frame, scrollable=True):
        for w in frame.winfo_children(): w.destroy()
        if scrollable:
            container = tk.Frame(frame)
            container.pack(fill='both', expand=True)
            hbar = tk.Scrollbar(container, orient='horizontal')
            hbar.pack(side='bottom', fill='x')
            vbar = tk.Scrollbar(container, orient='vertical')
            vbar.pack(side='right', fill='y')
            canvas = tk.Canvas(container, xscrollcommand=hbar.set, yscrollcommand=vbar.set)
            canvas.pack(side='left', fill='both', expand=True)
            hbar.config(command=canvas.xview)
            vbar.config(command=canvas.yview)
            plot_container = tk.Frame(canvas)
            canvas.create_window((0,0), window=plot_container, anchor='nw')
            plot_container.bind('<Configure>', lambda e: canvas.config(scrollregion=canvas.bbox('all')))
            canvas_tk = FigureCanvasTkAgg(fig, master=plot_container)
            canvas_tk.draw()
            canvas_tk.get_tk_widget().pack(fill='both', expand=True)
            toolbar = tk.Frame(frame)
            toolbar.pack(side='bottom', fill='x')
            NavigationToolbar2Tk(canvas_tk, toolbar).update()
        else:
            canvas_tk = FigureCanvasTkAgg(fig, master=frame)
            canvas_tk.draw()
            canvas_tk.get_tk_widget().pack(fill='both', expand=True)
        notebook.select(frame)

    def threaded_task(func):
        show_overlay()
        root.update()
        try:
            func()
        finally:
            hide_overlay()

    def update_histogram_tab(name):
        pkl_path = os.path.join(PKL_FOLDER, name + ".pkl")
        plotter = HistogramPlotter(frame_hist, pkl_path)
        plotter.plot_histogram()

    def update_start_end():
        name = file_cb.get().strip()
        pkl_path = os.path.join(PKL_FOLDER, name + ".pkl")
        try:
            df = pd.read_pickle(pkl_path)
            df['Start_Time'] = pd.to_datetime(df['Start_Time'])
            starts = df['Start_Time'].dt.strftime("%Y-%m-%d %H:%M:%S.%f").tolist()
            unique_starts = sorted(set(starts))
            start_cb['values'] = unique_starts
            if unique_starts:
                start_cb.set(unique_starts[0])
                on_start_change()
        except Exception as e:
            messagebox.showerror("Errore PKL", f"Impossibile leggere {pkl_path}:\n{e}")

    def on_start_change(event=None):
        sel = start_cb.get().strip()
        if not sel:
            return
        name = file_cb.get().strip()
        pkl_path = os.path.join(PKL_FOLDER, name + ".pkl")
        try:
            df = pd.read_pickle(pkl_path)
            df['Start_Time'] = pd.to_datetime(df['Start_Time'])
            df['End_Time'] = pd.to_datetime(df['End_Time'])
            sel_dt = datetime.strptime(sel, "%Y-%m-%d %H:%M:%S.%f")
            ends = df[df['Start_Time'] == sel_dt]['End_Time'] \
                       .dt.strftime("%Y-%m-%d %H:%M:%S.%f").tolist()
            unique_ends = sorted(set(ends))
            end_cb['values'] = unique_ends
            if unique_ends:
                end_cb.set(unique_ends[0])
        except Exception as e:
            messagebox.showerror("Errore PKL", f"Impossibile leggere {pkl_path}:\n{e}")

    file_cb.bind("<<ComboboxSelected>>", lambda e: threaded_task(lambda: [clear_interval(), update_start_end(), auto_plot_full(), update_histogram_tab(file_cb.get().strip())]))
    start_cb.bind("<<ComboboxSelected>>", lambda e: threaded_task(plot_interval))

    # avvio iniziale
    threaded_task(lambda: [update_start_end(), auto_plot_full(), update_histogram_tab(file_cb.get().strip())])

    root.mainloop()

if __name__ == '__main__':
    run_interface()
