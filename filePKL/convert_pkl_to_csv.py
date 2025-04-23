import os
import pandas as pd

def convert_all_pkl_to_csv(source_dir, destination_dir):
    """
    Converte tutti i file .pkl nella directory source_dir in file .csv,
    salvandoli nella directory destination_dir.
    """
    # Crea la cartella di destinazione se non esiste
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Ottieni la lista dei file .pkl nella cartella di origine
    pkl_files = [file for file in os.listdir(source_dir) if file.endswith(".pkl")]
    total_files = len(pkl_files)
    print(f"Trovati {total_files} file .pkl nella cartella {source_dir}.")
    
    # Itera su ogni file .pkl e convertilo in CSV
    for idx, file in enumerate(pkl_files, start=1):
        pkl_path = os.path.join(source_dir, file)
        csv_filename = os.path.splitext(file)[0] + ".csv"
        csv_path = os.path.join(destination_dir, csv_filename)
        
        print(f"[{idx}/{total_files}] Elaborazione del file: {file}")
        df = pd.read_pickle(pkl_path)
        df.to_csv(csv_path, index=False)
        print(f"   Convertito: {pkl_path} -> {csv_path}\n")

if __name__ == "__main__":
    # Definisci il percorso di origine e destinazione
    source_directory = "C:/Users/Ivan/PycharmProjects/visualizer-Donato-Ivan/visualizer-Ivan/visualizer/ENV_TEST/filePKL"
    destination_directory = "C:/Users/Ivan/PycharmProjects/visualizer-Donato-Ivan/visualizer-Ivan/visualizer/ENV_TEST/filePKL"
    
    convert_all_pkl_to_csv(source_directory, destination_directory)
