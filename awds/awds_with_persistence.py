import os
from itertools import chain

import numpy as np

from awds.adaptive_threshold import AdaptiveThreshold
from awds.detector import Detector
from awds.reader import ReaderInterface
from awds.spectra import Spectra
from awds.whistler import WhistlerModel
from awds.persistence import StoreInterface

# Base folder for storing detection pickles
PKL_FOLDER = os.path.join(os.getcwd(), "filePKL")

def get_value_base_on_l(L):
    if L < 1.4:
        # Low L
        low_f, high_f, fn, d0, d0_min, d0_max = 4.5e3, 11.5e3, 75e3, 10, 5, 20
    elif 1.4 <= L < 2.3:
        # Mid L
        low_f, high_f, fn, d0, d0_min, d0_max = 4.5e3, 11.5e3, 25e3, 50, 20, 80
    elif 2.3 <= L < 3.5:
        # Higher L
        low_f, high_f, fn, d0, d0_min, d0_max = 4e3, 10e3, 20e3, 70, 40, 100
    elif 3.5 <= L < 4.5:
        # High L
        low_f, high_f, fn, d0, d0_min, d0_max = 4e3, 8e3, 14e3, 65, 35, 95
    else:
        # Very High L
        low_f, high_f, fn, d0, d0_min, d0_max = 4e3, 8e3, 14e3, 65, 35, 95

    return low_f, high_f, fn, d0, d0_min, d0_max


class AWDS:
    def main(self, reader: ReaderInterface, store: StoreInterface, path, file_name, debug_enabled=False):
        self.__print(debug_enabled, f"Reading File {file_name}")
        # pre-processing data
        
        directory_path = PKL_FOLDER
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        try:
            vlf = reader.read(path, file_name)
            spectra = Spectra()
            df_w = store.create_df()

            loc = 0
            whistlers_file = os.path.join(directory_path, file_name.replace(vlf.file_extension, "pkl"))
            if vlf.split is not None:
                n = 0
                for df in vlf.split:
                    try:
                        start_analysis = (str(df.DateTime.values[0]), str(df.DateTime.values[-1]), 0, 20e3, -1, 0, -1)
                        store.add_info_to_df(df, df_w, loc, start_analysis)
                        loc += 1

                        n += 1
                        L = df.L.values[0]
                        vlf_signal = np.asarray(list(chain.from_iterable(df.Signal.values)))
                        freqs, time, spectrogram = spectra.spectrogram(vlf_signal, df.Frequency.values[0])
                        time = time * 2
                        self.__print(debug_enabled, f"L value {L}")

                        t_res, f_res = spectra.get_time_res(time), spectra.get_freq_res(freqs)
                        low_f, high_f, fn, d0, d0_min, d0_max = get_value_base_on_l(L)
                        lower_freq, upper_freq = low_f / 1e3, high_f / 1e3

                        self.__print(debug_enabled, "Generate Kernel")
                        # generate whistler model for correlation
                        modelW = WhistlerModel(t_res, f_res, low_f, high_f, fn)
                        kernel = modelW.whistler_sim(d0)

                        self.__print(debug_enabled, "Apply Transformations")
                        # apply transformations
                        spectrogramSlice = spectra.apply_slice(lower_freq, upper_freq, freqs, spectrogram)
                        spectrogramSliceZscore = spectra.apply_zscore(spectrogramSlice[0])

                        self.__print(debug_enabled, "Get Correlations")
                        corr = spectra.get_correlation(spectrogramSliceZscore, kernel)

                        self.__print(debug_enabled, "Detecting")
                        # detect process
                        adaptiveThreshold = AdaptiveThreshold()
                        detector = Detector()
                        pulse = adaptiveThreshold.detection_pulse(corr, 'fusion_cfar')
                        self.__print(debug_enabled, f"detection_pulse {len(pulse)}")
                        start_index = detector.detection_starting_locations(corr, pulse, t_res)
                        self.__print(debug_enabled, f"detection_starting_locations {len(start_index)}")
                        outputs = detector.detection_starting_locations_final(start_index)
                        self.__print(debug_enabled, f"Outputs {len(outputs)}")
                        self.__print(debug_enabled, "Detect locations")
                        bboxes = detector.detection_bounding_boxes(outputs, spectrogramSliceZscore, t_res, f_res,
                                                                   lower_freq,
                                                                   upper_freq, modelW, d0_min, d0_max)

                        for output in bboxes:
                            start = output[0] * 1000
                            end = output[1] * 1000
                            start_time = df.DateTime.values[0] + np.timedelta64(int(start), 'ms')
                            end_time = df.DateTime.values[0] + np.timedelta64(int(end), 'ms')

                            output_analysis = (
                                str(start_time), str(end_time), output[2] * 1e3, output[3] * 1e3,
                                d0, output[5], int(output[4]))
                            store.add_info_to_df(df, df_w, loc, output_analysis)
                            loc += 1

                    except Exception as e:
                        self.__print(debug_enabled, f"error {repr(e)}")
                        error_analysis = (str(df.DateTime.values[0]), str(df.DateTime.values[-1]), -2, -2, -2, -2, -2)
                        store.add_info_to_df(df, df_w, loc, error_analysis)

                df_w.to_pickle(whistlers_file)
            else:
                f = open(os.path.join(directory_path, "no_burst_found.txt"), "a")
                f.write(f"{file_name}\n")
                f.close()

        except:
            f = open(os.path.join(directory_path, "error_open_file.txt"), "a")
            f.write(f"{file_name}\n")
            f.close()

    @staticmethod
    def __print(enabled, text):
        if enabled:
            print(text)
