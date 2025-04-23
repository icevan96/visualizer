import h5py
import pandas as pd
from datetime import timedelta
import os
from itertools import chain
import numpy as np
from awds.reader import ReaderInterface, VLFInformation
import math


class Debug:
    def __init__(self, survey, burst, freq):
        self.survey = survey
        self.burst = burst
        self.freq = freq


class EFD(ReaderInterface):

    def debug(self, path: str, file_name: str) -> Debug:
        full_path = os.path.join(path, file_name)
        with h5py.File(full_path, "r") as f:
            Workmode = f['WORKMODE'][()][:, 0]
            UTC_TIME = f['UTC_TIME'][()][:, 0]
            MAG_LAT = f['MAG_LAT'][()][:, 0]
            MAG_LON = f['MAG_LON'][()][:, 0]
            GEO_LAT = f['GEO_LAT'][()][:, 0]
            GEO_LON = f['GEO_LON'][()][:, 0]
            ALT = f['ALTITUDE'][()][:, 0]
            FREQ = f['FREQ'][()][:, 0]

            A131_W = f['A131_W'][()]
            A132_W = f['A132_W'][()]
            A133_W = f['A133_W'][()]
            A131_P = f['A131_P'][()]
            A132_P = f['A132_P'][()]
            A133_P = f['A133_P'][()]

            d = [
                'DateTime',
                "OrbitNumber",
                'Frequency',
                'GEO_LAT',
                'GEO_LON',
                'ALTITUDE',
                'WORKMODE',
                'Signal',
                'X',
                'Z',
                'L',
                'MAG_LAT',
                'MAG_LON'
            ]

            OrbitNumber = file_name.split("_")[6]
            sampling_frequency = 51200
            signal_A131_W = np.array_split(np.asarray(list(chain.from_iterable(A131_W))),
                                           int(A131_W.shape[0] * A131_W.shape[1] / sampling_frequency))
            signal_A132_W = np.array_split(np.asarray(list(chain.from_iterable(A132_W))),
                                           int(A132_W.shape[0] * A132_W.shape[1] / sampling_frequency))
            signal_A133_W = np.array_split(np.asarray(list(chain.from_iterable(A133_W))),
                                           int(A133_W.shape[0] * A133_W.shape[1] / sampling_frequency))

            listw = []
            index = np.where(Workmode == 2)
            W_UTC_TIME = UTC_TIME[index]
            W_GEO_LAT = GEO_LAT[index]
            W_GEO_LON = GEO_LON[index]
            W_ALT = ALT[index]
            W_MAG_LAT = MAG_LAT[index]
            W_MAG_LON = MAG_LON[index]

            for i in range(W_UTC_TIME.size):
                listw.append((W_UTC_TIME[i], OrbitNumber, 50000, W_GEO_LAT[i], W_GEO_LON[i], W_ALT[i], 2,
                              signal_A131_W[i],
                              signal_A132_W[i],
                              signal_A133_W[i],
                              1 / math.pow(math.cos(W_MAG_LAT[i]), 2), W_MAG_LAT[i], W_MAG_LON[i]))

            df_w = pd.DataFrame(listw, columns=d)
            # convert UTC_TIME in pandas datatime format: YYYYMMDDHHMMSSmsmsms
            df_w['DateTime'] = pd.to_datetime(df_w['DateTime'],
                                              format='%Y%m%d%H%M%S%f',
                                              errors='coerce')

            listp = []
            for i in range(UTC_TIME.size):
                listp.append((UTC_TIME[i], OrbitNumber, 50000, GEO_LAT[i], GEO_LON[i], ALT[i], 1,
                              A131_P[i],
                              A132_P[i],
                              A133_P[i],
                              1 / math.pow(math.cos(MAG_LAT[i]), 2), MAG_LAT[i], MAG_LON[i]))

            df_p = pd.DataFrame(listp, columns=d)
            # convert UTC_TIME in pandas datatime format: YYYYMMDDHHMMSSmsmsms
            df_p['DateTime'] = pd.to_datetime(df_p['DateTime'],
                                              format='%Y%m%d%H%M%S%f',
                                              errors='coerce')

            return Debug(df_p, df_w, FREQ)

    def read(self, path: str, file_name: str, split_seconds: int = 10) -> VLFInformation:
        try:
            full_path = os.path.join(path, file_name)
            with h5py.File(full_path, "r") as f:
                if 'A131_W' not in list(f.keys()):
                    return VLFInformation(file_name, "h5", 0, 50000, vlf_signal=None)

                # Read the whole file at onc
                Workmode = f['WORKMODE'][()][:, 0]
                UTC_TIME = f['UTC_TIME'][()][:, 0]
                MAG_LAT = f['MAG_LAT'][()][:, 0]
                MAG_LON = f['MAG_LON'][()][:, 0]
                GEO_LAT = f['GEO_LAT'][()][:, 0]
                GEO_LON = f['GEO_LON'][()][:, 0]
                ALT = f['ALTITUDE'][()][:, 0]
                FREQ = f['FREQ'][()][:, 0]

                A131_W = f['A131_W'][()]
                A132_W = f['A132_W'][()]
                A133_W = f['A133_W'][()]
                A131_P = f['A131_P'][()]
                A132_P = f['A132_P'][()]
                A133_P = f['A133_P'][()]

                d = [
                    'DateTime',
                    "OrbitNumber",
                    'Frequency',
                    'GEO_LAT',
                    'GEO_LON',
                    'ALTITUDE',
                    'WORKMODE',
                    'Signal',
                    'X',
                    'Z',
                    'L',
                    'MAG_LAT',
                    'MAG_LON'
                ]
                try:
                    OrbitNumber = file_name.split("_")[6]
                    sampling_frequency = 51200
                    listp = []
                    signal_A131_W = np.array_split(np.asarray(list(chain.from_iterable(A131_W))),
                                                   int(A131_W.shape[0] * A131_W.shape[1] / sampling_frequency))
                    signal_A132_W = np.array_split(np.asarray(list(chain.from_iterable(A132_W))),
                                                   int(A132_W.shape[0] * A132_W.shape[1] / sampling_frequency))
                    signal_A133_W = np.array_split(np.asarray(list(chain.from_iterable(A133_W))),
                                                   int(A133_W.shape[0] * A133_W.shape[1] / sampling_frequency))
                    index = -1
                    index_s = -1
                except Exception as e:
                    print(f"{file_name} Error on signal creations {repr(e)}")
                    return VLFInformation(file_name, "h5", 0, 50000, vlf_signal=None)

                try:
                    count_workmode = sum(1 for mode in Workmode if mode != 2)
                    final_range = count_workmode+ len(signal_A131_W) -1
                    for i in range(final_range):
                        if Workmode[i] == 2:
                            index += 1

                        else:
                            index_s += 1

                        listp.append((UTC_TIME[i], OrbitNumber, 50000, GEO_LAT[i], GEO_LON[i], ALT[i], Workmode[i],
                                      signal_A131_W[index] if Workmode[i] == 2 else A131_P[index_s],
                                      signal_A132_W[index] if Workmode[i] == 2 else A132_P[index_s],
                                      signal_A133_W[index] if Workmode[i] == 2 else A133_P[index_s],
                                      1 / math.pow(math.cos(math.radians(MAG_LAT[i])), 2), MAG_LAT[i], MAG_LON[i]))
                except Exception as e:
                    print(f"{file_name} error: EFD.read = list index out of range {repr(e)}")
                    return VLFInformation(file_name, "h5", 0, 50000, vlf_signal=None)
                try:
                    if len(listp) < 1:
                        print(f"listp < 1 - {file_name}")
                        return VLFInformation(file_name, "h5", 0, 50000, vlf_signal=None)

                    df = pd.DataFrame(listp, columns=d)
                    # convert UTC_TIME in pandas datatime format: YYYYMMDDHHMMSSmsmsms
                    df['DateTime'] = pd.to_datetime(df['DateTime'],
                                                    format='%Y%m%d%H%M%S%f',
                                                    errors='coerce')
                    df_tmp = df.copy()
                    # select only burst mode
                    df = df[df.WORKMODE == 2].reset_index()
                    print(df.size)

                    vlf = VLFInformation(file_name, "h5", df["L"].values[0], 50000, vlf_signal=df_tmp,
                                         split=self.split_file(df, split_seconds), other=FREQ)
                except:
                    print(f"{file_name} error on dataframe creations")
                    vlf = VLFInformation(file_name, "h5", 0, 50000, vlf_signal=None)
        except:
            print(f"{file_name} error on opening")
            vlf = VLFInformation(file_name, "h5", 0, 50000, vlf_signal=None)

        return vlf

    def split_file(self, df, seconds, overlap=0):
        start_time = df.DateTime.loc[0] - timedelta(seconds=overlap)
        split = []
        while True:
            end_time = start_time + timedelta(seconds=seconds + overlap)
            mask = (df.DateTime > start_time) & (df.DateTime <= end_time)
            value = df.loc[mask]
            if value.size > 0:
                split.append(value)

            if start_time > df.DateTime.values[-1]:
                break
            else:
                start_time = end_time - timedelta(seconds=overlap)

        return split
