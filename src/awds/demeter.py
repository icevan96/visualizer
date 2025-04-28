import os
import struct
from datetime import datetime, timedelta
from io import open

import numpy as np
import pandas as pd
from tqdm import tqdm

from awds.reader import ReaderInterface, VLFInformation


def chunks(array, size):
    size = max(1, size)
    return (array[i:i + size] for i in range(0, len(array), size))


class Demeter1131(ReaderInterface):
    """
     file description
     https://cdpp-archive.cnes.fr/project/data/documents/PLAS-DED-DMT_ICE-00532-CN/dmt_n1_1131_1.html

    """
    ROW_SIZE = 33063
    DATE_START_POSITION = 8
    DATE_END_POSITION = 22
    ORBIT_PARAM_START_POSITION = 38
    ORBIT_PARAM_END_POSITION = 50
    GEOMAG_PARAM_START_POSITION = 54
    GEOMAG_PARAM_END_POSITION = 114
    DATA_TYPE_START_POSITION = 204
    DATA_TYPE_END_POSITION = 225
    HEADER_START_POSITION = 257
    HEADER_END_POSITION = 295
    SIGNAL_START_POSITION = 295
    SIGNAL_END_POSITION = 33063
    SIGNAL = "Signal"
    DATE_TIME = "DateTime"
    BASIC_COLUMNS = ["OrbitNumber", "SubOrbitType", "GeocLat", "GeocLong", "Altitude", "GeomagLat", "GeomagLong",
                     "MLT", "InvLat", "L", "ConjsatGeocLat", "ConjsatGeocLong", "Nconj110GeocLat",
                     "Nconj110GeocLong", "Sconj110GeocLat", "Sconj110GeocLong", "b_field_x", "b_field_y", "b_field_z",
                     "GyroFreq", "DataType", "CoordSystem",
                     "DataUnit", "Frequency", "SampleNumber",
                     "TotalDuration", "Nam1c"]

    def read_df(self, path: str, file_name: str):
        full_path = path + file_name
        with open(full_path, "rb") as binary_file:
            # Read the whole file at once
            _df = self.__read_lines(binary_file.read())

        return _df

    # np.asarray(list(chain.from_iterable(_df.Signal.values)))
    def read(self, path: str, file_name: str, split_seconds: int = 10) -> VLFInformation:
        full_path = os.path.join(path, file_name)
        with open(full_path, "rb") as binary_file:
            # Read the whole file at once
            _df = self.__read_lines(binary_file.read())

        return VLFInformation(file_name, "dat", _df.L.values.mean(), _df.Frequency.loc[0], vlf_signal=None,
                              split=self.split_file(_df, split_seconds))

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

    def __read_lines(self, data):
        rows = int(len(data) / self.ROW_SIZE)
        fullList = self.BASIC_COLUMNS.copy()
        fullList.append(self.SIGNAL)
        fullList.insert(0, self.DATE_TIME)
        listp = []
        n = 0
        # with tqdm(total=rows) as pbar:
        for line in chunks(data, self.ROW_SIZE):
            date = struct.unpack(">7h", line[self.DATE_START_POSITION:self.DATE_END_POSITION])
            orbit_n = struct.unpack(">2h", line[self.DATE_END_POSITION:self.DATE_END_POSITION + 4])
            orbit_param = struct.unpack(">3f", line[self.ORBIT_PARAM_START_POSITION:self.ORBIT_PARAM_END_POSITION])
            geomag_param = struct.unpack(">15f",
                                         line[self.GEOMAG_PARAM_START_POSITION:self.GEOMAG_PARAM_END_POSITION])
            data_type = struct.unpack("@21s", line[self.DATA_TYPE_START_POSITION:self.DATA_TYPE_END_POSITION])
            header_info = struct.unpack(">9s16sfhf3s", line[self.HEADER_START_POSITION:self.HEADER_END_POSITION])
            dt = np.dtype(np.float32)
            dt = dt.newbyteorder('>')
            array = np.frombuffer(line[self.SIGNAL_START_POSITION:self.SIGNAL_END_POSITION], dtype=dt)

            dt_string = f"{date[0]} {date[1]} {date[2]} {date[3]} {date[4]} {date[5]} {date[6]}"
            dt_object = datetime.strptime(dt_string, "%Y %m %d %H %M %S %f")

            t_f = (dt_object,) + orbit_n + orbit_param + geomag_param + data_type + header_info + (array,)
            listp.append(t_f)
            n += 1
            # pbar.update(1)

        return pd.DataFrame(listp, columns=fullList)


class Demeter1132(ReaderInterface):
    """
     file description
     https://cdpp-archive.cnes.fr/project/data/documents/PLAS-DED-DMT_ICE-00533-CN/dmt_n1_1132_1.html

    """
    ROW_SIZE = 33063
    DATE_START_POSITION = 8
    DATE_END_POSITION = 22
    ORBIT_PARAM_START_POSITION = 38
    ORBIT_PARAM_END_POSITION = 50
    GEOMAG_PARAM_START_POSITION = 54
    GEOMAG_PARAM_END_POSITION = 114
    DATA_TYPE_START_POSITION = 204
    DATA_TYPE_END_POSITION = 225
    HEADER_START_POSITION = 257
    HEADER_END_POSITION = 295
    SIGNAL_START_POSITION = 295
    SIGNAL_END_POSITION = 33063
    SIGNAL = "Signal"
    DATE_TIME = "DateTime"
    BASIC_COLUMNS = ["NB", "NBF", "TOTAL_DUR", "FREQ_RES"]

    def read_df(self, path: str, file_name: str):
        full_path = path + file_name
        with open(full_path, "rb") as binary_file:
            # Read the whole file at once
            _df = self.__read_lines(binary_file.read())

        return _df

    # np.asarray(list(chain.from_iterable(_df.Signal.values)))
    def read(self, path: str, file_name: str, split_seconds: int = 10) -> VLFInformation:
        full_path = path + file_name
        with open(full_path, "rb") as binary_file:
            # Read the whole file at once
            _df = self.__read_lines(binary_file.read())

        return VLFInformation(file_name, "dat", _df.L.values.mean(), _df.Frequency.loc[0], vlf_signal=None,
                              split=self.split_file(_df, split_seconds))

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

    def __read_lines(self, data):
        number = struct.unpack(">Bh2f", data[285:296])
        start = 204 + 114
        row_size = (number[0] * number[1] * 4) + start
        rows = int(len(data) / row_size)
        fullList = self.BASIC_COLUMNS.copy()
        fullList.append(self.SIGNAL)
        fullList.insert(0, self.DATE_TIME)
        listp = []
        n = 0

        # with tqdm(total=rows) as pbar:
        for line in chunks(data, row_size):
            date = struct.unpack(">7h", line[self.DATE_START_POSITION:self.DATE_END_POSITION])

            header = struct.unpack(">Bh2f", line[285:296])

            dt = np.dtype(np.float32)
            dt = dt.newbyteorder('>')
            array = np.frombuffer(line[start:], dtype=dt)

            dt_string = f"{date[0]} {date[1]} {date[2]} {date[3]} {date[4]} {date[5]} {date[6]}"
            dt_object = datetime.strptime(dt_string, "%Y %m %d %H %M %S %f")

            t_f = (dt_object,) + header + (array,)
            listp.append(t_f)
            n += 1
            # pbar.update(1)

        return pd.DataFrame(listp, columns=fullList)


class Demeter1138:
    ROW_SIZE = 5706
    DATE_START_POSITION = 8
    DATE_END_POSITION = 22
    ORBIT_PARAM_START_POSITION = 38
    ORBIT_PARAM_END_POSITION = 50
    GEOMAG_PARAM_START_POSITION = 54
    GEOMAG_PARAM_END_POSITION = 114
    DATA_TYPE_START_POSITION = 204
    DATA_TYPE_END_POSITION = 225
    HEADER_START_POSITION = 257
    HEADER_END_POSITION = 288
    CLASS_DESCR_START_POSITION = 288
    CLASS_DESCR_END_POSITION = 458
    SPECTR_INTENSITY_START_POSITION = 458
    SPECTR_INTENSITY_END_POSITION = 3146

    DATE_TIME = "DateTime"
    BASIC_COLUMNS = ["OrbitNumber", "SubOrbitType", "GeocLat", "GeocLong", "Altitude", "GeomagLat", "GeomagLong",
                     "MLT", "InvLat", "L", "ConjsatGeocLat", "ConjsatGeocLong", "Nconj110GeocLat",
                     "Nconj110GeocLong", "Sconj110GeocLat", "Sconj110GeocLong", "b_field_x", "b_field_y", "b_field_z",
                     "GyroFreq", "DataType",
                     "DATA_SUB_TYPE", "STUDY_TITLE", "NAME_C", "T_RES", "NB_CL", "NB_S_OR_P", "NB_C",
                     'UNIT_NAME', 'MIN_TABLE', 'MAX_TABLE',
                     'SP_VALID_TABLE', 'SP_INTENS_TABLE', 'SP_UNCERT_TABLE']

    def read(self, path, file_name):
        full_path = path + file_name
        with open(full_path, "rb") as binary_file:
            # Read the whole file at once
            _df = self.__read_lines(binary_file.read())

        return _df

    def __read_lines(self, data):
        rows = int(len(data) / self.ROW_SIZE)
        fullList = self.BASIC_COLUMNS.copy()
        fullList.insert(0, self.DATE_TIME)
        listp = []
        n = 0
        # with tqdm(total=rows) as pbar:
        for line in chunks(data, self.ROW_SIZE):
            date = struct.unpack(">7h", line[self.DATE_START_POSITION:self.DATE_END_POSITION])
            orbit_n = struct.unpack(">2h", line[self.DATE_END_POSITION:self.DATE_END_POSITION + 4])
            orbit_param = struct.unpack(">3f", line[self.ORBIT_PARAM_START_POSITION:self.ORBIT_PARAM_END_POSITION])
            geomag_param = struct.unpack(">15f",
                                         line[self.GEOMAG_PARAM_START_POSITION:self.GEOMAG_PARAM_END_POSITION])
            data_type = struct.unpack("@21s", line[self.DATA_TYPE_START_POSITION:self.DATA_TYPE_END_POSITION])
            header_info = struct.unpack(">B20s3sf3B", line[self.HEADER_START_POSITION:self.HEADER_END_POSITION])
            unit_name = struct.unpack("@10s",
                                      line[self.CLASS_DESCR_START_POSITION:self.CLASS_DESCR_START_POSITION + 10])
            class_descr_min_start = self.CLASS_DESCR_START_POSITION + 10
            class_descr_min_end = class_descr_min_start + 80
            dt = np.dtype(np.float32)
            dt = dt.newbyteorder('>')
            arrayMin = np.frombuffer(line[class_descr_min_start:class_descr_min_end], dtype=dt)
            arrayMax = np.frombuffer(line[class_descr_min_end:self.CLASS_DESCR_END_POSITION], dtype=dt)

            dt = np.dtype(np.uint8)
            dt = dt.newbyteorder('>')
            sp_valid_table_end = self.SPECTR_INTENSITY_START_POSITION + 128
            sp_valid_table = np.frombuffer(line[self.SPECTR_INTENSITY_START_POSITION:sp_valid_table_end], dtype=dt)
            sp_intens_table = np.frombuffer(line[sp_valid_table_end:self.SPECTR_INTENSITY_END_POSITION], dtype=dt)
            sp_uncert_table = np.frombuffer(line[self.SPECTR_INTENSITY_END_POSITION:self.ROW_SIZE], dtype=dt)

            dt_string = f"{date[0]} {date[1]} {date[2]} {date[3]} {date[4]} {date[5]} {date[6]}"
            dt_object = datetime.strptime(dt_string, "%Y %m %d %H %M %S %f")

            t_f = (dt_object,) + orbit_n + orbit_param + geomag_param + data_type + header_info + unit_name + (
                arrayMin, arrayMax, sp_valid_table, sp_intens_table, sp_uncert_table)
            listp.append(t_f)

            n += 1

        return pd.DataFrame(listp, columns=fullList)

    def make_df_for_analysis(self, df):
        columns = ["L", "OrbitNumber", "SubOrbitType", "GeocLat", "GeocLong", "Altitude", "GeomagLat",
                   "GeomagLong"]

        classed_c = ['DateTime', "L", "OrbitNumber", "SubOrbitType", "GeocLat", "GeocLong", "Altitude", "GeomagLat",
                     "GeomagLong",
                     '(0.0-2.5)',
                     '(2.5-3.2)',
                     '(3.2-4.0)',
                     '(4.0-5.0)',
                     '(5.0-6.3)',
                     '(6.3-7.9)',
                     '(7.9-10.0)',
                     '(10.0-12.6)',
                     '(12.6-15.9)',
                     '(15.9-20.0)',
                     '(20.0-25.2)',
                     '(25.2-31.7)',
                     '(31.7-40.0)',
                     '(40.0-50.4)',
                     '(50.4-63.5)',
                     '(63.5-80.0)',
                     '(80.0-101.0)',
                     '(101.0-127.0)',
                     '(127.0-202.0)',
                     '(0.0-0.0)']

        listp = []
        for i in tqdm(range(df.shape[0])):
            df_loc = df.loc[i]
            v = df_loc['SP_INTENS_TABLE'].reshape(128, 20)

            for index in range(v.shape[0]):
                res = df_loc['T_RES'] * index * 1000
                array_v = v[index]
                count = np.count_nonzero(array_v)
                if count > 0:
                    values = tuple(v[index])
                    time = df_loc.DateTime + np.timedelta64(int(res), 'ms')
                    listp.append((time,) + tuple(df_loc[columns].values) + values)

        return pd.DataFrame(listp, columns=classed_c)
