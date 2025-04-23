import pandas as pd


class StoreInterface:
    def create_df(self) -> pd.DataFrame:
        """Load in the file for extracting text."""
        pass

    def store_df(self, df, path):
        """Load in the file for extracting text."""
        pass

    def add_info_to_df(self, main_df, df, loc, info):
        """Load in the file for extracting text."""
        pass


class StoreEFD(StoreInterface):
    def create_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["OrbitNumber", 'Frequency', 'GEO_LAT', 'GEO_LON', 'ALTITUDE', 'L', "MAG_LAT", "MAG_LON",
                     "Start_Time", "End_Time", "Start_Freq", "End_Freq", "D0_Driver", 'Driver_corr', "D0"])

    def store_df(self, df, path):
        df.to_pickle(path)

    def add_info_to_df(self, main_df, df, loc, info):
        df.loc[loc] = self.information_to_store(main_df) + info

    def information_to_store(self, df):
        OrbitNumber = df["OrbitNumber"].values[0]
        GeocLat = round(df["GEO_LAT"].values[0], 3)
        GeocLong = round(df["GEO_LON"].values[0], 3)
        Altitude = round(df["ALTITUDE"].values[0], 3)
        Frequency = df["Frequency"].values[0]
        l = round(df["L"].values[0], 3)
        MatLat = round(df["MAG_LAT"].values[0], 3)
        MatLong = round(df["MAG_LON"].values[0], 3)

        return (OrbitNumber, Frequency, GeocLat, GeocLong, Altitude, l, MatLat, MatLong)


class StoreDemeter(StoreInterface):
    def create_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["OrbitNumber", "SubOrbitType", "GeocLat",
                     "GeocLong", "Altitude", "GeomagLat", "GeomagLong",
                     "MLT", "InvLat", "L", "ConjsatGeocLat", "ConjsatGeocLong", "Nconj110GeocLat",
                     "Nconj110GeocLong", "Sconj110GeocLat", "Sconj110GeocLong", "b_field_x", "b_field_y", "b_field_z",
                     "GyroFreq", "DataUnit", "Frequency", "SampleNumber",
                     "TotalDuration", "Nam1c", "Start_Time", "End_Time", "Start_Freq", "End_Freq", "D0_Driver",
                     'Driver_corr', "D0"])

    def store_df(self, df, path):
        df.to_pickle(path)

    def add_info_to_df(self, main_df, df, loc, info):
        df.loc[loc] = self.information_to_store(df) + info

    def information_to_store(self, df):
        OrbitNumber = df["OrbitNumber"].values[0]
        SubOrbitType = df["SubOrbitType"].values[0]
        GeocLat = round(df["GeocLat"].values[0], 3)
        GeocLong = round(df["GeocLong"].values[0], 3)
        Altitude = round(df["Altitude"].values[0], 3)
        GeomagLat = round(df["GeomagLat"].values[0], 3)
        GeomagLong = round(df["GeomagLong"].values[0], 3)
        MLT = round(df["MLT"].values[0], 3)
        InvLat = round(df["InvLat"].values[0], 3)
        ConjsatGeocLat = round(df["ConjsatGeocLat"].values[0], 3)
        ConjsatGeocLong = round(df["ConjsatGeocLong"].values[0], 3)
        Nconj110GeocLat = round(df["Nconj110GeocLat"].values[0], 3)
        Nconj110GeocLong = round(df["Nconj110GeocLong"].values[0], 3)
        Sconj110GeocLat = round(df["Sconj110GeocLat"].values[0], 3)
        Sconj110GeocLong = round(df["Sconj110GeocLong"].values[0], 3)
        b_field_x = round(df["b_field_x"].values[0], 3)
        b_field_y = round(df["b_field_y"].values[0], 3)
        b_field_z = round(df["b_field_z"].values[0], 3)
        l = round(df["L"].values[0], 3)
        GyroFreq = round(df["GyroFreq"].values[0], 3)
        DataUnit = df["DataUnit"].values[0]
        Frequency = df["Frequency"].values[0]
        SampleNumber = df["SampleNumber"].values[0]
        TotalDuration = df["TotalDuration"].values[0]
        Nam1c = df["Nam1c"].values[0]

        return (
            OrbitNumber, SubOrbitType, GeocLat, GeocLong, Altitude, GeomagLat, GeomagLong, MLT, InvLat, l,
            ConjsatGeocLat, ConjsatGeocLong, Nconj110GeocLat, Nconj110GeocLong, Sconj110GeocLat, Sconj110GeocLong,
            b_field_x, b_field_y, b_field_z, GyroFreq, DataUnit, Frequency, SampleNumber, TotalDuration, Nam1c)
