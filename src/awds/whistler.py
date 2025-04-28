import numpy as np


class WhistlerModel:

    def __init__(self, t_res, f_res, low_f, high_f, fn):
        self.__f_res = f_res
        self.__t_res = t_res
        self.__f = np.linspace(low_f, high_f, 1000)
        self.__fn = fn
        self.mapW = {}
        for d in np.concatenate(( np.arange(1, 101, 1), np.arange(100, 201, 1))):
            self.whistler_sim(d)

    def whistler_trace(self, An, D0):
        """generate the whistler trace
        Params
            An: normalised equatorial electron gyrofrequency
            D0: zero dispersion
            fn: nose frequency
            f: frequency range
        return
            t: time range
            """
        t = (D0 / ((1 + An) * np.sqrt(self.__f))) * (
                ((1 + An) - (3 * An - 1) * (self.__f / self.__fn)) / (1 - An * self.__f / self.__fn))
        return np.array(t)

    def whistler_sim(self, D0, An=0.35, magnitude=1):
        """Generate a 2D representation of the whistler trace
        Params
            An: normalised equatorial electron gyrofrequency
            D0: zero dispersion
            fn: nose frequency
            f: frequency range
        return
            t: time range"""

        d_s = str(D0)
        if d_s in self.mapW.keys():
            return self.mapW[d_s]
        else:
            t = self.whistler_trace(An=An, D0=D0)
            t_trans, f_trans = (t - t.min()) / self.__t_res, (self.__f - self.__f.min()) * 1e-3 / self.__f_res
            t_trans, f_trans = t_trans.astype(int), f_trans.astype(int)
            coor = np.array([(t, f) for t, f in zip(t_trans, f_trans)])
            data = np.zeros((t_trans.max() + 1, f_trans.max() + 1))
            for x, y in coor:
                data[x, y] = magnitude
            self.mapW[d_s] = data.T
            return self.mapW[d_s]
