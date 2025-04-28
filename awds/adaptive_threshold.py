from math import floor, ceil

import numpy as np


class AdaptiveThreshold:
    # N: number of noise cells on each side of the CUT
    #                             G: number of gaurd cells on each side of the CUT
    #                             T1: number of smallest cells to discard
    #                             T2: number of highest cells to discard
    #                             pfa: probability of false alarm
    N, G = 10, 7
    T1, T2 = floor(0.3 * N), floor(0.8 * N)
    desiredX_dB = 8
    theorical_pfa = (1 / (1 + ((10 ** (desiredX_dB / 10)) / (2 * N)))) ** (2 * N)
    ca = 2 * N * ((theorical_pfa ** (-1 / (2 * N))) - 1)
    stride = 1
    window = 2 * (N + G) + 1

    def ca_cfar(self, corr):
        """Cell Averaging Constant False Alarm Rate (CFAR)
        Params
            ...
            cfar_params: [N,G,pfa]
                            N: number of noise cells on each side of the CUT
                            G: number of gaurd cells on each side of the CUT
                            pfa: probability of false alarm
        Return
            detector: adaptive threshold
        """
        N, G, pfa = self.N, self.G, self.theorical_pfa
        threshold = lambda w: self.ca * (((w[:N] ** 2).sum() + (w[N + 2 * G + 1:] ** 2).sum()) / (2 * N))
        windows = self.rolling(corr, self.window)
        detector = np.array(list(map(threshold, windows)))
        return detector

    def os_cfar(self, corr):
        """Order Statistic Constant False Alarm Rate (OS CFAR)
        Params
            ...
            cfar_params: [N,G,k,pfa]
                            N: number of noise cells on each side of the CUT
                            G: number of gaurd cells on each side of the CUT
                            k: kth order statistic, 2N-k >= nbr of expected cell targets
                            pfa: probability of false alarm
        Return
            detector: adaptive threshold"""
        N, G, pfa = self.N, self.G, self.theorical_pfa
        k = N - 1
        threshold = lambda w: self.ca * ((np.sort(np.concatenate((w[:N], w[N + 2 * G + 1:])))[k]) ** 2)
        windows = self.rolling(corr, self.window)
        detector = np.array(list(map(threshold, windows)))
        return detector

    def tm_cfar(self, corr):
        """Trimmed Mean Constant False Alarm Rate (TM CFAR)
        Params
            ...
            cfar_params: [N,G,k,pfa]
                            N: number of noise cells on each side of the CUT
                            G: number of gaurd cells on each side of the CUT
                            T1: number of smallest cells to discard
                            T2: number of highest cells to discard
                            pfa: probability of false alarm
        Return
            detector: adaptive threshold"""
        N, G, pfa, T1, T2 = self.N, self.G, self.theorical_pfa, self.T1, self.T2
        threshold = lambda w: self.ca * (
                (np.sort(np.concatenate((w[:N], w[N + 2 * G + 1:])))[T1:2 * N - T2]) ** 2).sum() / (
                                      2 * N - (T2 + T1))
        windows = self.rolling(corr, self.window)
        detector = np.array(list(map(threshold, windows)))
        return detector

    def fusion_cfar(self, corr):
        """Apply the fusion CFAR algorithm on the pulses obtainde from the cfar techniques
        Params
            ...
            cfar_params: [N,G,k,pfa]
                            N: number of noise cells on each side of the CUT
                            G: number of gaurd cells on each side of the CUT
                            k: kth order statistic, 2N-k >= nbr of expected cell targets
                            T1: number of smallest cells to discard
                            T2: number of highest cells to discard
                            pfa: probability of false alarm
        Return
            detector: adaptive threshold"""
        ca = self.detection_pulse(corr, 'ca_cfar')
        os = self.detection_pulse(corr, 'os_cfar')
        tm = self.detection_pulse(corr, 'tm_cfar')
        fusion = np.bitwise_or(np.bitwise_and(ca, np.bitwise_or(os, tm)), np.bitwise_and(os, tm))
        return fusion

    def detection_pulse(self, corr, cfar):
        """Detection decision pulse
        Params
            ...
            cfar: type of cfar techniques
            cafar_params: parameters of the cfar techniques
        Return
            pulses: cfar detector decisions"""

        global pulses
        corr_sqrt = corr ** 2
        get_pulse = lambda detector: np.array(
            [True if sig > thres else False for sig, thres in zip(corr_sqrt, detector)])
        if cfar == 'ca_cfar':
            pulses = get_pulse(self.ca_cfar(corr))
        if cfar == 'os_cfar':
            pulses = get_pulse(self.os_cfar(corr))
        if cfar == 'tm_cfar':
            pulses = get_pulse(self.tm_cfar(corr))
        if cfar == 'fusion_cfar':
            pulses = self.fusion_cfar(corr)

        pulses[0], pulses[-1] = False, False

        return pulses

    @staticmethod
    def diff(signal, window):
        """Derivate the signal based on the dt=window
        Params
            signal: signal to be derivated
            window: time step
        Return
            first derivative of the signal"""
        windows = AdaptiveThreshold.rolling(signal, window)
        z = lambda w: (w[int(window / 2):].mean() - w[:int(window / 2)].mean())
        return np.array(list(map(z, windows)))

    @staticmethod
    def rolling(signal, window):
        """Return a rolling window of the signal
        Params
            signal: signal to be rolled
            window: window size
        Return
            roll: windows"""

        pad_size = window - 1
        padded_signal = np.concatenate((np.full(ceil(pad_size / 2), signal[0]), signal))
        padded_signal = np.concatenate((padded_signal, np.full(floor(pad_size / 2), signal[-1])))
        roll = []
        for ix in range(0, len(padded_signal) - window + 1):
            roll.append(padded_signal[ix:ix + window])

        return np.array(roll)
