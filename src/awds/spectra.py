import numpy as np
import scipy.stats as stats
from scipy import signal


class Spectra:
    # spectrogram default parameters
    __NPERSEG = 256  # Length of each segment. Defaults to None, but if window is str or tuple, is set to 256,
    # and if window is array_like, is set to the length of the window.
    __NOVERLAP = 0  # Number of points to overlap between segments. If None, noverlap = nperseg // 8. Defaults to None.
    __NFFT = 256  # Length of the FFT used, if a zero padded FFT is desired. If None, the FFT length is nperseg. Defaults to None.
    __DETREND = 'constant'  # Specifies how to detrend each segment. If detrend is a string, it is passed as the type argument to
    # the detrend function. If it is a function, it takes a segment and returns a detrended segment.
    # If detrend is False, no detrending is done. Defaults to ‘constant’ [‘linear’, ‘constant’]
    __SCALING = 'spectrum'  # Selects between computing the power spectral density (‘density’) where Sxx has units of V**2/Hz and
    # computing the power spectrum (‘spectrum’) where Sxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density’.
    __MODE = 'magnitude'  # Defines what kind of return values are expected. Options are [‘psd’, ‘complex’, ‘magnitude’, ‘angle’, ‘phase’].

    def spectrogram(self, vlf_signal, sampling_frequency, log10=True, kHz=True):
        freqs, time, Sxx = signal.spectrogram(vlf_signal, fs=sampling_frequency,
                                              nperseg=self.__NPERSEG,
                                              noverlap=self.__NOVERLAP, nfft=self.__NFFT, detrend=self.__DETREND,
                                              scaling=self.__SCALING,
                                              mode=self.__MODE)

        if kHz:
            freqs /= 1e3
        if log10:
            Sxx = np.log10(Sxx)

        return freqs, time, Sxx

    def apply_zscore(self, spectra):
        for ax in [0, 1]:
            spectra = stats.zscore(spectra, axis=ax)
        return spectra

    def apply_slice(self, lower_freq, upper_freq, freqs, spec):
        low = int(lower_freq / self.get_freq_res(freqs))
        upper = int(upper_freq / self.get_freq_res(freqs))
        spec_slice = spec[low:upper, :]
        freq_slice = freqs[low:upper]
        return spec_slice, freq_slice

    def get_time_res(self, time):
        return time[-1] / len(time)

    def get_freq_res(self, freq):
        return freq[-1] / len(freq)

    def get_correlation(self, spectra, kernel, mode='valid', method='fft'):
        if kernel.shape[0] > spectra.shape[0]:
            kernel = kernel[:spectra.shape[0], :]

        return signal.correlate(10 ** np.copy(spectra), kernel, mode=mode, method=method)[0]

    def get_time_freq_ratio(self, time, freq, dec=0, integer=True):
        ratio = np.round(time.shape[0] / freq.shape[0], dec)
        if integer:
            ratio = int(ratio)
        return ratio + 1 if ratio % 2 == 0 else ratio
