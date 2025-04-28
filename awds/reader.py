from numpy import ndarray


class VLFInformation:
    def __init__(self, file_name: str, file_extension: str, L: float, sampling_frequency: float, vlf_signal,
                 split=None, other=None):
        self.file_name = file_name
        self.file_extension = file_extension
        self.L = L
        self.sampling_frequency = sampling_frequency
        self.vlf_signal = vlf_signal
        self.split = split
        self.other = other


class ReaderInterface:
    def read(self, path: str, file_name: str, split_seconds: int = 10) -> VLFInformation:
        """Load in the file for extracting text."""
        pass
