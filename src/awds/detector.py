import numpy as np
from scipy import signal

from awds.adaptive_threshold import AdaptiveThreshold
from awds.whistler import WhistlerModel


class Detector:
    mapKernel = {}

    @staticmethod
    def detection_starting_locations(corr, pulses, time_res):
        """Location of the whistler after detection
        Params
            ...
            cfar: type of cfar techniques
            cafar_params: parameters of the cfar techniques
        Return
            indices: indices of detected whistler, indices as starting point and not 5kHz point
        """
        spikes = AdaptiveThreshold.diff(pulses, 2)
        highs, lows = np.argwhere(spikes == 1), np.argwhere(spikes == -1)
        assert len(highs) == len(lows)
        ix = []
        for h, l in zip(highs, lows):
            h, l = h[0], l[0]
            i = h + np.argmax(corr[h:l])
            r = 10 * np.log10(corr[i] ** 2)
            ix.append(['%.3f' % (i * time_res), '%.3f' % r])
        ix = np.array(ix)
        return ix

    @staticmethod
    def detection_starting_locations_final(starts, threshold=0, time_error=1):
        """Location of the whistler after detection
        Params
            ...
            cfar: type of cfar techniques
            cafar_params: parameters of the cfar techniques
            threshold:
            time_error: number of decimal places for time onversion
        Return
            final: starting time and matching correlatiob
        """
        thresholded = list()
        # round as per time_err decimal point
        for s in starts:
            if float(s[1]) >= threshold:
                thresholded.append([round(float(s[0]), time_error), float(s[1])])
        thresholded = np.array(thresholded)
        uniques = np.sort(np.array(list(set(thresholded[:, 0]))))
        # get maximum corr per time_error point
        final = np.array([[u, thresholded[np.argwhere(thresholded[:, 0] == u).ravel(), 1].max()] for u in uniques])
        return final

    @staticmethod
    def detection_bounding_boxes(output, spectra, time_res, freq_res, lower_freq, upper_freq,
                                 whistler_model: WhistlerModel, d0_min, d0_max, time_error=1, kernel_even=False):
        """Location of the whistler after detection
        Params
            ...
            cfar: type of cfar techniques
            cafar_params: parameters of the cfar techniques
            threshold:
            time_error: number of decimal places for time onversion
        Return
            bbox: bounding box [x1,x2,y1,y2,c] in time and frequency with c, the result of the correlation
        """
        bboxes = []

        for o in output:
            start = o[0]
            bbox = np.array([int(start / time_res), int((start + 1) / time_res),
                             int(lower_freq / freq_res), int(upper_freq / freq_res)])

            data = spectra[:, bbox[0]:bbox[1]]

            D0 = np.arange(1, 200 + 1, 1)
            peaks = []
            for d in D0:

                kernel = whistler_model.whistler_sim(An=0.35, D0=d, magnitude=1)

                if kernel.shape[0] > data.shape[0]:
                    kernel = kernel[:data.shape[0], :]

                if kernel_even:
                    if kernel.shape[1] > data.shape[1]:
                        # print(F"Kernel {d} > data: kernel= {kernel.shape} data= {data.shape} ")
                        kernel = kernel[:, :data.shape[1]]
                    else:
                        kernel = np.concatenate(
                            (kernel, np.zeros((kernel.shape[0], (data.shape[1] - kernel.shape[1])))),
                            axis=1)

                corr = signal.correlate(data, kernel, mode='valid')[0]
                # if len(corr) > 1:
                #     print(F"Kernel {d} corr > 1: kernel= {kernel.shape} data= {data.shape} corre={len(corr)}")

                peaks.append(corr.max())

            d_final = D0[np.argmax(peaks)]
            kernel = whistler_model.whistler_sim(An=0.35, D0=d_final, magnitude=1)
            duration = kernel.shape[1] * time_res

            tmp = (round(start, time_error),
                   round(start + duration, time_error),
                   lower_freq,
                   upper_freq,
                   d_final,
                   o[1],
                   peaks)

            bboxes.append(tmp)

        return bboxes

    def detection_bounding_boxes_2(self, output, spectra, time_res, freq_res, lower_freq, upper_freq,
                                   whistlerModel: WhistlerModel, d0_min, d0_max, time_error=1):
        """Location of the whistler after detection
        Params
            ...
            cfar: type of cfar techniques
            cafar_params: parameters of the cfar techniques
            threshold:
            time_error: number of decimal places for time conversion
        Return
            bbox: bounding box [x1,x2,y1,y2,c] in time and frequency with c, the result of the correlation
        """
        bboxes = []
        interval = self.generate_interval(45)
        for o in output:
            start = o[0]
            bbox = np.array([int(start / time_res), int((start + 1) / time_res),
                             int(lower_freq / freq_res), int(upper_freq / freq_res)])
            data = spectra[bbox[2]:bbox[3], bbox[0]:bbox[1]]

            D0 = self.get_D0_interval(interval, data, whistlerModel)
            peaks = []
            for d in D0:
                d_s = str(d)
                if d_s in self.mapKernel.keys():
                    kernel = self.mapKernel[d_s]
                else:
                    kernel = whistlerModel.whistler_sim(An=0.35, D0=d, magnitude=1)
                    self.mapKernel[d_s] = kernel

                corr = signal.correlate(data, kernel[:data.shape[0], :], mode='valid')[0]
                peaks.append(corr.max())

            duration = self.mapKernel[str(D0[np.argmax(peaks)])].shape[1] * time_res
            tmp = (round(start, time_error), round(start + duration, time_error), lower_freq, upper_freq,
                   D0[np.argmax(peaks)], o[1], peaks, D0)

            bboxes.append(tmp)

        return bboxes, bboxes

    def get_D0_interval(self, interval, data, whistlerModel: WhistlerModel):
        peaks = []
        for i in interval:
            d = i[0]
            d_s = str(d)
            if d_s in self.mapKernel.keys():
                kernel = self.mapKernel[d_s]
            else:
                kernel = whistlerModel.whistler_sim(An=0.35, D0=d, magnitude=1)
                self.mapKernel[d_s] = kernel

            corr = signal.correlate(data, kernel[:data.shape[0], :], mode='valid')[0]
            peaks.append(corr.max())

        i = interval[np.argmax(peaks)]

        return np.arange(i[1], i[2] + 1, 1)

    @staticmethod
    def generate_interval(m):
        d0_min, d0_max = 1, 200
        a = np.arange(d0_min, d0_max + 1, 1)
        step = int(a.size / m)
        start = 0
        list_interval = []
        for i in a[::step]:
            sub_array = a[start:(i - 1)]
            if sub_array.size > 0:
                list_interval.append((sub_array[int(sub_array.size / 2)], start, (i - 1)))
            start = i

        if list_interval[-1][2] < d0_max:
            sub_array = a[start:d0_max]
            if sub_array.size > 0:
                list_interval.append((sub_array[int(sub_array.size / 2)], start, d0_max))

        return list_interval
