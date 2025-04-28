import os
from awds.whistler import WhistlerModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class WhistlerDetectionVisualizer:
    """
    A class for visualizing whistler detections with their corresponding kernel models.

    This class handles the extraction of spectrogram sections around detections,
    generates whistler model kernels for each detection, and creates visualizations
    that overlay the kernels on the spectrograms.
    """

    def __init__(self,
                 plot_obj=None,
                 padding_factor=0.2,
                 kernel_alpha=0.7,
                 kernel_threshold=0.1,
                 output_dir="../output/figures/detections_with_kernels"):
        """
        Initialize the WhistlerDetectionVisualizer.

        Args:
            plot_obj: The SpectrogramPlot object used for calculating figure sizes
            padding_factor: Amount of padding to add around each detection (as a fraction of detection size)
            kernel_alpha: Transparency level for the kernel overlay (0-1)
            kernel_threshold: Threshold for displaying kernel values (relative to max value)
            output_dir: Directory to save the visualization images
        """

        # Store configuration
        self.plot_obj = plot_obj
        self.padding_factor = padding_factor
        self.kernel_alpha = kernel_alpha
        self.kernel_threshold = kernel_threshold
        self.output_dir = output_dir

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def find_nearest_idx(self, array, value):
        """Find the index of the value in array that is closest to the target value."""
        import numpy as np
        return np.abs(array - value).argmin()

    def generate_kernel(self, t_res, f_res, low_f, high_f, fn, d0):
        """
        Generate a whistler kernel for the given parameters.

        Args:
            t_res: Time resolution
            f_res: Frequency resolution
            low_f: Lower frequency bound
            high_f: Higher frequency bound
            fn: Nominal frequency
            d0: Dispersion parameter

        Returns:
            2D array representing the whistler kernel
        """
        model = WhistlerModel(t_res, f_res, low_f, high_f, fn)
        return model.whistler_sim(d0)

    def resize_kernel(self, kernel, target_height, target_width):
        """
        Resize the kernel to match the target dimensions.

        Args:
            kernel: The kernel array to resize
            target_height: Target height in pixels
            target_width: Target width in pixels

        Returns:
            Resized kernel array
        """
        import numpy as np
        from scipy.ndimage import zoom

        # Get kernel dimensions
        kernel_height, kernel_width = kernel.shape if len(kernel.shape) == 2 else (1, len(kernel))

        # Resize if needed
        if kernel_height != target_height or kernel_width != target_width:
            zoom_factors = (target_height / kernel_height, target_width / kernel_width)
            try:
                return zoom(kernel, zoom_factors, order=1)
            except Exception as e:
                print(f"Warning: Could not resize kernel. Using original size. Error: {e}")
                return kernel
        return kernel

    def position_kernel(self, kernel, target_array, y_start, x_start):
        """
        Position the kernel within the target array at the specified position.

        Args:
            kernel: The kernel array to position
            target_array: The array in which to position the kernel
            y_start: Starting y-index (frequency axis)
            x_start: Starting x-index (time axis)

        Returns:
            Array of same size as target_array with kernel positioned
        """

        # Create an empty mask for the kernel
        kernel_mask = np.zeros_like(target_array)

        # Calculate end positions
        y_end = min(y_start + kernel.shape[0], target_array.shape[0])
        x_end = min(x_start + kernel.shape[1], target_array.shape[1])

        # Position the kernel
        try:
            kernel_sized = kernel[:y_end - y_start, :x_end - x_start]
            kernel_mask[y_start:y_end, x_start:x_end] = kernel_sized
            return kernel_mask
        except Exception as e:
            print(f"Warning: Could not overlay kernel. Error: {e}")
            print(f"Kernel shape: {kernel.shape}, Section shape: {target_array.shape}")
            print(f"Position: y={y_start}:{y_end}, x={x_start}:{x_end}")
            return kernel_mask

    def process_detection(self,
                          spectrogram,
                          time,
                          freqs,
                          detection,
                          t_res,
                          f_res,
                          low_f,
                          high_f,
                          fn,
                          detection_index=0,
                          spectrogram_cmap='jet',
                          kernel_cmap='jet',
                          show_kernel=True):
        """
        Process a single detection and create a visualization.

        Args:
            spectrogram: 2D array of spectrogram data
            time: Array of time values (x-axis)
            freqs: Array of frequency values (y-axis)
            detection: Detection box as (x1, x2, y1, y2, D0)
            t_res: Time resolution
            f_res: Frequency resolution
            low_f: Lower frequency bound
            high_f: Higher frequency bound
            fn: Nominal frequency
            detection_index: Index of the detection for labeling
            kernel_cmap: Colormap to use for the kernel
            spectrogram_cmap: Colormap to use for the spectrogram
            show_kernel: Whether to show the kernel
        Returns:
            Path to the saved figure
        """
        # Extract detection parameters
        x1, x2, y1, y2, D0 = detection

        # Calculate padding
        time_padding = (x2 - x1) * self.padding_factor
        freq_padding = (y2 - y1) * self.padding_factor

        # Calculate padded boundaries
        x1_padded = max(x1 - time_padding, time[0])
        x2_padded = min(x2 + time_padding, time[-1])
        y1_padded = max(y1 - freq_padding, freqs[0])
        y2_padded = min(y2 + freq_padding, freqs[-1])

        # Find indices in the arrays
        t1_idx = self.find_nearest_idx(time, x1_padded)
        t2_idx = self.find_nearest_idx(time, x2_padded)
        f1_idx = self.find_nearest_idx(freqs, y1_padded)
        f2_idx = self.find_nearest_idx(freqs, y2_padded)

        # Swap frequency indices if needed (depends on how freqs is ordered)
        if f1_idx > f2_idx:
            f1_idx, f2_idx = f2_idx, f1_idx

        # Extract the section of the spectrogram
        spectrogram_section = spectrogram[f1_idx:f2_idx + 1, t1_idx:t2_idx + 1]
        time_section = time[t1_idx:t2_idx + 1]
        freq_section = freqs[f1_idx:f2_idx + 1]

        # Get the unpadded detection indices within this section
        x1_section_idx = self.find_nearest_idx(time_section, x1)
        x2_section_idx = self.find_nearest_idx(time_section, x2)
        y1_section_idx = self.find_nearest_idx(freq_section, y1)
        y2_section_idx = self.find_nearest_idx(freq_section, y2)

        # Generate the whistler kernel for this detection
        kernel = self.generate_kernel(t_res, f_res, low_f, high_f, fn, D0)

        # Resize the kernel to match the detection dimensions in the section
        detection_width = x2_section_idx - x1_section_idx + 1
        detection_height = y2_section_idx - y1_section_idx + 1
        kernel = self.resize_kernel(kernel, detection_height, detection_width)

        # Create a figure for this detection
        fig_size = self.plot_obj.calculate_figure_size(len(time_section) / len(freq_section)) if self.plot_obj else (
            10, 8)
        fig, ax = plt.subplots(figsize=fig_size)

        # Plot the spectrogram section
        pcm = ax.pcolormesh(time_section, freq_section, spectrogram_section, cmap=spectrogram_cmap)

        if show_kernel:
            # Create and position the kernel
            kernel_mask = self.position_kernel(kernel, spectrogram_section, y1_section_idx, x1_section_idx)

            # Mask kernel values below threshold
            kernel_mask_threshold = kernel_mask > self.kernel_threshold * np.max(kernel_mask)
            masked_kernel = np.ma.masked_where(~kernel_mask_threshold, kernel_mask)

            if kernel_cmap == 'transparent_red':
                transparent_cmap = LinearSegmentedColormap.from_list('transparent_red', [(1, 0, 0, 0), (1, 0, 0, 1)])
                ax.pcolormesh(time_section, freq_section, masked_kernel, cmap=transparent_cmap, alpha=self.kernel_alpha)
            elif kernel_cmap == 'transparent_blue':
                transparent_cmap = LinearSegmentedColormap.from_list('transparent_blue', [(0, 0, 1, 0), (0, 0, 1, 1)])
                ax.pcolormesh(time_section, freq_section, masked_kernel, cmap=transparent_cmap, alpha=self.kernel_alpha)
            elif kernel_cmap.startswith('transparent_'):
                color_name = kernel_cmap.split('_')[1]
                color_map = {
                    'green': (0, 1, 0),
                    'yellow': (1, 1, 0),
                    'purple': (0.5, 0, 0.5),
                    'cyan': (0, 1, 1),
                    'magenta': (1, 0, 1),
                    'orange': (1, 0.65, 0),
                }
                # Default to blue if color not found
                rgb = color_map.get(color_name, (0, 0, 1))
                transparent_cmap = LinearSegmentedColormap.from_list(
                    kernel_cmap, [(rgb[0], rgb[1], rgb[2], 0), (rgb[0], rgb[1], rgb[2], 1)]
                )
                ax.pcolormesh(time_section, freq_section, masked_kernel, cmap=transparent_cmap, alpha=self.kernel_alpha)
            else:
                # Use a standard matplotlib colormap
                ax.pcolormesh(time_section, freq_section, masked_kernel, cmap=kernel_cmap, alpha=self.kernel_alpha)

        # Add a rectangle around the actual detection
        detection_rect = plt.Rectangle(
            (time_section[x1_section_idx], freq_section[y1_section_idx]),
            time_section[x2_section_idx] - time_section[x1_section_idx],
            freq_section[y2_section_idx] - freq_section[y1_section_idx],
            edgecolor='white', facecolor='none', linewidth=1
        )
        ax.add_patch(detection_rect)

        plt.colorbar(pcm, ax=ax)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')

        if show_kernel:
            ax.set_title(f"Detection {detection_index + 1} (D0: {D0:.2f}) with Whistler Kernel")
        else:
            ax.set_title(f"Detection {detection_index + 1} (D0: {D0:.2f})")


        if show_kernel:
            file_name = f"{self.output_dir}/detection_{detection_index + 1}_D0_{D0:.2f}_with_kernel.png"
        else:
            file_name = f"{self.output_dir}/detection_{detection_index + 1}_D0_{D0:.2f}.png"
        plt.tight_layout()
        plt.savefig(file_name, dpi=300)
        plt.close(fig)

        print(f"Saved detection {detection_index + 1} with kernel to {file_name}")
        return file_name

    def process_all_detections(self,
                               spectrogram,
                               time,
                               freqs,
                               detections,
                               t_res,
                               f_res,
                               low_f,
                               high_f,
                               fn,
                               spectrogram_cmap='jet',
                               kernel_cmap='jet',
                               show_kernel=True):
        """
        Process all detections and create visualizations.

        Args:
            spectrogram: 2D array of spectrogram data
            time: Array of time values (x-axis)
            freqs: Array of frequency values (y-axis)
            detections: List of detection boxes, each as (x1, x2, y1, y2, D0)
            t_res: Time resolution
            f_res: Frequency resolution
            low_f: Lower frequency bound
            high_f: Higher frequency bound
            fn: Nominal frequency
            kernel_cmap: Colormap to use for the kernel
            spectrogram_cmap: Colormap to use for the spectrogram
            show_kernel: Whether to show the kernel

        Returns:
            List of paths to the saved figures
        """
        saved_figures = []
        for i, detection in enumerate(detections):
            saved_fig = self.process_detection(
                spectrogram, time, freqs, detection,
                t_res, f_res, low_f, high_f, fn,
                detection_index=i, kernel_cmap=kernel_cmap, show_kernel=show_kernel, spectrogram_cmap=spectrogram_cmap
            )
            saved_figures.append(saved_fig)
        return saved_figures
