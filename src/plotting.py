import numpy as np
from matplotlib import patches, pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union


@dataclass
class FontConfig:
    """Configuration for font sizes and line widths."""
    title: int = 24
    legend: int = 18
    label: int = 16
    ticks: int = 12
    linewidth: int = 1


class SpectrogramPlot:
    """Class for creating spectrogram plots with various customization options."""

    def __init__(self, font_config: Optional[FontConfig] = None):
        """
        Initialize the SpectrogramPlot with font configuration.

        Args:
            font_config: Configuration for font sizes and line widths
        """
        self.font_config = font_config or FontConfig()

    def calculate_figure_size(self, time_freq_ratio: float, freq_size: int = 4) -> Tuple[int, int]:
        """
        Calculate the figure size based on time to frequency ratio.

        Args:
            time_freq_ratio: Ratio of time to frequency
            freq_size: Base size for frequency dimension

        Returns:
            Tuple of height and width
        """
        height = freq_size * time_freq_ratio
        width = freq_size
        return height, width

    def draw_detection_boxes(self,
                             detections: List[Tuple[float, float, float, float, float]],
                             color: str = 'white'): #cornflowerblue
        """
        Draw detection boxes on the current plot.

        Args:
            detections: List of detection boxes, each as (x1, x2, y1, y2, D0)
            color: Color for the boxes and text
        """
        for x1, x2, y1, y2, D0 in detections:
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=self.font_config.linewidth,
                edgecolor=color,
                facecolor='none'
            )
            plt.text(
                x1, y2,
                f'{D0:.2f}',
                fontsize=self.font_config.ticks,
                bbox={'facecolor': color, 'pad': 2, 'ec': color}
            )
            plt.gca().add_patch(rect)

    def plot(self,
             spec: np.ndarray,
             time: Optional[np.ndarray] = None,
             freq: Optional[np.ndarray] = None,
             figsize: Optional[Tuple[int, int]] = None,
             xlabel: str = 'Time [s]',
             ylabel: str = 'Frequency [kHz]',
             zlabel: str = 'Spectrum magnitude [dB]',
             xaxis: bool = True,
             yaxis: bool = True,
             ticks: List[int] = [1, 2],
             title: Optional[str] = None,
             cmap: str = 'jet',
             show_colorbar: bool = False,
             vlines: List[float] = [],
             detections: List[Tuple[float, float, float, float, float]] = [],
             labels: List[str] = [],
             labels_position: List[float] = [],
             save: bool = False,
             file_name: str = "spectrogram.png",
             show: bool = True) -> plt.Figure:
        """
        Create a spectrogram plot with customizable options.

        Args:
            spec: 2D array of spectrogram data
            time: Array of time values (x-axis)
            freq: Array of frequency values (y-axis)
            figsize: Figure size as (width, height)
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            zlabel: Label for color scale
            xaxis: Whether to use provided time values for x-axis
            yaxis: Whether to use provided frequency values for y-axis
            ticks: Tick intervals for [x, y] axes
            title: Plot title
            cmap: Colormap name
            show_colorbar: Whether to show the colorbar
            vlines: List of x-positions to draw vertical lines
            detections: List of detection boxes, each as (x1, x2, y1, y2, D0)
            labels: Text labels for x-axis
            labels_position: Positions for text labels
            save: Whether to save the figure to a file
            file_name: File name when saving the figure
            show: Whether to display the figure

        Returns:
            The matplotlib Figure object
        """
        # Create figure
        fig = plt.figure(figsize=figsize)

        # Draw vertical lines
        for line_pos in vlines:
            plt.axvline(line_pos, color='black', linewidth=self.font_config.linewidth)

        # Create the spectrogram plot
        if xaxis and yaxis and time is not None and freq is not None:
            img = plt.pcolormesh(time, freq, spec, cmap=cmap)
            if ticks:
                if labels and labels_position:
                    plt.xticks(labels_position, labels=labels, fontsize=self.font_config.ticks)
                plt.yticks(
                    np.arange(freq[0], freq[-1], ticks[1]),
                    fontsize=self.font_config.ticks
                )
        else:
            img = plt.pcolormesh(spec, cmap=cmap)
            if ticks:
                plt.xticks(
                    np.arange(0, spec.shape[1], ticks[0]),
                    fontsize=self.font_config.ticks
                )
                plt.yticks(
                    np.arange(0, spec.shape[0], ticks[1]),
                    fontsize=self.font_config.ticks
                )

        # Add colorbar if requested
        if show_colorbar:
            cbar = fig.colorbar(mappable=img, label=zlabel)
            cbar.ax.tick_params(labelsize=self.font_config.ticks)
            cbar.set_label(zlabel, fontsize=self.font_config.label)

        # Add labels and title
        if xlabel:
            plt.xlabel(xlabel, fontsize=self.font_config.label)
        if ylabel:
            plt.ylabel(ylabel, fontsize=self.font_config.label)
        if title:
            plt.title(title, fontsize=self.font_config.title)

        # Draw detection boxes
        self.draw_detection_boxes(detections)

        plt.tight_layout()
        # Save or show the figure
        if save:
            plt.savefig(file_name, bbox_inches='tight', dpi=300)
            plt.close()
        elif show:
            plt.show()

        return fig

