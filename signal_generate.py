#!/usr/bin/env python3
"""Generate sine waves and plot their waveforms and frequency spectra."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt


class SineWaveGenerator:
    """Generate one quantized sine wave at the requested frequency."""

    def __init__(
        self,
        frequency_hz: float,
        sample_rate_hz: int = 48000,
        bit_depth: int = 24,
        duration_seconds: float = 1.0,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("Sample rate must be greater than 0 Hz")
        if bit_depth < 2:
            raise ValueError("Bit depth must be at least 2 bits")
        if duration_seconds <= 0:
            raise ValueError("Duration must be greater than 0 seconds")
        if frequency_hz <= 0:
            raise ValueError("Frequency must be greater than 0 Hz")
        if frequency_hz >= sample_rate_hz / 2:
            raise ValueError("Frequencies must be below the Nyquist frequency")

        self.frequency_hz = frequency_hz
        self.sample_rate_hz = sample_rate_hz
        self.bit_depth = bit_depth
        self.duration_seconds = duration_seconds

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return sample times and a normalized, bit-depth-quantized waveform."""
        sample_count = int(self.sample_rate_hz * self.duration_seconds)
        if sample_count < 1:
            raise ValueError("Duration must produce at least one sample")

        time_seconds = np.arange(sample_count) / self.sample_rate_hz
        peak_level = (1 << (self.bit_depth - 1)) - 1
        waveform = np.sin(2 * np.pi * self.frequency_hz * time_seconds)
        waveform = np.rint(waveform * peak_level) / peak_level
        return time_seconds, waveform


class SignalSummer:
    """Combine multiple waveforms by adding their sample values."""

    def sum_signals(self, signals: Mapping[float, np.ndarray]) -> np.ndarray:
        if not signals:
            raise ValueError("At least one signal is required")

        waveforms = list(signals.values())
        if any(waveform.shape != waveforms[0].shape for waveform in waveforms[1:]):
            raise ValueError("All signals must have the same shape")
        return np.sum(waveforms, axis=0)


class ButterworthNotchFilter:
    """Apply a zero-phase Butterworth band-stop filter."""

    def __init__(
        self,
        sample_rate_hz: int,
        low_cutoff_hz: float = 300.0,
        high_cutoff_hz: float = 1000.0,
        order: int = 4,
    ) -> None:
        nyquist_hz = sample_rate_hz / 2
        if sample_rate_hz <= 0:
            raise ValueError("Sample rate must be greater than 0 Hz")
        if not 0 < low_cutoff_hz < high_cutoff_hz < nyquist_hz:
            raise ValueError(
                "Cutoff frequencies must satisfy 0 < low < high < Nyquist"
            )
        if order < 1:
            raise ValueError("Filter order must be at least 1")

        self.sample_rate_hz = sample_rate_hz
        self.low_cutoff_hz = low_cutoff_hz
        self.high_cutoff_hz = high_cutoff_hz
        self.order = order

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Return the input after suppressing frequencies in the stop band."""
        if signal.ndim != 1 or len(signal) == 0:
            raise ValueError("Signal must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(signal)):
            raise ValueError("Signal must contain only finite values")

        sections = butter(
            self.order,
            (self.low_cutoff_hz, self.high_cutoff_hz),
            btype="bandstop",
            fs=self.sample_rate_hz,
            output="sos",
        )
        return sosfiltfilt(sections, signal)


class SignalPlotter:
    """Plot one waveform and its single-sided Fourier amplitude spectrum."""

    def __init__(self, sample_rate_hz: int) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("Sample rate must be greater than 0 Hz")
        self.sample_rate_hz = sample_rate_hz

    def plot(self, time_seconds: np.ndarray, signal: np.ndarray) -> None:
        if len(time_seconds) == 0:
            raise ValueError("At least one sample is required")
        if signal.shape != time_seconds.shape:
            raise ValueError("Signal and time arrays must have the same shape")

        figure, (waveform_axis, spectrum_axis) = plt.subplots(
            1, 2, figsize=(12, 4)
        )
        display_duration = min(0.02, time_seconds[-1] + 1 / self.sample_rate_hz)
        visible_samples = time_seconds < display_duration
        waveform_axis.plot(
            time_seconds[visible_samples],
            signal[visible_samples],
        )
        waveform_axis.set(
            title="Filtered summed waveform",
            xlabel="Time (seconds)",
            ylabel="Amplitude",
        )
        waveform_axis.grid(True)

        window = np.hanning(len(signal))
        spectrum = np.fft.rfft(signal * window)
        frequencies = np.fft.rfftfreq(
            len(signal), d=1 / self.sample_rate_hz
        )
        amplitude = np.abs(spectrum) / np.sum(window)
        if len(signal) > 1:
            if len(signal) % 2 == 0:
                amplitude[1:-1] *= 2
            else:
                amplitude[1:] *= 2
        spectrum_axis.plot(frequencies, amplitude)
        spectrum_axis.set(
            title="Filtered summed signal Fourier transform",
            xlabel="Frequency (Hz)",
            ylabel="Amplitude",
            xlim=(0, self.sample_rate_hz / 2),
        )
        spectrum_axis.grid(True)

        figure.tight_layout()
        plt.show()


def main() -> None:
    generators = [
        SineWaveGenerator(frequency_hz=frequency)
        for frequency in (100.0, 500.0, 1000.0, 5000.0, 10000.0)
    ]
    time_seconds, first_signal = generators[0].generate()
    signals = {generators[0].frequency_hz: first_signal}
    for generator in generators[1:]:
        _, signal = generator.generate()
        signals[generator.frequency_hz] = signal
    summed_signal = SignalSummer().sum_signals(signals)
    sample_rate_hz = generators[0].sample_rate_hz
    filtered_signal = ButterworthNotchFilter(sample_rate_hz).apply(summed_signal)
    SignalPlotter(sample_rate_hz).plot(time_seconds, filtered_signal)


if __name__ == "__main__":
    main()