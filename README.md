# WAV file analyser

## Generate and plot sine waves

Run `python signal_generate.py` to generate 100 Hz, 500 Hz, 1000 Hz, 5000 Hz,
and 10000 Hz sine waves. Each `SineWaveGenerator` instance takes one
`frequency_hz` parameter. `SignalSummer` adds the waves sample by sample, then
`ButterworthNotchFilter` applies a zero-phase 300–1000 Hz band-stop filter.
The program plots only the filtered sum and its single-sided Fourier amplitude
spectrum.

The default sample rate is 48 kHz, bit depth is 24 bits, and signal duration is
one second. Samples are quantized to the selected bit depth and normalized for
plotting. The waveform plot shows the first 20 ms; the spectrum uses the full
signal with a Hann window. Install NumPy, Matplotlib, and SciPy if they are not
already available:

```bash
python -m pip install numpy matplotlib scipy
```

## Plot a WAV spectrogram

Run `python WAV_analyser.py path/to/audio.wav` to report WAV analysis results
and display a power spectrogram. If no path is supplied, the program prompts
for one. Stereo channels are averaged to mono. The spectrogram uses
1024-sample Hann windows with 50% overlap by default.