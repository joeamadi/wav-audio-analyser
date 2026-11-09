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

Run `python WAV_analyser.py path/to/audio.wav` to analyze the WAV file and save
the results to `wav_analysis.db`. You can also pass a folder to analyze every
WAV file directly inside it (not including subfolders), for example
`python WAV_analyser.py path/to/audio_folder`.
If no path is supplied, the program prompts for a file or folder. Use
`--database path/to/results.db` to choose another SQLite database. Aggregate
results are stored in `wav_analysis`, with per-channel levels stored in
`wav_analysis_channels`. The reporter and spectrogram classes are available for
use directly, but the command-line entry point does not print results or show
plots. After a successful save, it prints the full database location.