#!/usr/bin/env python3
"""Report basic metadata and signal levels for a PCM WAV file."""

import math
import wave
import readline
from dataclasses import dataclass


@dataclass
class WAVAnalysisResult:
	duration_seconds: float
	sample_rate_hz: int
	bit_depth: int
	channel_count: int
	peak_level_dbfs: float
	rms_level_dbfs: float

	def as_dict(self):
		return {
			"duration_seconds": self.duration_seconds,
			"sample_rate_hz": self.sample_rate_hz,
			"bit_depth": self.bit_depth,
			"channel_count": self.channel_count,
			"peak_level_dbfs": self.peak_level_dbfs,
			"rms_level_dbfs": self.rms_level_dbfs,
		}


class WAVAnalyzer:
	def __init__(self, filepath):
		self.filepath = filepath

	def analyze(self):
		with wave.open(self.filepath, "rb") as wav_file:
			channels = wav_file.getnchannels()
			sample_rate = wav_file.getframerate()
			sample_width = wav_file.getsampwidth()
			frame_count = wav_file.getnframes()

			if wav_file.getcomptype() != "NONE":
				raise ValueError("Only uncompressed PCM WAV files are supported")
			if sample_width not in (1, 2, 3, 4):
				raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")

			audio_data = wav_file.readframes(frame_count)

		bit_depth = sample_width * 8
		full_scale = 1 << (bit_depth - 1)
		peak = 0
		sum_squares = 0
		sample_count = len(audio_data) // sample_width

		for offset in range(0, len(audio_data), sample_width):
			sample_bytes = audio_data[offset : offset + sample_width]
			if sample_width == 1:
				# 8-bit PCM WAV samples are unsigned.
				sample = sample_bytes[0] - 128
			else:
				sample = int.from_bytes(sample_bytes, byteorder="little", signed=True)
			magnitude = abs(sample)
			peak = max(peak, magnitude)
			sum_squares += sample * sample

		peak_dbfs = 20 * math.log10(peak / full_scale) if peak else float("-inf")
		rms = math.sqrt(sum_squares / sample_count) if sample_count else 0.0
		rms_dbfs = 20 * math.log10(rms / full_scale) if rms else float("-inf")

		return WAVAnalysisResult(
			duration_seconds=frame_count / sample_rate if sample_rate else 0.0,
			sample_rate_hz=sample_rate,
			bit_depth=bit_depth,
			channel_count=channels,
			peak_level_dbfs=peak_dbfs,
			rms_level_dbfs=rms_dbfs,
		)


def main():
	wav_file = input("Enter the path to the WAV file: ").strip()

	try:
		results = WAVAnalyzer(wav_file).analyze()
	except (OSError, EOFError, wave.Error, ValueError) as error:
		print(f"Error: {error}")
		return

	for name, value in results.as_dict().items():
		if name == "duration_seconds":
			print(f"Duration: {value:.3f} seconds")
		elif name == "sample_rate_hz":
			print(f"Sample rate: {value} Hz")
		elif name == "bit_depth":
			print(f"Bit depth: {value} bits")
		elif name == "channel_count":
			print(f"Channels: {value}")
		else:
			formatted = f"{value:.2f} dBFS" if math.isfinite(value) else "-inf dBFS"
			print(f"{name.replace('_', ' ').capitalize()}: {formatted}")


if __name__ == "__main__":
	main()
