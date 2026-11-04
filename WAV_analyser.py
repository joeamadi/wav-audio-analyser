#!/usr/bin/env python3
"""Report basic metadata and signal levels for a PCM WAV file."""

import argparse
import math
import wave
import readline
from dataclasses import dataclass
from typing import Dict


@dataclass
class WAVAnalysisResult:
	duration_seconds: float
	sample_rate_hz: int
	bit_depth: int
	channel_count: int
	peak_level_dbfs: float
	rms_level_dbfs: float
	channel_peak_levels_dbfs: tuple
	channel_rms_levels_dbfs: tuple

	def as_dict(self) -> Dict[str, object]:
		return {
			"duration_seconds": self.duration_seconds,
			"sample_rate_hz": self.sample_rate_hz,
			"bit_depth": self.bit_depth,
			"channel_count": self.channel_count,
			"peak_level_dbfs": self.peak_level_dbfs,
			"rms_level_dbfs": self.rms_level_dbfs,
			"channel_peak_levels_dbfs": self.channel_peak_levels_dbfs,
			"channel_rms_levels_dbfs": self.channel_rms_levels_dbfs,
		}


class WAVAnalyzer:
	def __init__(self, filepath: str):
		self.filepath = filepath

	def analyze(self) -> WAVAnalysisResult:
		with wave.open(self.filepath, "rb") as wav_file:
			channels = wav_file.getnchannels()
			sample_rate = wav_file.getframerate()
			sample_width = wav_file.getsampwidth()
			frame_count = wav_file.getnframes()

			if sample_rate == 0:
				raise ValueError("WAV sample rate must be greater than 0 Hz")
			if wav_file.getcomptype() != "NONE":
				raise ValueError("Only uncompressed PCM WAV files are supported")
			if sample_width not in (1, 2, 3, 4):
				raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")

			bit_depth = sample_width * 8
			full_scale = 1 << (bit_depth - 1)
			peak = 0
			sum_squares = 0
			sample_count = 0
			sample_index = 0
			channel_peaks = [0] * channels
			channel_sum_squares = [0] * channels
			channel_sample_counts = [0] * channels
			chunk_size = 4096

			while True:
				audio_data = wav_file.readframes(chunk_size)
				if not audio_data:
					break

				chunk_sample_count = len(audio_data) // sample_width
				sample_count += chunk_sample_count
				for offset in range(0, chunk_sample_count * sample_width, sample_width):
					sample_bytes = audio_data[offset : offset + sample_width]
					if sample_width == 1:
						# 8-bit PCM WAV samples are unsigned.
						sample = sample_bytes[0] - 128
					else:
						sample = int.from_bytes(
							sample_bytes, byteorder="little", signed=True
						)
					magnitude = abs(sample)
					channel_index = sample_index % channels
					peak = max(peak, magnitude)
					sum_squares += sample * sample
					channel_peaks[channel_index] = max(
						channel_peaks[channel_index], magnitude
					)
					channel_sum_squares[channel_index] += sample * sample
					channel_sample_counts[channel_index] += 1
					sample_index += 1

		peak_dbfs = 20 * math.log10(peak / full_scale) if peak else float("-inf")
		rms = math.sqrt(sum_squares / sample_count) if sample_count else 0.0
		rms_dbfs = 20 * math.log10(rms / full_scale) if rms else float("-inf")
		channel_peak_dbfs = tuple(
			20 * math.log10(channel_peak / full_scale) if channel_peak else float("-inf")
			for channel_peak in channel_peaks
		)
		channel_rms_dbfs = tuple(
			20 * math.log10(
				math.sqrt(channel_sum / channel_count) / full_scale
			) if channel_sum else float("-inf")
			for channel_sum, channel_count in zip(channel_sum_squares, channel_sample_counts)
		)

		return WAVAnalysisResult(
			duration_seconds=frame_count / sample_rate,
			sample_rate_hz=sample_rate,
			bit_depth=bit_depth,
			channel_count=channels,
			peak_level_dbfs=peak_dbfs,
			rms_level_dbfs=rms_dbfs,
			channel_peak_levels_dbfs=channel_peak_dbfs,
			channel_rms_levels_dbfs=channel_rms_dbfs,
		)


class WAVAnalysisReporter:
	def print_results(self, results: WAVAnalysisResult):
		print(f"Duration: {results.duration_seconds:.3f} seconds")
		print(f"Sample rate: {results.sample_rate_hz} Hz")
		print(f"Bit depth: {results.bit_depth} bits")
		print(f"Channels: {results.channel_count}")
		print(f"Overall peak level: {self._format_level(results.peak_level_dbfs)}")
		print(f"Overall RMS level: {self._format_level(results.rms_level_dbfs)}")
		for channel, (peak, rms) in enumerate(
			zip(results.channel_peak_levels_dbfs, results.channel_rms_levels_dbfs),
			start=1,
		):
			print(f"Channel {channel} peak level: {self._format_level(peak)}")
			print(f"Channel {channel} RMS level: {self._format_level(rms)}")

	@staticmethod
	def _format_level(value: float) -> str:
		return f"{value:.2f} dBFS" if math.isfinite(value) else "-inf dBFS"


def main():
	parser = argparse.ArgumentParser(description="Analyze a PCM WAV file")
	parser.add_argument(
		"wav_file",
		nargs="?",
		help="Path to the WAV file (prompted if omitted)",
	)
	args = parser.parse_args()
	wav_file = args.wav_file or input("Enter the path to the WAV file: ").strip()

	try:
		results = WAVAnalyzer(wav_file).analyze()
	except (OSError, EOFError, wave.Error, ValueError) as error:
		print(f"Error: {error}")
		return

	WAVAnalysisReporter().print_results(results)


if __name__ == "__main__":
	main()
