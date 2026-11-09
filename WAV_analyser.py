#!/usr/bin/env python3
"""Report basic metadata and signal levels for a PCM WAV file."""

import argparse
import math
import wave
import readline
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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


class SQLiteWAVAnalysisWriter:
	"""Persist WAV analysis results and per-channel levels in SQLite."""

	def __init__(self, database_path: str | Path):
		self.database_path = str(Path(database_path).expanduser().resolve())

	def write(self, filepath: str | Path, results: WAVAnalysisResult) -> None:
		connection = sqlite3.connect(self.database_path)
		try:
			with connection:
				connection.execute("PRAGMA foreign_keys = ON")
				connection.execute(
					"""
					CREATE TABLE IF NOT EXISTS wav_analysis (
						id INTEGER PRIMARY KEY AUTOINCREMENT,
						filepath TEXT NOT NULL,
						duration_seconds REAL NOT NULL,
						sample_rate_hz INTEGER NOT NULL,
						bit_depth INTEGER NOT NULL,
						channel_count INTEGER NOT NULL,
						peak_level_dbfs REAL NOT NULL,
						rms_level_dbfs REAL NOT NULL,
						analyzed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
					)
					"""
				)
				connection.execute(
					"""
					CREATE TABLE IF NOT EXISTS wav_analysis_channels (
						analysis_id INTEGER NOT NULL,
						channel_number INTEGER NOT NULL,
						peak_level_dbfs REAL NOT NULL,
						rms_level_dbfs REAL NOT NULL,
						PRIMARY KEY (analysis_id, channel_number),
						FOREIGN KEY (analysis_id) REFERENCES wav_analysis(id)
							ON DELETE CASCADE
					)
					"""
				)
				cursor = connection.execute(
					"""
					INSERT INTO wav_analysis (
						filepath, duration_seconds, sample_rate_hz, bit_depth,
						channel_count, peak_level_dbfs, rms_level_dbfs
					) VALUES (?, ?, ?, ?, ?, ?, ?)
					""",
					(
						str(Path(filepath).expanduser().resolve()),
						results.duration_seconds,
						results.sample_rate_hz,
						results.bit_depth,
						results.channel_count,
						results.peak_level_dbfs,
						results.rms_level_dbfs,
					),
				)
				analysis_id = cursor.lastrowid
				if analysis_id is None:
					raise sqlite3.DatabaseError("Could not retrieve inserted analysis ID")
				connection.executemany(
					"""
					INSERT INTO wav_analysis_channels (
						analysis_id, channel_number, peak_level_dbfs, rms_level_dbfs
					) VALUES (?, ?, ?, ?)
					""",
					(
						(analysis_id, channel, peak, rms)
						for channel, (peak, rms) in enumerate(
							zip(
								results.channel_peak_levels_dbfs,
								results.channel_rms_levels_dbfs,
							),
							start=1,
						)
					),
				)
		finally:
			connection.close()


@dataclass(frozen=True)
class WAVFileMetadata:
	channel_count: int
	sample_rate_hz: int
	sample_width_bytes: int
	frame_count: int


class WAVFileValidator:
	"""Validate shared WAV properties and return the file metadata."""

	@staticmethod
	def validate(wav_file: wave.Wave_read) -> WAVFileMetadata:
		channel_count = wav_file.getnchannels()
		sample_rate_hz = wav_file.getframerate()
		sample_width_bytes = wav_file.getsampwidth()

		if sample_rate_hz <= 0:
			raise ValueError("WAV sample rate must be greater than 0 Hz")
		if channel_count <= 0:
			raise ValueError("WAV file must have at least one channel")
		if wav_file.getcomptype() != "NONE":
			raise ValueError("Only uncompressed PCM WAV files are supported")
		if sample_width_bytes not in (1, 2, 3, 4):
			raise ValueError(
				f"Unsupported PCM sample width: {sample_width_bytes} bytes"
			)

		return WAVFileMetadata(
			channel_count=channel_count,
			sample_rate_hz=sample_rate_hz,
			sample_width_bytes=sample_width_bytes,
			frame_count=wav_file.getnframes(),
		)


class WAVAnalyzer:
	def __init__(self, filepath: str):
		self.filepath = filepath

	def analyze(self) -> WAVAnalysisResult:
		with wave.open(self.filepath, "rb") as wav_file:
			metadata = WAVFileValidator.validate(wav_file)
			channels = metadata.channel_count
			sample_rate = metadata.sample_rate_hz
			sample_width = metadata.sample_width_bytes
			frame_count = metadata.frame_count

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


class WAVSpectrogramGenerator:
	"""Read a PCM WAV file and plot its mono spectrogram."""

	def __init__(
		self,
		filepath: str,
		segment_length: int = 1024,
		overlap: Optional[int] = None,
	):
		if segment_length < 1:
			raise ValueError("Segment length must be at least 1 sample")
		if overlap is not None and not 0 <= overlap < segment_length:
			raise ValueError("Overlap must be non-negative and less than segment length")

		self.filepath = filepath
		self.segment_length = segment_length
		self.overlap = overlap

	def generate(self	) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Return frequency bins, time bins, and power spectral density."""
		with wave.open(self.filepath, "rb") as wav_file:
			metadata = WAVFileValidator.validate(wav_file)
			channel_count = metadata.channel_count
			sample_rate = metadata.sample_rate_hz
			sample_width = metadata.sample_width_bytes
			frame_count = metadata.frame_count

			if frame_count == 0:
				raise ValueError("WAV file contains no audio samples")

			audio_data = wav_file.readframes(frame_count)

		bytes_per_frame = sample_width * channel_count
		if len(audio_data) != frame_count * bytes_per_frame:
			raise ValueError("WAV data does not match its declared frame count")
		samples = self._decode_pcm(audio_data, sample_width)
		full_scale = 1 << (sample_width * 8 - 1)
		samples = samples.reshape(-1, channel_count).astype(np.float64) / full_scale
		mono_signal = np.mean(samples, axis=1)

		segment_length = min(self.segment_length, len(mono_signal))
		overlap = self.overlap
		if overlap is None:
			overlap = segment_length // 2
		else:
			overlap = min(overlap, segment_length - 1)

		frequencies, times, power = spectrogram(
			mono_signal,
			fs=sample_rate,
			window="hann",
			nperseg=segment_length,
			noverlap=overlap,
			scaling="density",
			mode="psd",
		)
		return frequencies, times, power

	@staticmethod
	def _decode_pcm(audio_data: bytes, sample_width: int) -> np.ndarray:
		if sample_width == 1:
			return np.frombuffer(audio_data, dtype=np.uint8).astype(np.int16) - 128
		if sample_width == 2:
			return np.frombuffer(audio_data, dtype="<i2")
		if sample_width == 3:
			packed = np.frombuffer(audio_data, dtype=np.uint8).reshape(-1, 3)
			samples = (
				packed[:, 0].astype(np.int32)
				| (packed[:, 1].astype(np.int32) << 8)
				| (packed[:, 2].astype(np.int32) << 16)
			)
			return np.where(samples & 0x800000, samples - 0x1000000, samples)
		return np.frombuffer(audio_data, dtype="<i4")

	def plot(self) -> None:
		"""Display the WAV file's power spectral density over time."""
		frequencies, times, power = self.generate()
		power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny))
		figure, axis = plt.subplots(figsize=(12, 6))
		image = axis.pcolormesh(
			times,
			frequencies,
			power_db,
			shading="auto",
			cmap="magma",
		)
		axis.set(
			title=f"Spectrogram: {Path(self.filepath).name}",
			xlabel="Time (seconds)",
			ylabel="Frequency (Hz)",
		)
		figure.colorbar(image, ax=axis, label="Power spectral density (dB/Hz)")
		figure.tight_layout()
		plt.show()


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
	parser = argparse.ArgumentParser(
		description="Analyze a PCM WAV file or all WAV files in a folder"
	)
	parser.add_argument(
		"path",
		nargs="?",
		help="Path to a WAV file or folder (prompted if omitted)",
	)
	parser.add_argument(
		"--database",
		default="wav_analysis.db",
		help="SQLite database path (default: wav_analysis.db)",
	)
	args = parser.parse_args()

	try:
		path_text = args.path or input(
			"Enter the path to a WAV file or folder: "
		).strip()
		if not path_text:
			print("Error: a WAV file or folder path is required")
			return
		input_path = Path(path_text).expanduser()
		is_directory = input_path.is_dir()
		if is_directory:
			wav_files = sorted(
				(
					path
					for path in input_path.iterdir()
					if path.is_file() and path.suffix.lower() == ".wav"
				),
				key=lambda path: path.name.lower(),
			)
		else:
			wav_files = [input_path]
	except (OSError, EOFError) as error:
		print(f"Error: {error}")
		return

	if is_directory and not wav_files:
		print(f"No WAV files found in folder: {input_path}")
		return

	database_writer = SQLiteWAVAnalysisWriter(args.database)
	saved_results = False
	for wav_file in wav_files:
		try:
			results = WAVAnalyzer(str(wav_file)).analyze()
		except (OSError, wave.Error, ValueError) as error:
			print(f"Error reading {wav_file}: {error}")
			continue

		try:
			database_writer.write(wav_file, results)
			saved_results = True
		except sqlite3.Error as error:
			print(f"Error saving analysis for {wav_file}: {error}")

	if saved_results:
		print(f"Results saved to: {database_writer.database_path}")


if __name__ == "__main__":
	main()
