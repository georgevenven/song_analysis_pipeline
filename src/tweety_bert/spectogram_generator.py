import os
import numpy as np
from scipy.signal import spectrogram, windows, ellip, filtfilt
import matplotlib.pyplot as plt
import multiprocessing
import soundfile as sf
import gc
import csv
import sys
import psutil

class WavtoSpec:
    def __init__(self, src_dir, dst_dir, csv_file_dir=None):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.csv_file_dir = csv_file_dir
        self.use_csv = csv_file_dir is not None

    def process_directory(self):
        song_info = self.read_csv_file(self.csv_file_dir) if self.use_csv else {}
        audio_files = [os.path.join(root, file) 
                       for root, dirs, files in os.walk(self.src_dir) 
                       for file in files if file.lower().endswith('.wav')]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = [pool.apply_async(self.process_file, args=(self, file_path, song_info)) 
                       for file_path in audio_files]
            for result in results:
                result.get()

    @staticmethod
    def process_file(instance, file_path, song_info):
        instance.convert_to_spectrogram(file_path, song_info)

    def convert_to_spectrogram(self, file_path, song_info, min_length_ms=1000):
        try:
            with sf.SoundFile(file_path, 'r') as wav_file:
                samplerate = wav_file.samplerate
                data = wav_file.read(dtype='int16')
                if wav_file.channels > 1:
                    data = data[:, 0]

            length_in_ms = (len(data) / samplerate) * 1000
            if length_in_ms < min_length_ms:
                print(f"File {file_path} is below the length threshold and will be skipped.")
                return

            b, a = ellip(5, 0.2, 40, 500/(samplerate/2), 'high')
            data = filtfilt(b, a, data)

            NFFT = 1024
            step_size = 119
            overlap_samples = NFFT - step_size
            window = windows.gaussian(NFFT, std=NFFT/8)

            # find a better way than this, highly inelegant 
            song_name = '.'.join(os.path.basename(file_path).split('.')[0:-1])

            segments_to_process = self.get_segments_to_process(song_name=song_name, song_info=song_info, samplerate=samplerate)
            if segments_to_process is None:
                print(f"No segments to process for {file_path}. Skipping spectrogram generation.")
                return  # Skip processing if no segments are defined

            for start_sample, end_sample in segments_to_process:
                segment_data = data[start_sample:end_sample]
                f, t, Sxx = spectrogram(segment_data, fs=samplerate, window=window, nperseg=NFFT, noverlap=overlap_samples)
                Sxx_log = 10 * np.log10(Sxx + 1e-6)
                Sxx_log_clipped = np.clip(Sxx_log, a_min=-2, a_max=None)
                Sxx_log_normalized = (Sxx_log_clipped - np.min(Sxx_log_clipped)) / (np.max(Sxx_log_clipped) - np.min(Sxx_log_clipped))

                spec_filename = os.path.splitext(os.path.basename(file_path))[0]
                spec_file_path = os.path.join(self.dst_dir, spec_filename + '_' + str(start_sample) + '.npz')
                np.savez_compressed(spec_file_path, s=Sxx_log_normalized)

                print(f"Spectrogram saved to {spec_file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        finally:
            plt.close('all')
            gc.collect()

    def read_csv_file(self, csv_file):
        song_info = {}
        csv.field_size_limit(sys.maxsize)
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                song_name = row['song_name']
                song_ms = eval(row['song_ms'])
                song_info[song_name] = song_ms
        return song_info

    def get_segments_to_process(self, song_name, song_info, samplerate):
        segments_to_process = []
        if song_name in song_info:
            for start_ms, end_ms in song_info[song_name]:
                start_sample = int(start_ms * samplerate / 1000)
                end_sample = int(end_ms * samplerate / 1000)
                segments_to_process.append((start_sample, end_sample))
        else:
            return None  # Return None to indicate no segments to process
        return segments_to_process
