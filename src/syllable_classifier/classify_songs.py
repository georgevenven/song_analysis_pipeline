# convert wav files to spectograms 
from src.tweety_bert.spectogram_generator import WavtoSpec
from tqdm import tqdm
import tempfile
import os 
import shutil
import torch 
import numpy as np
import pandas as pd 
from scipy.io import wavfile
from pathlib import Path
from scipy.signal import spectrogram, windows, ellip, filtfilt
import soundfile as sf
import matplotlib.pyplot as plt
import gc
import csv
import sys
import psutil
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F

class Inference():
    def __init__(self, input_path=None, output_path=None, plot_spec_results=False, model=None, sorted_songs_path=None, threshold=.5, min_length=500, pad_song=50):
        """
        input path == nested bird song structure
        """
        self.input_path = input_path
        self.output_path = output_path
        self.plot_spec_results = plot_spec_results

        self.threshold=threshold
        self.min_length=min_length
        self.pad_song=pad_song

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.sorted_songs_path = sorted_songs_path

        # make sure output dir exists 
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if plot_spec_results:
            if not os.path.exists(self.output_path):
                os.makedirs(os.path.join(self.output_path, 'specs'))
        
        # class if onset/offset is a dictionary where a key is the class id and the value is a list of the onset/offset (there are two one for bins one for ms)
        self.database = pd.DataFrame(columns=["song_name", "directory", "class_id_onset_offset_ms", "class_id_onset_offset_bins"])


    def calculate_length_ms(sample_rate, data):
        """
        Calculate the length of an audio file in milliseconds.

        Parameters:
        - sample_rate: int, the sample rate of the audio file (samples per second).
        - data: numpy.ndarray, the audio data.

        Returns:
        - length_ms: float, the length of the audio file in milliseconds.
        """
        # Calculate the number of samples
        num_samples = data.shape[0]

        # Calculate the duration in seconds
        duration_seconds = num_samples / sample_rate

        # Convert the duration to milliseconds
        length_ms = duration_seconds * 1000

        return length_ms

    def process_spectrogram(self, model, spec, device, max_length=1000):
        """
        Process the spectrogram in chunks, pass through the classifier, and return the binary predictions based on BCE.
        """
        # Calculate the number of chunks needed
        num_chunks = int(np.ceil(spec.shape[1] / max_length))
        combined_predictions = []
        actual_lengths = []  # List to store the actual lengths of each chunk

        for i in range(num_chunks):
            # Extract the chunk
            start_idx = i * max_length
            end_idx = min((i + 1) * max_length, spec.shape[1])
            chunk = spec[:, start_idx:end_idx]
            actual_length = chunk.shape[1]  # Store the actual length before padding
            actual_lengths.append(actual_length)

            # Pad the chunk if it's less than max_length
            if chunk.shape[1] < max_length:
                padding = np.zeros((chunk.shape[0], max_length - chunk.shape[1]))
                chunk = np.concatenate((chunk, padding), axis=1)

            # Forward pass through the model
            # Ensure chunk is on the correct device
            chunk_tensor = torch.Tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(chunk_tensor)
            logits = logits.squeeze().detach().cpu()

            # logit shape is (length, logits) --> they are raw values so we need to take the sigmoid of them 
            logits = torch.sigmoid(logits) 

            combined_predictions.append(logits)

        # Concatenate all chunks' predictions
        final_predictions = np.concatenate(combined_predictions, axis=0)

        # Adjust the final predictions to remove padding by slicing according to actual_lengths
        final_predictions = final_predictions[:, :sum(actual_lengths)]

        return final_predictions

    def create_spectrogram(self, file_path, song_info, min_length_ms=1000):
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

            song_name = '.'.join(os.path.basename(file_path).split('.')[0:-1])
            segments_to_process = self.get_segments_to_process(song_name=song_name, song_info=song_info, samplerate=samplerate)
            if segments_to_process is None:
                print(f"No segments to process for {file_path}. Skipping spectrogram generation.")
                return  # Skip processing if no segments are defined

            full_spectrogram = None

            for start_sample, end_sample in segments_to_process:
                segment_data = data[start_sample:end_sample]
                f, t, Sxx = spectrogram(segment_data, fs=samplerate, window=window, nperseg=NFFT, noverlap=overlap_samples)
                Sxx_log = 10 * np.log10(Sxx + 1e-6)
                Sxx_log_clipped = np.clip(Sxx_log, a_min=-2, a_max=None)
                Sxx_log_normalized = (Sxx_log_clipped - np.min(Sxx_log_clipped)) / (np.max(Sxx_log_clipped) - np.min(Sxx_log_clipped))

                if full_spectrogram is None:
                    full_spectrogram = Sxx_log_normalized
                else:
                    full_spectrogram = np.hstack((full_spectrogram, Sxx_log_normalized))

            return full_spectrogram

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
        # convert song_info to a dictionary 
        song_info = {row['song_name']: eval(row['song_ms']) for _, row in song_info.iterrows()}

        segments_to_process = []
        if song_name in song_info:
            for start_ms, end_ms in song_info[song_name]:
                start_sample = int(start_ms * samplerate / 1000)
                end_sample = int(end_ms * samplerate / 1000)
                segments_to_process.append((start_sample, end_sample))
        else:
            return None  # Return None to indicate no segments to process
        return segments_to_process

        
    def classify_all_songs(self):
        rows_to_add = []  # Initialize an empty list to collect rows

        specs_dir = os.path.join(self.output_path, 'specs')
        if not os.path.exists(specs_dir):
                os.makedirs(specs_dir)

        total_songs = 0
        for bird in os.listdir(self.input_path):
            bird_path = os.path.join(self.input_path, bird)
            if os.path.isdir(bird_path):
                for day in os.listdir(bird_path):
                    day_path = os.path.join(bird_path, day)
                    if os.path.isdir(day_path):
                        total_songs += len([song for song in os.listdir(day_path) if os.path.isfile(os.path.join(day_path, song))])

        song_progress = tqdm(total=total_songs, desc="Classifying Syllables in Song")
        save_interval = 100

        for bird in tqdm(os.listdir(self.input_path), desc="Processing birds"):
            bird_path = os.path.join(self.input_path, bird)
            if not os.path.isdir(bird_path):
                continue

            for day in os.listdir(bird_path):
                day_path = os.path.join(bird_path, day)
                if not os.path.isdir(day_path):
                    continue

                for song in os.listdir(day_path):
                    song_src_path = os.path.join(day_path, song)
                    song_name = Path(song).stem

                    # Check if the song has already been processed
                    if song_name in self.database['song_name'].values:
                        print(f"Skipping {song_name}, already processed.")
                        song_progress.update(1)
                        continue

                    try:
                        sorted_songs_df = pd.read_csv(self.sorted_songs_path)
                        if not sorted_songs_df[sorted_songs_df['song_name'] == song_name].empty:
                            song_info = sorted_songs_df.loc[sorted_songs_df['song_name'] == song_name].iloc[0].to_dict()
                        else:
                            print(f"No sorted information available for {song_name}, skipping.")
                            continue

                        spec = self.create_spectrogram(file_path=song_src_path, song_info=sorted_songs_df)

                        spec = spec[20:216]

                        # zscore spec
                        spec = (spec - np.mean(spec)) / np.std(spec)

                        if spec is None:
                            continue

                        sample_rate, _ = wavfile.read(song_src_path)
                        predictions = self.process_spectrogram(model=self.model, spec=spec, device=self.device, max_length=1000)
                        class_ids = np.argmax(predictions, axis=1)

                        # Plotting the spectrogram with class labels if enabled
                        if self.plot_spec_results:
                            plt.figure(figsize=(10, 8))
                            plt.subplot(2, 1, 1)
                            plt.imshow(spec, aspect='auto', origin='lower')
                            plt.title(f"Spectrogram for {song_name}")
                            plt.xlabel('Time')
                            plt.ylabel('Frequency')

                            plt.subplot(2, 1, 2)
                            class_bar = np.repeat(class_ids[np.newaxis, :], 10, axis=0)  # Repeat class_ids to make visible bar
                            plt.imshow(class_bar, aspect='auto', extent=[0, spec.shape[1], 0, 1], cmap='tab10')
                            plt.title(f"Class Labels for {song_name}")
                            plt.xlabel('Time')
                            plt.ylabel('Class')
                            plt.tight_layout()
                            plt.savefig(os.path.join(specs_dir, f"{song_name}.png"))
                            plt.close()

                        # Calculate the duration of each timebin based on the wav file
                        wav_length_ms = (len(spec) / sample_rate) * 1000  # Total duration of the wav file in milliseconds
                        timebin_duration_ms = wav_length_ms / len(class_ids)  # Duration of each timebin in milliseconds

                        # Initialize song_ms as an empty list
                        song_ms = []
                        start_index = None  # Use index to track the start of a song segment

                        for index, class_id in enumerate(class_ids):
                            if start_index is None or class_ids[start_index] != class_id:
                                if start_index is not None:
                                    # The current song segment ends
                                    end_index = index  # Note the end index of the segment
                                    start_ms = start_index * timebin_duration_ms  # Calculate start ms of the segment
                                    end_ms = end_index * timebin_duration_ms  # Calculate end ms of the segment
                                    song_ms.append({"class_id": class_ids[start_index], "start_ms": start_ms, "end_ms": end_ms})  # Append the start and end ms as a tuple
                                # A new song segment starts
                                start_index = index  # Note the start index of the segment

                        # Check for a segment that might end at the last index
                        if start_index is not None:
                            end_index = len(class_ids)  # The end index is the length of the class_ids array
                            start_ms = start_index * timebin_duration_ms  # Calculate start ms of the last segment
                            end_ms = end_index * timebin_duration_ms  # Calculate end ms of the last segment
                            song_ms.append({"class_id": class_ids[start_index], "start_ms": start_ms, "end_ms": end_ms})  # Append the start and end ms as a tuple

                        new_row = {"song_name": song_name, "directory": song_src_path, "class_probabilities": predictions.tolist(), "class_id_timebins": class_ids.tolist(), "class_ms": song_ms}
                        rows_to_add.append(new_row)  # Append the new row to the list

                        song_progress.update(1)  # Update progress bar for each song processed

                        # Check if it's time to save progress
                        if len(rows_to_add) >= save_interval:
                            # Concatenate new rows to the DataFrame and save to CSV
                            self.database = pd.concat([self.database, pd.DataFrame(rows_to_add)], ignore_index=True)
                            self.database.to_csv(output_csv_path, index=False)
                            rows_to_add = []  # Clear the list after saving

                    except Exception as e:
                        print(f"Error processing song {song}: {e}")

            # Close the progress bar
            song_progress.close()

        # After the loop, concatenate any remaining rows to the DataFrame and save to CSV
        output_csv_path = os.path.join(self.output_path, "classified_songs.csv")  # Specify the CSV filename
        if rows_to_add:
            self.database = pd.concat([self.database, pd.DataFrame(rows_to_add)], ignore_index=True)
            self.database.to_csv(output_csv_path, index=False)  # Use the new path with filename
            print(f"Database saved to {output_csv_path}")


