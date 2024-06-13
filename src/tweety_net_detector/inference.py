# convert wav files to spectograms 
from src.tweety_net_detector.model import TweetyNet
from src.tweety_net_detector.utils import load_model
from src.tweety_net_detector.spectogram_generator import WavtoSpec
import src.tweety_net_detector.post_processing as post_processing 
from tqdm import tqdm
import tempfile
import os 
import shutil
import torch 
import numpy as np
import pandas as pd 
from scipy.io import wavfile
from pathlib import Path

class Inference():
    def __init__(self, input_path=None, output_path=None, plot_spec_results=False, model_path=None, threshold=.5, min_length=500, pad_song=50):
        """
        input path == nested bird song structure
        """
        self.input_path = input_path
        self.output_path = output_path
        self.plot_spec_results = plot_spec_results

        weight_path = os.path.join(model_path, "weights.pth")
        config_path = os.path.join(model_path, "config.json")

        self.threshold=threshold
        self.min_length=min_length
        self.pad_song=pad_song

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(weight_path=weight_path, config_path=config_path)
        self.model = model.to(self.device)


        # make sure output dir exists 
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if plot_spec_results:
            if not os.path.exists(self.output_path):
                os.makedirs(os.path.join(self.output_path, 'specs'))
        
        self.database = pd.DataFrame(columns=["song_name", "directory", "song_status", "song_probabilties", "song_ms", "total_song_length"])

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
        
    def sort_all_songs(self):
        wav_to_spec = WavtoSpec()
        rows_to_add = []  # Initialize an empty list to collect rows

        # Load existing database if it exists to resume sorting
        output_csv_path = os.path.join(self.output_path, 'database.csv')
        if os.path.exists(output_csv_path):
            self.database = pd.read_csv(output_csv_path)
        else:
            self.database = pd.DataFrame(columns=["song_name", "directory", "song_status", "song_probabilties", "song_ms", "total_song_length"])

        total_songs = 0
        for bird in os.listdir(self.input_path):
            bird_path = os.path.join(self.input_path, bird)
            if os.path.isdir(bird_path):
                for day in os.listdir(bird_path):
                    day_path = os.path.join(bird_path, day)
                    if os.path.isdir(day_path):
                        total_songs += len([song for song in os.listdir(day_path) if os.path.isfile(os.path.join(day_path, song))])

        song_progress = tqdm(total=total_songs, desc="Sorting songs")
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
                        spec = wav_to_spec.process_file(song_src_path)
                        if spec is not None:
                            sample_rate, wavfile_signal = wavfile.read(song_src_path)

                            spec_mean = spec.mean()
                            spec_std = spec.std()
                            spec = (spec - spec_mean) / spec_std

                            predictions = post_processing.process_spectrogram(model=self.model, spec=spec, device=self.device, max_length=2048)
                            smoothed_song = post_processing.moving_average(predictions, window_size=100)
                            processed_song = post_processing.post_process_segments(smoothed_song, min_length=self.min_length, pad_song=self.pad_song, threshold=self.threshold)

                            song_name = Path(song).stem
                            if self.plot_spec_results:
                                post_processing.plot_spectrogram_with_processed_song(file_name=song_name, spectrogram=spec, smoothed_song=smoothed_song, processed_song=processed_song, directory=os.path.join(self.output_path, 'specs'))

                            song_status = np.where(processed_song > self.threshold, 1, 0)

                            # Calculate the duration of each timebin based on the wav file
                            wav_length_ms = (len(wavfile_signal) / sample_rate) * 1000  # Total duration of the wav file in milliseconds
                            timebin_duration_ms = wav_length_ms / len(song_status)  # Duration of each timebin in milliseconds

                            # Initialize song_ms as an empty list
                            song_ms = []
                            start_index = None  # Use index to track the start of a song segment

                            for index, status in enumerate(song_status):
                                if status == 1 and start_index is None:
                                    # A new song segment starts
                                    start_index = index  # Note the start index of the segment
                                elif status == 0 and start_index is not None:
                                    # The current song segment ends
                                    end_index = index  # Note the end index of the segment
                                    start_ms = start_index * timebin_duration_ms  # Calculate start ms of the segment
                                    end_ms = end_index * timebin_duration_ms  # Calculate end ms of the segment
                                    song_ms.append((start_ms, end_ms))  # Append the start and end ms as a tuple
                                    start_index = None  # Reset start_index for the next segment

                            # Check for a segment that might end at the last index
                            if start_index is not None:
                                end_index = len(song_status)  # The end index is the length of the song_status array
                                start_ms = start_index * timebin_duration_ms  # Calculate start ms of the last segment
                                end_ms = end_index * timebin_duration_ms  # Calculate end ms of the last segment
                                song_ms.append((start_ms, end_ms))  # Append the start and end ms as a tuple

                            song_probabilties = processed_song
                            total_song_length = np.sum(song_status) * timebin_duration_ms

                            new_row = {"song_name": song_name, "directory": song_src_path, "song_status": song_status.tolist(), "song_probabilties": song_probabilties.tolist(), "song_ms": song_ms, "total_song_length": total_song_length}
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
        if rows_to_add:
            self.database = pd.concat([self.database, pd.DataFrame(rows_to_add)], ignore_index=True)
            self.database.to_csv(output_csv_path, index=False)
            print(f"Database saved to {output_csv_path}")

    def sort_single_song(self, song_path):
        wav_to_spec = WavtoSpec()
        song_name = Path(song_path).stem 


        spec = wav_to_spec.process_file(song_path)
        if spec is None:
            print(f"Skipping {song_name}, unable to generate spectrogram.")
            return

        sample_rate, wavfile_signal = wavfile.read(song_path)
        spec_mean = spec.mean()
        spec_std = spec.std()
        spec = (spec - spec_mean) / spec_std

        predictions = post_processing.process_spectrogram(model=self.model, spec=spec, device=self.device, max_length=2048)
        smoothed_song = post_processing.moving_average(predictions, window_size=100)
        processed_song = post_processing.post_process_segments(smoothed_song, min_length=self.min_length, pad_song=self.pad_song, threshold=self.threshold)

        song_ms = [index * (1000 / sample_rate) for index in processed_song]
        new_row = {"song_name": song_name, "directory": song_path, "song_timebins": processed_song, "song_ms": song_ms}

        # Check if the song is already in the database and update it, otherwise append a new row
        if song_name in self.database['song_name'].values:
            self.database.loc[self.database['song_name'] == song_name, ['directory', 'song_timebins', 'song_ms']] = [song_path, processed_song, song_ms]
        else:
            self.database = pd.concat([self.database, pd.DataFrame([new_row])], ignore_index=True)

    def visualize_single_spec(self, song_path):
        self.sort_single_song(song_path)  # Process the song to get the necessary data

        song_name = os.path.basename(song_path).split(".")[0]
        spec_row = self.database.loc[self.database['song_name'] == song_name].iloc[0]
        spectrogram = spec_row['spectrogram']  # Assuming 'spectrogram' is stored in the database
        smoothed_song = spec_row['smoothed_song']  # Assuming 'smoothed_song' is stored in the database
        processed_song = spec_row['processed_song']  # Assuming 'processed_song' is stored in the database

        # Visualize the spectrogram without saving the image
        post_processing.plot_spectrogram_with_processed_song(directory=None, file_name=song_name, spectrogram=spectrogram, smoothed_song=smoothed_song, processed_song=processed_song)