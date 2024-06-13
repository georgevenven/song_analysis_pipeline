import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
from scipy.signal import windows, spectrogram, ellip, filtfilt
import shutil
from pathlib import Path
from scipy.signal import resample
import itertools
import csv
import sys
import multiprocessing
import soundfile as sf
import wave
import gc
from memory_profiler import profile
import psutil

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Memory usage in MB

class WavtoSpec:
    def __init__(self, src_dir, dst_dir, csv_file_dir=None):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.csv_file_dir = csv_file_dir
        self.use_csv = csv_file_dir is not None

    def process_file(self, file_path, song_info):
        self.convert_to_spectrogram(file_path, song_info)

    def process_directory(self):
        if self.use_csv:
            song_info = self.read_csv_file(self.csv_file_dir)  # Read the CSV file and store song information
        else:
            song_info = {}  # Empty dictionary when not using CSV file

        # Get the list of audio files
        audio_files = []
        for root, dirs, files in os.walk(self.src_dir):
            dirs[:] = [d for d in dirs if d not in ['.DS_Store']]  # Ignore irrelevant directories
            audio_files.extend([os.path.join(root, file) for file in files if file.lower().endswith('.wav')])

        # Create a pool of worker processes
        num_processes = multiprocessing.cpu_count()  # Use the number of available CPU cores
        # pool = multiprocessing.Pool(processes=num_processes)
        pool = multiprocessing.Pool(processes=1)

        # Create a progress bar
        progress_bar = tqdm(total=len(audio_files), unit='file', desc='Processing files')

        # Distribute the work among the processes
        results = []
        for file_path in audio_files:
            result = pool.apply_async(self.process_file, args=(file_path, song_info))
            results.append(result)

        # Update the progress bar as files are processed
        for result in results:
            result.get()
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        # Close the pool
        pool.close()
        pool.join()

    def read_csv_file(self, csv_file):
        song_info = {}
        csv.field_size_limit(sys.maxsize)  # Increase the field size limit
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                song_name = row['song_name']
                song_ms = eval(row['song_ms'])  # Evaluate the string to get the list of tuples
                song_info[song_name] = song_ms
        return song_info

    # def convert_to_spectrogram(self, file_path, song_info):
    #     offset_constant = .001

    #     try:
    #         with sf.SoundFile(file_path, 'r') as wav_file:
    #             samplerate = wav_file.samplerate
    #             num_channels = wav_file.channels

    #             # Read the entire audio data
    #             data = wav_file.read(dtype='int16')
    #             if num_channels > 1:
    #                 data = data[:, 0]  # Use the first channel if multi-channel audio

    #             song_name = os.path.splitext(os.path.basename(file_path))[0]

    #             # Define spectrogram parameters
    #             NFFT = 1024  
    #             step_size = 128
    #             overlap_samples = NFFT - step_size

    #             segments_to_process = self.get_segments_to_process(song_name, song_info, data, samplerate)

    #             for start_sample, end_sample in segments_to_process:
    #                 segment_data = data[start_sample:end_sample]
    #                 # Generate and process spectrogram for the segment
    #                 f, t, Sxx = spectrogram(segment_data, fs=samplerate, nperseg=NFFT, noverlap=overlap_samples)
    #                 Sxx_log = 10 * np.log10(Sxx + offset_constant)
    #                 Sxx_z_scored = self.z_score_spectrogram(Sxx_log)

    #                 # Save the entire spectrogram
    #                 self.save_spectrogram(song_name, Sxx_z_scored, start_sample, samplerate)

    #                 # Delete intermediate arrays
    #                 del segment_data, Sxx, Sxx_log, Sxx_z_scored

    #             # Delete data array
    #             del data

    #     except Exception as e:
    #         print(f"Error processing {file_path}: {e}")

    #     finally:
    #         plt.close('all')  # Ensure all plots are closed
    #         gc.collect()  # Force garbage collection

    #### Yarden method, mine is commented out
    def convert_to_spectrogram(self, file_path, song_info, min_length_ms=1000, whisper_seg_pre_processing=True):
        try:
            # Read the WAV file
            with sf.SoundFile(file_path, 'r') as wav_file:
                samplerate = wav_file.samplerate
                data = wav_file.read(dtype='int16')
                num_channels = wav_file.channels

            if num_channels > 1:
                data = data[:, 0]  # Use the first channel if multi-channel audio

            # Calculate the length of the audio file in milliseconds
            length_in_ms = (data.shape[0] / samplerate) * 1000
            if length_in_ms < min_length_ms:
                print(f"File {file_path} is below the length threshold and will be skipped.")
                return  # Skip processing this file

            # High-pass filter (adjust the filtering frequency as necessary)
            b, a = ellip(5, 0.2, 40, 500/(samplerate/2), 'high')
            data = filtfilt(b, a, data)

            # Canary song analysis parameters
            NFFT = 1024  # Number of points in FFT
            step_size = 119  # Step size for overlap

            # Calculate the overlap in samples
            overlap_samples = NFFT - step_size

            # Use a Gaussian window
            window = windows.gaussian(NFFT, std=NFFT/8)

            # Get the segments to process based on song_info
            segments_to_process = self.get_segments_to_process(os.path.basename(file_path), song_info, data, samplerate)

            for start_sample, end_sample in segments_to_process:
                segment_data = data[start_sample:end_sample]

                # Compute the spectrogram with the Gaussian window
                f, t, Sxx = spectrogram(segment_data, fs=samplerate, window=window, nperseg=NFFT, noverlap=overlap_samples)

                # Convert to dB
                Sxx_log = 10 * np.log10(Sxx + 1e-6)

                # Post-processing: Clipping and Normalization
                clipping_level = -2  # dB
                Sxx_log_clipped = np.clip(Sxx_log, a_min=clipping_level, a_max=None)
                Sxx_log_normalized = (Sxx_log_clipped - np.min(Sxx_log_clipped)) / (np.max(Sxx_log_clipped) - np.min(Sxx_log_clipped))

                # Assuming label is an integer or float
                labels = np.full((Sxx_log_normalized.shape[1],), 0)  # Adjust the label array as needed

                # Define the path where the spectrogram will be saved
                spec_filename = os.path.splitext(os.path.basename(file_path))[0]
                spec_file_path = os.path.join(self.dst_dir, spec_filename + '_' + str(start_sample) + '.npz')

                # Saving the spectrogram and the labels
                np.savez_compressed(spec_file_path, s=Sxx_log_normalized, labels=labels)

                # Print out the path to the saved file
                print(f"Spectrogram saved to {spec_file_path}")
                # Delete data array
                del Sxx_log_normalized


        except ValueError as e:
            print(f"Error reading {file_path}: {e}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        finally:
            plt.close('all')  # Ensure all plots are closed
            gc.collect()  # Force garbage collection


    def save_spectrogram(self, song_name, Sxx_z_scored, start_sample, samplerate):
        segment_filename = f"{song_name}_{int(start_sample / samplerate * 1000)}.npz"
        segment_file_path = os.path.join(self.dst_dir, segment_filename)
        np.savez_compressed(segment_file_path, s=Sxx_z_scored)

    #@profile
    def get_segments_to_process(self, song_name, song_info, data, samplerate):
        segments_to_process = []
        if self.use_csv and song_name in song_info:
            song_segments = song_info[song_name]
            print(song_segments)
            for start_ms, end_ms in song_segments:
                start_sample = int(start_ms * samplerate / 1000)
                end_sample = int(end_ms * samplerate / 1000)
                segments_to_process.append((start_sample, end_sample))
        else:
            # If not using CSV, or song not found, process the entire file as one segment
            segments_to_process.append((0, len(data)))
        return segments_to_process

    def z_score_spectrogram(self, Sxx_log):
        mean = Sxx_log.mean()
        std = Sxx_log.std()
        Sxx_z_scored = (Sxx_log - mean) / std
        return Sxx_z_scored


    def visualize_random_spectrogram(self):
        # Get a list of all '.npz' files in the destination directory
        npz_files = list(Path(self.dst_dir).glob('*.npz'))
        if not npz_files:
            print("No spectrograms available to visualize.")
            return
        
        # Choose a random spectrogram file
        random_spec_path = random.choice(npz_files)
        
        # Load the spectrogram data from the randomly chosen file
        with np.load(random_spec_path) as data:
            spectrogram_data = data['s']
        
        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_data, aspect='auto', origin='lower')
        plt.title(f"Random Spectrogram: {random_spec_path.stem}")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(format='%+2.0f Z Scores')
        plt.show()

    def plot_grid_of_spectrograms(self, min_length=100):
        # Get a list of all '.npz' files in the destination directory
        npz_files = list(Path(self.dst_dir).glob('*.npz'))
        if len(npz_files) < 25:
            print("Not enough spectrograms available to create a 5x5 grid. Please ensure there are at least 25 spectrograms in the destination directory.")
            return

        selected_spec_paths = []
        attempted_paths = set()

        # Keep selecting random spectrograms until we have 25 with enough time bins
        while len(selected_spec_paths) < 25:
            if len(attempted_paths) >= len(npz_files):
                print("Tried all available spectrograms, but could not find enough with sufficient time bins.")
                return
            spec_path = random.choice(npz_files)
            if spec_path in attempted_paths:
                continue  # Skip if this spectrogram has already been attempted
            attempted_paths.add(spec_path)

            with np.load(spec_path) as data:
                spectrogram_data = data['s']
                if spectrogram_data.shape[1] > min_length:  # Check if it has more than min_length time bins
                    selected_spec_paths.append(spec_path)  # Add to the list if it meets the criteria

        # Set up the subplot grid
        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
        fig.suptitle('5 x 5 Grid of Random Spectrograms', fontsize=16)

        for ax, spec_path in zip(axes.flatten(), selected_spec_paths):
            with np.load(spec_path) as data:
                spectrogram_data = data['s']
                # Take the first set of min_length bins
                spectrogram_data = spectrogram_data[:, :min_length]

                # Plot the spectrogram on its respective subplot
                img = ax.imshow(spectrogram_data, aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(spec_path.stem, fontsize=8)
                ax.axis('off')  # Hide axes for better visualization

                # Create a color bar for the current subplot
                fig.colorbar(img, ax=ax, format='%+2.0f dB')

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def compare_spectrogram_permutations(self, param_dict):
        """
        Generates and plots spectrograms for a single random .wav file using permutations of NFFT and step_size parameters.
        Only the first 1000 time bins of each spectrogram are plotted, with the row parameters (NFFT and step size)
        displayed on the left-hand side of each row.

        Parameters:
        param_dict (dict): A dictionary with 'NFFT' and 'step_size' as keys and lists of their possible values.
        """
        # Randomly select 1 .wav file
        wav_files = [f for f in Path(self.src_dir).glob('**/*.wav')]
        selected_file = random.choice(wav_files)

        # Generate all permutations of NFFT and step_size
        permutations = list(itertools.product(param_dict['NFFT'], param_dict['step_size']))

        # Set up the plot
        fig, axs = plt.subplots(len(permutations), 1, figsize=(10, 5 * len(permutations)), squeeze=False)
        
        for i, (nfft, step_size) in enumerate(permutations):
            # Generate the spectrogram for the current permutation and file
            spectrogram_data = self.generate_spectrogram_data(str(selected_file), nfft, step_size)
            spectrogram_data = spectrogram_data[:, :1000]  # Select the first 1000 time bins

            # Plot the spectrogram
            ax = axs[i, 0]
            img = ax.imshow(spectrogram_data, aspect='auto', origin='lower', cmap='viridis')
            ax.set_ylabel(f'NFFT: {nfft}\nStep Size: {step_size}')
            ax.set_xlabel('Time Bins')
            ax.axis('on')

            # Adding a colorbar to each subplot
            fig.colorbar(img, ax=ax)

        plt.tight_layout()
        plt.show()

    def generate_spectrogram_data(self, file_path, nfft, step_size, offset_constant=1):
        """
        Generates spectrogram data for a given .wav file, NFFT, and step size.

        Parameters:
        file_path (str): Path to the .wav file.
        nfft (int): Number of FFT points.
        step_size (int): Step size for the FFT overlap.

        Returns:
        ndarray: The spectrogram data.
        """
        # This is a simplified version of convert_to_spectrogram method
        samplerate, data = wavfile.read(file_path)
        if len(data.shape) > 1:
            data = data[:, 1]  # Select the right channel if stereo

        # # Use a Gaussian window
        # window = windows.gaussian(nfft, std=nfft/8)

        # Calculate the overlap in samples
        overlap_samples = nfft - step_size

        # Compute the spectrogram with the Gaussian window
        f, t, Sxx = spectrogram(data, fs=samplerate, nperseg=nfft, noverlap=overlap_samples)

        # Convert to dB
        Sxx_log = 10 * np.log10(Sxx + offset_constant)

        mean = Sxx_log.mean()
        std = Sxx_log.std()
        Sxx_z_scored = (Sxx_log - mean) / std

        return Sxx_z_scored

# Helper function to find the next power of two
def nextpow2(x):
    return np.ceil(np.log2(np.abs(x))).astype('int')

def copy_yarden_data(src_dirs, dst_dir):
    """
    Copies all .npz files from a list of source directories to a destination directory.

    Parameters:
    src_dirs (list): A list of source directories to search for .npz files.
    dst_dir (str): The destination directory where .npz files will be copied.
    """
    # Ensure the destination directory exists
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    # Create a list to store all the .npz files found
    npz_files = []

    # Find all .npz files in source directories
    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.npz'):
                    npz_files.append((os.path.join(root, file), file))

    # Copy the .npz files to the destination directory with progress bar
    for src_file_path, file in tqdm(npz_files, desc='Copying files'):
        dst_file_path = os.path.join(dst_dir, file)
        
        # Ensure we don't overwrite files in the destination
        if os.path.exists(dst_file_path):
            print(f"File {file} already exists in destination. Skipping copy.")
            continue

        # Copy the .npz file to the destination directory
        shutil.copy2(src_file_path, dst_file_path)
        print(f"Copied {file} to {dst_dir}")
