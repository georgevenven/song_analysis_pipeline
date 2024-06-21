import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import random
import matplotlib.pyplot as plt
import logging

class SongDataSet_Image(Dataset):
    def __init__(self, file_dir, num_classes=100, infinite_loader=True, segment_length=1000, decoder=False):
        self.file_paths = [os.path.join(file_dir, file) for file in os.listdir(file_dir)]
        self.num_classes = num_classes
        self.infinite_loader = infinite_loader
        self.segment_length = segment_length
        self.decoder = decoder

    def __getitem__(self, idx):
        if self.infinite_loader:
            original_idx = idx  # Store the original index for logging
            idx = random.randint(0, len(self.file_paths) - 1)
        file_path = self.file_paths[idx]

        try:
            # Load data and preprocess
            data = np.load(file_path, allow_pickle=True)
            spectogram = data['s']

            # if file contains any NaN, replace with 0 
            spectogram = np.nan_to_num(spectogram, nan=0)

            # Calculate mean and standard deviation only for non-zero (non-NaN replaced) elements
            valid_elements = spectogram[spectogram != 0]
            spec_mean = np.mean(valid_elements) if valid_elements.size > 0 else 0
            spec_std = np.std(valid_elements) if valid_elements.size > 0 else 1

            # Z-score the spectrogram, avoiding division by zero
            spectogram = (spectogram - spec_mean) / spec_std if spec_std > 0 else spectogram - spec_mean

            if self.decoder == False:
                spectogram = spectogram[20:216]
                ground_truth_labels = np.zeros(spectogram.shape[1])
                # Calculate mean and standard deviation of the spectrogram
            else:
                spectogram = spectogram.T
                ground_truth_labels = np.array(data['labels'], dtype=int)
            # Process labels

            ground_truth_labels = torch.from_numpy(ground_truth_labels).long().squeeze(0)
            spectogram = torch.from_numpy(spectogram).float().permute(1, 0)
            ground_truth_labels = F.one_hot(ground_truth_labels, num_classes=self.num_classes).float()

            # Truncate if larger than context window
            if spectogram.shape[0] > self.segment_length:
                # Get random view of size segment
                # Find range of valid starting pts (essentially these are the possible starting pts for the length to equal segment window)
                starting_points_range = spectogram.shape[0] - self.segment_length        
                start = torch.randint(0, starting_points_range, (1,)).item()  
                end = start + self.segment_length     

                spectogram = spectogram[start:end]
                ground_truth_labels = ground_truth_labels[start:end]

            # Pad with 0s if shorter
            if spectogram.shape[0] < self.segment_length:
                pad_amount = self.segment_length - spectogram.shape[0]
                spectogram = F.pad(spectogram, (0, 0, 0, pad_amount), 'constant', 0)
                ground_truth_labels = F.pad(ground_truth_labels, (0, 0, 0, pad_amount), 'constant', 0)  # Adjusted padding for labels

            return spectogram, ground_truth_labels

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            # Recursively call __getitem__ with a different index if in infinite loader mode
            if self.infinite_loader:
                return self.__getitem__(random.randint(0, len(self.file_paths) - 1))
            else:
                raise e
    
    def __len__(self):
        if self.infinite_loader:
            # Return an arbitrarily large number to simulate an infinite dataset
            return int(1e12)
        else:
            return len(self.file_paths)

class CollateFunction:
    def __init__(self, segment_length=1000):
        pass
    def __call__(self, batch):
        # Unzip the batch (a list of (spectogram, ground_truth_labels) tuples)
        spectograms, ground_truth_labels = zip(*batch)

        # Stack tensors along a new dimension to match the BERT input size.
        spectograms = torch.stack(spectograms, dim=0)
        ground_truth_labels = torch.stack(ground_truth_labels, dim=0)

        # Final reshape for model
        spectograms = spectograms.unsqueeze(1).permute(0,1,3,2)

        return spectograms, ground_truth_labels