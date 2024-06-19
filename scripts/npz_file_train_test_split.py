import os
import numpy as np

def split_npz_into_chunks(npz_file, chunk_size, train_folder, test_folder, train_ratio=0.8):
    # Create train and test directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Load the npz file
    data = np.load(npz_file, allow_pickle=True)
    
    length = data["hdbscan_labels"].shape[0]
    num_chunks = length // chunk_size
    for i in range(0, num_chunks):
        # if random chance above .8 than test, else train
        if np.random.random() < train_ratio:
            # go through all the keys, save the chunk to the train folder
            chunk_data = {}
            for key in data.files:
                if data[key].ndim == 0:
                    chunk_data[key] = data[key]
                else:
                    chunk_data[key] = data[key][i*chunk_size:(i+1)*chunk_size]
            np.savez(os.path.join(train_folder, f"chunk_{i}.npz"), **chunk_data)
        else:
            chunk_data = {}
            for key in data.files:
                if data[key].ndim == 0:
                    chunk_data[key] = data[key]
                else:
                    chunk_data[key] = data[key][i*chunk_size:(i+1)*chunk_size]

            np.savez(os.path.join(test_folder, f"chunk_{i}.npz"), **chunk_data)
