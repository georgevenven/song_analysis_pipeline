import os
import numpy as np

def split_npz_into_chunks(npz_file, chunk_size, train_folder, test_folder, train_ratio=0.8):
    # Create train and test directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Load the npz file
    data = np.load(npz_file, allow_pickle=True)
    
    # Get the temporal dimension size from one of the arrays
    time_dim = None
    for key in data.files:
        if data[key].ndim > 0:
            time_dim = data[key].shape[-1]
            break

    if time_dim is None:
        raise ValueError("No valid temporal dimension found in the npz file.")

    # Calculate the number of chunks
    num_chunks = time_dim // chunk_size

    # Split the data into chunks
    chunks = {key: np.array_split(data[key], num_chunks, axis=-1) for key in data.files}

    # Shuffle the chunks
    indices = np.arange(num_chunks)
    np.random.shuffle(indices)

    # Calculate the number of train chunks
    train_size = int(num_chunks * train_ratio)

    # Split the indices into train and test
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Save the chunks into train and test folders
    for i, idx in enumerate(train_indices):
        chunk_data = {key: chunks[key][idx] for key in data.files}
        np.savez(os.path.join(train_folder, f"chunk_{i}.npz"), **chunk_data)

    for i, idx in enumerate(test_indices):
        chunk_data = {key: chunks[key][idx] for key in data.files}
        np.savez(os.path.join(test_folder, f"chunk_{i}.npz"), **chunk_data)


