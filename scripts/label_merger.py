import numpy as np

def read_label_file(file_path):
    combine = []
    noise = []
    mode = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if 'Combine:' in line:
                mode = 'combine'
            elif 'Mark as Noise:' in line:
                mode = 'noise'
            elif line:
                if mode == 'combine':
                    combine.append(list(map(int, line.split(','))))
                elif mode == 'noise':
                    noise.extend(list(map(int, line.split(','))))
    return combine, noise

def process_labels(npz_path, combine, noise):
    with np.load(npz_path, allow_pickle=True) as data:
        original_hdb_scan_labels = data['hdbscan_labels'].copy()
        hdb_scan_labels = original_hdb_scan_labels.copy()
        other_data = {key: data[key] for key in data.files if key not in ['hdbscan_labels', 'ground_truth_labels']}
    
    # Process combine
    for group in combine:
        target_label = group[0]
        for label in group:
            hdb_scan_labels[hdb_scan_labels == label] = target_label
    
    # Process noise
    for label in noise:
        hdb_scan_labels[hdb_scan_labels == label] = -1
    
    # Save the original and modified labels along with other data back to the npz file
    np.savez(npz_path, hdbscan_labels=original_hdb_scan_labels, ground_truth_labels=hdb_scan_labels, **other_data)

