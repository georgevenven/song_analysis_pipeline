# Sort Song --> Train TweetyBERT --> HDBSCAN --> Manual Overview --> Merge / Identify Noise Clusters --> Train Classifier --> Analyze 

from scripts.npz_file_train_test_split import split_npz_into_chunks
from scripts.label_merger import read_label_file, process_labels



input_file = 'merge_combine.txt'  # Path to the input file
npz_file = 'files/labels_HDBSCAN_Classification.npz'  # Path to the npz file

combine, noise = read_label_file(input_file)
process_labels(npz_file, combine, noise)

# Train Non-Linear Classifier
# Example usage
npz_file = '/home/george-vengrovski/Documents/projects/song_analysis_pipeline/files/labels_HDBSCAN_Classification.npz'
train_folder = "/media/george-vengrovski/disk1/song_analysis_pipeline_testing_delete_when_works/temp/linear_classifier_train"
test_folder = "/media/george-vengrovski/disk1/song_analysis_pipeline_testing_delete_when_works/temp/linear_classifier_test"
split_npz_into_chunks(npz_file, chunk_size=1000, train_folder=train_folder, test_folder=test_folder)

# # Use it to analyze whole dataset 
# import shutil

# # delete temp folder contents
# temp_folder_path = parameters["temp_path"]
# shutil.rmtree(temp_folder_path)
# os.makedirs(temp_folder_path, exist_ok=True) 