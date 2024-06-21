from scripts.lasso_tool import lasso_tool
from scripts.inspecting_hdbscan_labels import plot_longest_segments_by_label
import json

# File paths
file_path = "/home/george-vengrovski/Documents/projects/song_analysis_pipeline/files/labels_HDBSCAN_Classification.npz"

# Load JSON parameters from 'parameters.json'
with open('parameters.json', 'r') as file:
    parameters = json.load(file)

if parameters["syllable_discovery"] == "HDBSCAN":
    output_file_path = "spectrogram_segments.png"
    # Call the function
    plot_longest_segments_by_label(file_path, output_file_path)
    print("10 Examples of each cluster has been saved to /temp/joined_clusters, please inspect and merge if needed")

if parameters["syllable_discovery"] == "lasso":
    output_file_path = "spectrogram_segments.png"
    lasso_tool(file_path, output_file_path)
    # plot_longest_segments_by_label(file_path, output_file_path)
