# Sort Song --> Train TweetyBERT --> HDBSCAN --> Manual Overview --> Merge / Identify Noise Clusters --> Train Classifier --> Analyze 
# This script executes for each individual singer / experimental condition 
# When done, if you want to do mass sorting, create multiple parameters.json ... sus out the file structure 

import json
import os 
import torch 

from src.tweety_net_detector.inference import Inference 
from src.tweety_bert.spectogram_generator import WavtoSpec
from src.tweety_bert.experiment_manager import ExperimentRunner
from src.tweety_bert.analysis import plot_umap_projection 
from src.tweety_bert.utils import load_model
from src.tweety_bert.linear_probe import LinearProbeModel, LinearProbeTrainer, ModelEvaluator
from scripts.test_train_split import split_dataset

# Load JSON parameters from 'parameters.json'
with open('parameters.json', 'r') as file:
    parameters = json.load(file)

print(parameters["input_path"]+"database.csv")

if parameters["sort_songs"]:
    # [FUTURE FEATURE] In the future make model path changable, although low priority for now 
    sorter = Inference(input_path = parameters["input_path"], output_path = parameters["temp_path"], plot_spec_results=False, model_path="models/song_detector/sorter-1")
    sorter.sort_all_songs()
    # [INEFFICENCY] The output of this is still technically wav files, even though the process does create spectograms and destroy them, which is an efficency that should be squashed in the future 

###################### Future FEATURE ###################### 
# else:
#     # if song are already sorted...
#     pass

#     # convert to spectogram

#     # already spectogram
############################################################ 


# # generate specs into /temp
generated_sorted_specs_directory = os.path.join(parameters["temp_path"], "sorted_specs")
if not os.path.exists(generated_sorted_specs_directory):
    os.makedirs(generated_sorted_specs_directory)

wav_to_spec = WavtoSpec(parameters["input_path"], generated_sorted_specs_directory, csv_file_dir="/media/george-vengrovski/disk1/song_analysis_pipeline_testing_delete_when_works/temp/database.csv")
wav_to_spec.process_directory()

# # train test split
# Example usage with moving files
# create directories for train and test 
train_dir = os.path.join(parameters["temp_path"], "train")
test_dir = os.path.join(parameters["temp_path"], "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

split_dataset(generated_sorted_specs_directory, 0.8, train_dir, test_dir, move_files=False)

# # load tweety bert and train 
config_path = "/home/george-vengrovski/Documents/projects/song_analysis_pipeline/models/tweety_bert/default_config/config.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
experiment_runner = ExperimentRunner(device="cuda", base_save_dir=parameters["temp_path"]+"/"+parameters["bird_id"]+"pretrain_run")

# load config json 
with open(config_path, 'r') as file:
    config = json.load(file)

# add config keys 
config["train_dir"] = train_dir
config["test_dir"] = test_dir

model = experiment_runner.run_experiment(config, "")

#TweetyBERT 128 OG Model 
plot_umap_projection(
model=model, 
device=device, 
data_dir=generated_sorted_specs_directory,
samples=parameters["umap_params"]["samples"], 
file_path=parameters["umap_params"]["file_path"], 
layer_index=parameters["umap_params"]["layer_index"], 
dict_key=parameters["umap_params"]["dict_key"], 
context=parameters["umap_params"]["context"], 
raw_spectogram=parameters["umap_params"]["raw_spectogram"],
save_dict_for_analysis = parameters["umap_params"]["save_dict_for_analysis"],
save_name=parameters["umap_params"]["save_name"],
)

# Join Clusters
print("10 Examples of each cluster has been saved to /temp/joined_clusters, please inspect and merge if needed")
