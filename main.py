# Sort Song --> Train TweetyBERT --> HDBSCAN --> Manual Overview --> Merge / Identify Noise Clusters --> Train Classifier --> Analyze 
# This script executes for each individual singer / experimental condition 

import json

# Load JSON parameters from 'parameters.json'
with open('parameters.json', 'r') as file:
    parameters = json.load(file)

if parameters["sort_songs"]:
    print("Sorting songs...")
    
else:
    print("Songs are assumed to be sorted as per config...")
