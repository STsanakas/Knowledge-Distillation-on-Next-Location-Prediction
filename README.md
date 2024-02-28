# Knowledge-Distillation-on-Next-Location-Prediction

Installation/Usage

 - In your virtual environment run ```pip install -r requirements.txt```
 - Assumes the following structure
```bash

Data_path_declared_in_geolife_data_utilities
├── Geolife
│   └── Geolife_datasets
│   └── Geolife_models
```
 - The datasets should be ```.npy``` files
 - The input dataset should be named ```<insert_name_here>X.npy``` and of shape ```(x,15,2)``` where ```x``` is the number of samples. For each sample the input consists of ```15 timesteps``` of ```distance``` (in km) and ```bearing``` (in degrees).
 - Likewise, the output dataset should be named ```<insert_name_here>Y.npy``` and of shape ```(x,2)``` where ```x``` is the number of samples. For each sample the input consists of the output ```distance``` (in km) and ```bearing``` (in degrees).
 - For the experiments in ```demo_script_run.py```, the data was split into 4 areas (```NE```, ```NW```, ```SE```, ```SW```) and 6 transportation means (```bike```, ```bus```, ```car```, ```subway```, ```taxi```, ```train```, ```walk```). So for every script to work, the dataset files expected are ```NE_bikeX.npy```, ```NE_bikeY.npy``` etc.
