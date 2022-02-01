# PyTorch Project Template

## Installation
### Install cuda
Follow instructions for cuda install: <br/>
(Windows) https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html <br/>
(Linux) https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

### Install python
Install python environment for example with conda:
```
  conda create -n torch python=3.9
  conda activate torch
```

### Install packages
Run install bash script to install packages.
```
  # run after installing python
  bash install.sh
```

## Project structure
    
    .
    ├── ...
    ├── src
        ├── dataloader                # Data files and functions for loading data
        ├── logger                    # Tracking and visualization of metrics
        ├── model                     # Model definition
        ├── trainer                   # Handler for model training and testing
        └── utils                     # Tools and utilities
    ├── train.py
    ├── test.py

## Customize project 
Edit the following functions: 

`dataloader:` Change the dataloader function __'load_data'__ to return pytorch Dataloader objects for the training and test data. <br/>

`model:` Define a model class in the model directory and edit the __train.py__ and __test.py__ files according to the model name. <br/>

`trainer:` Define the __'\_get_optimizer'__ and __'\_forward step'__ functions in the Trainer class which inherits from BaseTrainer. <br/>
They should return a pytorch optimizer class and a dictionary with a 'loss' entry and additional entries for metrics to be tracked. <br/>

   
