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

`model:` Define a model class in the model directory. <br/>

`trainer:` Define the __'\_get_optimizer'__ and __'\_forward step'__ functions in the Trainer class which inherits from BaseTrainer. <br/>
They should return a pytorch optimizer class and a dictionary with a 'loss' entry and additional entries for metrics to be tracked. <br/>

`train.py / test.py:` Import and load the model, add arguments if necessary.

## Usage

#### Training
The training functions will fit a model and track the specified metrics for each forward step (batch) and written to a Tensorboard event file during training. This file can be read with the provided utility functions and displayed using tensorboard.
The averages of the metrics are updated and displayed on the terminal during this process.

<img src="https://user-images.githubusercontent.com/27029923/151994353-d293f96e-5ad8-485d-adbb-a039fb33398f.png" width="100%" height="100%">

![image](https://user-images.githubusercontent.com/27029923/151994353-d293f96e-5ad8-485d-adbb-a039fb33398f.png "Training" {width=200px height=200px})

#### Tensorboard

<img src="https://user-images.githubusercontent.com/27029923/151995141-459071ce-7459-422d-ae76-b81f0c376e09.png" width="50%" height="50%">


If specified, model checkpoints are saved periodically. Training can be resumed by specifying the model state file path.

Model testing returns a dictionary with all metrics.


   
