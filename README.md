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
    
edit the following functions:
the dataloader function 'load_data' returns the pytorch Dataloader objects for the training and test data.

   
