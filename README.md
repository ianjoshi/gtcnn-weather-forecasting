# Weather Forecasting with Graph Neural Networks

A deep learning project that leverages Graph Neural Networks (GNNs) for weather forecasting using ERA5 climate data. This project implements both traditional and advanced graph-based models to predict weather patterns across a global grid.

## Project Overview

This project uses spatio-temporal graph neural networks to perform weather forecasting on ERA5 reanalysis data. The model treats the global weather system as a graph where each grid point is a node connected to its neighbors, enabling the capture of complex spatial and temporal dependencies in weather patterns.

### Features

- Processes ERA5 climate data variables
- Multiple model implementations (including non-graph models, non-neural graph models and graph models)
- Supports different graph connectivity patterns (e.g., strong/cartesian)
- Configurable input sequence length and forecast horizon via config files
- Train/validation/test splits are defined in the configuration files

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (for faster training)
- conda (for environment management)

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/ianjoshi/graph-ml-g12.git
    cd graph-ml-g12
    ```

2. Create and activate the conda environment:
    ```powershell
    conda env create -f utils/environment.yml
    conda activate weather-cast
    ```

## Data

The project uses ERA5 reanalysis data at 5.625° resolution. The dataset includes the following variables:
- Geopotential at 500hPa
- Temperature at 850hPa
- 2m Temperature
- 10m U-component of wind
- 10m V-component of wind

The data is split into the following time periods:
- Training: 2005-01-01 to 2014-12-31
- Validation: 2015-01-01 to 2016-12-31
- Testing: 2017-01-01 to 2018-12-31

## Usage

### Training

To train the model:

```bash
python scripts/train.py 
```
You can also input the model type. Default is set to GTCNN.

### Evaluation

To evaluate the model:

```bash
python scripts/evaluate.py 
```
You can also input the model type and the checkpoint. Default is set to GTCNN.

### Configuration

Model and training parameters are split across the YAML files in `utils/`:

- `utils/base_config.yaml` — base dataset and training defaults (paths, time splits, logging)
- `utils/model_config.yaml` — model-specific settings (hidden dimensions, layers, dropout, model type)
- `utils/environment.yml` — conda environment used for development and reproducibility

Key options you will commonly change:

- Input sequence length and forecast_horizon
- Neighborhood / graph construction parameters
- Model architecture parameters (hidden dimension, layers, dropout)
- Training hyperparameters (batch size, learning rate, epochs)

## Project Structure (updated)

```
├── data/                # Data processing modules
│   ├── dataloader.py    # PyTorch data loaders 
│   ├── dataset.py       # Dataset classes 
│   └── transforms.py    # Data transforms and prepocessing
├── models/              # Model implementations
│   ├── cnn3d.py         # 3D-CNN baseline
│   └── gtcnn.py         # Graph-temporal CNN (GTCNN)
├── scripts/             # Training and evaluation CLI 
│   ├── train.py
│   └── evaluate.py
├── utils/               # Configuration and environment 
│   ├── base_config.yaml
│   ├── model_config.yaml
│   └── environment.yml
├── checkpoints/         # Trained model checkpoints
└── reports/             # Experiment reports / logs
```

## Checkpoints and Reports

THe training script saves the best model under `./checkpoints` and the evaluation script outputs the performance reports under `./reports` summarising results for the model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.