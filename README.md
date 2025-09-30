# Weather Forecasting with Graph Neural Networks

A deep learning project that leverages Graph Neural Networks (GNNs) for weather forecasting using ERA5 climate data. This project implements both traditional and advanced graph-based models to predict weather patterns across a global grid.

## Project Overview

This project uses spatio-temporal graph neural networks to perform weather forecasting on ERA5 reanalysis data. The model treats the global weather system as a graph where each grid point is a node connected to its neighbors, enabling the capture of complex spatial and temporal dependencies in weather patterns.

### Features

- Processes ERA5 climate data variables including:
  - Geopotential at 500hPa
  - Temperature at 850hPa
  - 2m Temperature
  - 10m U-component of wind
  - 10m V-component of wind
- Implements both traditional and advanced graph-based models
- Supports different graph connectivity patterns (strong/cartesian)
- Configurable input sequence length and forecast horizon
- Train/validation/test split by time periods

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
    ```bash
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

### Evaluation

To evaluate the model:

```bash
python scripts/evaluate.py
```

### Configuration

Model and training parameters can be configured in `utils/default_config.yaml`. Key configuration options include:

- Input sequence length
- Forecast horizon
- Graph neighborhood size
- Model architecture parameters
- Training hyperparameters

## Project Structure

```
├── data/               # Data processing modules
│   ├── dataloader.py   # PyTorch data loaders
│   ├── dataset.py      # Dataset classes
│   └── transforms.py   # Data transformations
├── models/             # Model implementations
│   ├── advanced.py     # Advanced graph neural networks
│   └── traditional.py  # Traditional approaches
├── scripts/            # Training and evaluation scripts
├── utils/              # Utility functions and configs
└── notebooks/          # Jupyter notebooks for exploration
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.