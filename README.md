# Weather Forecasting with Graph Neural Networks

A deep learning project that leverages Graph Neural Networks (GNNs) for weather forecasting using ERA5 climate data. This project implements both traditional and advanced graph-based models to predict weather patterns across a global grid. Investigating physics loss and scalability conscious graph ml implementations.

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
- Training: 2010-01-01 to 2014-12-31
- Validation: 2015-01-01 to 2016-12-31
- Testing: 2017-01-01 to 2018-12-31

## Usage

### Training

To train a model, use:

```bash
python scripts/train.py [--model_type {gtcnn,sign,cnn3d}] [--physics_loss] [--save_name NAME] [--assert_shapes]
```

**Train script options:**
- `--model_type`: Select the model architecture. Choices are:
  - `gtcnn` (default): Graph-temporal convolutional neural network
  - `sign`: Spatial-Identity Graph Neural Network
  - `cnn3d`: 3D convolutional neural network (grid baseline)
- `--physics_loss` (optional): Adds a physics-based kinetic energy loss term (only for GTCNN).
- `--save_name`: Custom name for the checkpoint and summary files (default: uses the model type).
- `--assert_shapes`: Enables checks for tensor shapes on the first batch (for debugging purposes).

Model and training hyperparameters (such as input length, layers, epochs, learning rate) should be set in the YAML configuration files under `utils/`. See the Configuration section below.

Example:
```bash
python scripts/train.py --model_type gtcnn --physics_loss --save_name my_run
```

---

### Evaluation

To evaluate a trained model or run a baseline, use:

```bash
python scripts/evaluate.py [--model_type {gtcnn,sign,cnn3d,persistence,climatology}] [--checkpoint PATH] [--model_category {graph_based,grid_based}] [--save_name REPORT_NAME]
```

**Evaluate script options:**
- `--model_type`: Select the evaluation model. Choices are:
  - `gtcnn`: Graph-temporal CNN (requires checkpoint)
  - `sign`: Identity-based GNN (requires checkpoint)
  - `cnn3d`: Grid-based 3D CNN (requires checkpoint)
  - `persistence`: Baseline—uses last input as prediction
  - `climatology`: Baseline—uses daily climatology
- `--checkpoint`: Path to the `.pt` model checkpoint. If not specified for a neural model, uses `checkpoints/best_<model>.pt`.
- `--model_category`: Dataset input format. Choices are:
  - `graph_based` (default): For GTCNN/SIGN models
  - `grid_based`: For CNN3D
- `--save_name`: Custom name for the output report (default: `report_{model_type}.txt` in `./reports/`).

Example:
```bash
python scripts/evaluate.py --model_type cnn3d --checkpoint ./checkpoints/best_cnn3d.pt --model_category grid_based --save_name my_cnn3d_eval
```

See the configuration files for model and data options under `utils/`. Detailed evaluation metrics are saved in `./reports/`.


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

## Project Structure

```
├── data/                      # Data processing package
│   ├── __init__.py
│   ├── dataloader.py          # Creates PyTorch DataLoaders and batching logic
│   ├── dataset.py             # ERA5Dataset: load NetCDF and create graph node features
│   └── transforms.py          # Normalization and spatial/temporal transforms
├── models/                    
│   ├── __init__.py            
│   ├── cnn3d.py               # 3D-CNN baseline (gridded input)
│   └── gtcnn.py               # GTCNN: graph-temporal convolutional model
├── scripts/                   
│   ├── train.py               # Training loop 
│   └── evaluate.py            # Evaluation and reporting utilities
├── checkpoints/               # Saved model weights (best model files)
├── training_summaries/        # Training summary text files
├── plotters/                  # Plotting helpers
│   └── metrics_plotter.py     
├── plots/                     # Generated figures 
├── reports/                   # Textual experiment summaries 
├── utils/                     
│   ├── base_config.yaml       # Dataset and training defaults
│   ├── model_config.yaml      # Model hyperparameters and architecture choices
│   └── environment.yml        # Conda environment for reproducibility
└── README.md                  # This file
```

## Checkpoints and Reports

THe training script saves the best model under `./checkpoints/` and a summary in `./training_summaries/`. The evaluation script outputs the performance reports under `./reports/` summarising results for the model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.