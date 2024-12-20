# WinCLIP: Refactored Implementation for Few-Shot Anomaly Detection

This repository contains a refactored implementation of the unofficial WinCLIP framework for zero-/few-shot anomaly classification and segmentation. The unofficial implementation is available at [MALA Lab's GitHub repository](https://github.com/mala-lab/WinCLIP). We have extended and optimized this implementation to work seamlessly with the AeBAD dataset and introduced improvements to simplify usability and reproducibility.

## Key Features
- **Refactored Code**: Enhanced readability and modularity for better integration and customization.
- **AeBAD Compatibility**: Added support for the AeBAD dataset through a dedicated conversion script.
- **Custom Configurations**: Easily adjustable parameters through a centralized configuration file.
- **Optimized Performance**: Tested on NVIDIA GeForce RTX 3060, ensuring efficiency in training and evaluation.

---

## Setup and Dependencies
The code requires Python 3.10+ and the following packages:

```bash
torch >= 1.13.0
torchvision >= 0.14.0
scipy >= 1.10.1
scikit-image >= 0.21.0
numpy >= 1.24.3
tqdm >= 4.64.0
```
Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Device
This implementation has been tested and benchmarked on a **Single NVIDIA GeForce RTX 3060**.

## Folder Structure
The repository is structured as follows:
```bash
WinCLIP/
├── datasets/
│   ├── preprocess/
│   │   ├── convert_aebad.py  # Converts AeBAD dataset into a compatible format
│   │   ├── convert_visa.py   # Converts Visa dataset
│   │   └── mvtec_dataset.py  # Dataset handling script
│   └── ...
├── open_clip/                # Pretrained model configurations and checkpoints
├── binary_focal_loss.py      # Custom loss function implementation
├── check_sizes.py            # Utility script for dataset inspection
├── config.toml               # Configurations for different datasets
├── find.py                   # Script for utility management
├── main.py                   # Original main script
├── main_refactored.py        # Refactored main script for experimentation
├── main_refactored_aebad.py  # Specific script for running experiments on AeBAD
├── main_refactored_optimization.py # Optimization-focused script
└── requirements.txt          # Required Python packages
```

## Dataset Setup: AeBAD

To use this implementation, you need to prepare the AeBAD dataset and convert it into a compatible format. Follow these steps:

### 1. Download the AeBAD Dataset
The AeBAD dataset must be downloaded and placed in a directory structure.

### 2. Configure the Conversion Script
Modify the `input_root` variable in `datasets/preprocess/convert_aebad.py` to point to the directory where your AeBAD dataset is stored. For example:

```python
input_root = "/path/to/your/aebad/dataset"
```
### 3. Convert the Dataset
Run the conversion script to transform the AeBAD dataset into a compatible format:
```python
python datasets/preprocess/convert_aebad.py
```
## Configuration

The main configurations for running experiments are stored in the `config.toml` file. This file allows users to customize parameters such as dataset paths, model configurations, and experimental settings.

Below is an example configuration for running experiments on the AeBAD dataset:

```toml
[aebad]
datasetname = "aebad"                        # Dataset name identifier
dataset_root_dir = "/path/to/WinCLIP/AeBAD"  # Root directory of the AeBAD dataset
data_dir = "/path/to/WinCLIP/AeBAD"          # Data directory for input images and labels
model_cfg_path = "./open_clip/model_configs/ViT-B-16-plus-240.json"  # Path to the model configuration file
checkpoint_path = "./vit_b_16_plus_240-laion400m_e31-8fb26589.pt"    # Path to the pretrained model checkpoint
shot = 1                                     # Number of shots (few-shot setting)
obj_types = ["background", "illumination", "view", "same"]  # List of object types in the dataset
```
## Running the Code
After configuring the dataset and `config.toml`, use the following command to run the main experiment script:

Below is an example configuration for running experiments on the AeBAD dataset:
```python
python main_refactored_aebad.py
```
## Acknowledgements
This refactored implementation builds upon the work of MALA Lab's unofficial WinCLIP implementation. We express our gratitude for their foundational work. Additionally, we acknowledge the authors of the [WinCLIP CVPR'23 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf) and the [InCTRL CVPR'24 paper](https://github.com/mala-lab/InCTRL), which inspired this implementation.
