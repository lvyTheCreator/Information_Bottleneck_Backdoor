# Mutual Information Analysis for Backdoor Training

**Paper: Exploring Dynamic Properties of Backdoor Training Through Information Bottleneck**

This repository implements mutual information analysis techniques when neural networks training on backdoor poisoned datasets. It uses InfoNCE-based mutual information estimation between network inputs, intermediate representations, and outputs to identify potential backdoor patterns.

## Features

- Supports multiple model architectures (ResNet18, VGG16)
- Implements InfoNCE-based mutual information estimation
- Provides visualization tools for MI analysis
- Supports multiple backdoor attack types:
  - BadNet
  - WaNet
  - Blend
  - Label-Consistent
  - Adaptive Blend
- Includes early stopping and learning rate scheduling
- Integrates with Weights & Biases for experiment tracking
- Uses FFCV for efficient data loading
- Calculates MI bounds between inputs, labels, and outputs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/xxxxx/xxx.git
cd IB-backdoor
```

2. Create and configure conda environment:
```bash
# Create conda environment from ffcv.yaml
conda env create -f ffcv.yaml

# Activate the environment
conda activate ffcv
```


## Project Structure

```
IB-backdoor/
├── config.py                 # Configuration settings
├── ffcv_observeMI.py        # Main training and MI analysis script
├── plot.py                  # MI Visualization
├── plot_clustering.py                  # T-SNE Visualization
├── poison_data_generator/   # Backdoor dataset generation
│   └── poison_data_generator.py
├── model/                   # Model architectures
│   ├── resnet.py
│   ├── vgg16.py
│   └── TNet.py
├── estimator/                   # MI estimator experiments
│   ├── estimator.py            # MI estimator selection experiment(InfoNCE vs MINE)
└── util/                    # Utility functions
    ├── cal.py
    └── plot.py
```

## Usage

The project workflow consists of four main steps:

1. Generate backdoor dataset
2. Create FFCV dataloader
3. Train model and observe MI
4. Plot results

### Quick Start Example

Here's a complete example using the CIFAR-10 dataset with a BadNet attack:

```bash
# 1. Generate backdoor dataset
cd poison_data_generator
python poison_data_generator.py --attack_type='badnet' --dataset='cifar10' --poison_percentage=0.1 --target_class=0

# 2. Create FFCV dataloader
python ffcv_writer4.py \
    --output_path="data/cifar10/badnet/0.1" \
    --dataset=all \
    --train_data_path="data/cifar10/badnet/0.1/badnet_0.1.npz" \
    --test_data_path="data/cifar10/badnet/0.1/test_data.npz"

# 3. Train model and observe MI
python ffcv_observeMI.py \
    --outputs_dir="results/cifar10/badnet/ob_infoNCE_01_30_0.1_0.4+0.4" \
    --train_data_path="data/cifar10/badnet/0.1/train_data.beton" \
    --test_data_path="data/cifar10/badnet/0.1/test_data.beton" \
    --sample_data_path="data/cifar10/badnet/0.1/train_data" \
    --model='resnet18' \
    --noise_std_xt=0.4 \
    --noise_std_ty=0.4

# 4. Plot results
python plot.py --directory="results/cifar10/badnet/ob_infoNCE_01_30_0.1_0.4+0.4"

# 5. Calculate MI bounds (optional)
python MI_bound.py
```

### Configuration

The project uses a configuration system defined in `config.py`. You can modify the following parameters:

- Model architecture and parameters
- Training hyperparameters
- MI estimation parameters
- Data loading settings
- Weights & Biases logging configuration

### Supported Datasets

- CIFAR-10
- SVHN

### Supported Backdoor Attacks

1. **BadNet**: Inserts a fixed pattern trigger
2. **WaNet**: Applies spatial transformation
3. **Blend**: Blends a trigger pattern with the original image
4. **Label-Consistent**: Uses adversarial perturbations
5. **Adaptive Blend**: Blend with regularization samples

## Results

The project generates several types of visualizations:

1. Mutual Information plots between:
   - Input and intermediate representations (I(X;T))
   - Intermediate representations and outputs (I(T;Y))
2. Training loss by class
3. Information plane visualization
4. MI bounds between inputs, labels, and outputs

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ib-backdoor2025,
  author = {xxxxx},
  title = {Exploring Dynamic Properties of Backdoor Training Through Information Bottleneck},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/xxxxx/IB-backdoor}
}
```

## Acknowledgments

- Thanks to the FFCV team for the efficient data loading framework
- Weights & Biases for experiment tracking
- PyTorch team for the deep learning framework
