"""
FFCV Dataset Converter for Label-Consistent Backdoor Attack

This script converts NPZ format datasets to FFCV (.beton) format, specifically designed for
label-consistent backdoor attack scenarios. It supports:
1. Converting training and test datasets
2. Creating class-specific sample datasets
3. Separating poisoned and clean samples for class 0
4. Adding backdoor flags to training data

Usage:
    python ffcv_writer_clean.py --train_data_path <path> --test_data_path <path> 
                             --output_path <path> --dataset <type> 
                             --observe_classes <classes> --poison_rate <rate>

Arguments:
    --train_data_path: Path to training data (.npz)
    --test_data_path: Path to test data (.npz)
    --output_path: Directory to save .beton files
    --dataset: Type of dataset to generate ('all', 'train', 'test', or 'sample')
    --observe_classes: List of classes to observe (default: [0-9])
    --poison_rate: Ratio of poisoned samples in class 0 (default: 0.25)
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, TorchTensorField
from typing import List, Dict, Tuple, Optional
import argparse
import os
from pathlib import Path

class DatasetConfig:
    """Container for dataset configuration parameters"""
    def __init__(self, args):
        self.train_path = args.train_data_path
        self.test_path = args.test_data_path
        self.output_dir = Path(args.output_path)
        self.dataset_type = args.dataset
        self.observe_classes = args.observe_classes
        self.poison_rate = args.poison_rate
        self.device = 'cpu'

def load_npz_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate .npz data file
    
    Args:
        path: Path to .npz data file
        
    Returns:
        images: Image data array (N, H, W, C)
        labels: Label array (N,)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file {path} not found")
    
    data = np.load(path)
    return data['arr_0'], data['arr_1']

def create_tensor_dataset(images: np.ndarray,
                         labels: np.ndarray,
                         is_backdoor: Optional[torch.Tensor] = None,
                         device: str = 'cpu') -> TensorDataset:
    """Create standardized TensorDataset with optional backdoor flag
    
    Args:
        images: Image data (N, H, W, C)
        labels: Label data (N,)
        is_backdoor: Optional backdoor flag tensor (N,)
        device: Device for tensor storage
        
    Returns:
        TensorDataset containing images, labels, and optional backdoor flag
    """
    image_tensor = torch.tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    
    if is_backdoor is not None:
        return TensorDataset(image_tensor, label_tensor, is_backdoor)
    
    return TensorDataset(image_tensor, label_tensor)

def create_class_datasets(train_images: np.ndarray,
                         train_labels: np.ndarray,
                         config: DatasetConfig) -> Dict[str, TensorDataset]:
    """Create class-specific datasets with special handling for poisoned class 0
    
    Args:
        train_images: Training image data
        train_labels: Training label data
        config: Dataset configuration
        
    Returns:
        Dictionary containing class-specific datasets
    """
    datasets = {}
    
    # Create standard class datasets
    for cls in config.observe_classes:
        mask = (train_labels == cls)
        datasets[f'class_{cls}'] = create_tensor_dataset(
            train_images[mask], train_labels[mask], device=config.device)
    
    # Special handling for class 0 (poisoned class)
    if 0 in config.observe_classes:
        cls0_mask = (train_labels == 0)
        cls0_images = train_images[cls0_mask]
        cls0_labels = train_labels[cls0_mask]
        
        # Split poisoned and clean data
        poison_num = int(len(cls0_images) * config.poison_rate)
        datasets.update({
            'class_0_backdoor': create_tensor_dataset(
                cls0_images[:poison_num], cls0_labels[:poison_num], device=config.device),
            'class_0_clean': create_tensor_dataset(
                cls0_images[poison_num:], cls0_labels[poison_num:], device=config.device)
        })
    
    return datasets

def write_beton_dataset(dataset: TensorDataset,
                       output_path: Path,
                       fields: Dict,
                       dataset_name: str = ''):
    """Write dataset to Beton format
    
    Args:
        dataset: Dataset to write
        output_path: Output directory
        fields: Field definitions
        dataset_name: Dataset name for logging
    """
    writer = DatasetWriter(output_path, fields)
    writer.from_indexed_dataset(dataset)
    print(f"Dataset successfully written: {dataset_name} -> {output_path}, num_samples={len(dataset)}")

def main(config: DatasetConfig):
    """Main processing pipeline"""
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    train_images, train_labels = load_npz_data(config.train_path)
    test_images, test_labels = load_npz_data(config.test_path)
    
    # Create backdoor flags for training data
    total_samples = len(train_images)
    num_samples_per_class = np.sum(train_labels==0)
    poison_samples = int(num_samples_per_class * config.poison_rate) # % of class 0
    is_backdoor = torch.zeros(total_samples, dtype=torch.long, device=config.device)
    is_backdoor[:poison_samples] = 1
    
    # Create base datasets
    base_datasets = {
        'train': create_tensor_dataset(train_images, train_labels, is_backdoor, config.device),
        'test': create_tensor_dataset(test_images, test_labels, device=config.device)
    }
    
    # Create class-specific datasets
    class_datasets = create_class_datasets(train_images, train_labels, config)
    
    # Define common field templates
    COMMON_FIELDS = {
        'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
        'label': IntField()
    }
    TRAIN_FIELDS = {
        **COMMON_FIELDS,
        'is_backdoor': IntField()
    }
    
    # Write datasets based on configuration
    if config.dataset_type in ('all', 'train'):
        write_beton_dataset(
            base_datasets['train'],
            config.output_dir/'train_data.beton',
            TRAIN_FIELDS,
            'Training dataset'
        )
    
    if config.dataset_type in ('all', 'test'):
        write_beton_dataset(
            base_datasets['test'],
            config.output_dir/'test_data.beton',
            COMMON_FIELDS,
            'Test dataset'
        )

    if config.dataset_type in ('all', 'sample'):
        for name, dataset in class_datasets.items():
            write_beton_dataset(
                dataset,
                config.output_dir/f'{name}.beton',
                COMMON_FIELDS,
                f'Sample dataset {name}'
            )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FFCV Dataset Converter for Label-Consistent Backdoor Attack')
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to training data (.npz format)')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='Path to test data (.npz format)')
    parser.add_argument('--output_path', type=str, default='datasets',
                        help='Output directory path')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'train', 'test', 'sample'],
                        help='Dataset type to generate')
    parser.add_argument('--observe_classes', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9],
                        help='List of classes to observe')
    parser.add_argument('--poison_rate', type=float, default=0.25,
                        help='Poison data ratio for class 0 (0-1)')
    
    args = parser.parse_args()
    config = DatasetConfig(args)
    
    try:
        main(config)
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        exit(1)
