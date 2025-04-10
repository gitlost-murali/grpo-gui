"""
Flower dataset implementation for both standard PyTorch training and RL-based training.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils

# Flower classes
FLOWER_CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
FLOWER_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(FLOWER_CLASSES)}

# Standard prompt for all examples
FLOWER_PROMPT = """
You will be shown an image of a flower. Your task is to identify which type of flower it is.

You must answer in the following format:
<reasoning>
Reason about the features of the flower, that are relevant to identifying the flower.
</reasoning>
<answer>
[FLOWER_TYPE].
</answer>
Do not include any other text in your answer, or any other text after </answer>.

What flower is this? The options are: daisy, dandelion, rose, sunflower, tulip
"""

# Image transformations
NORMALIZE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

NO_NORMALIZE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class FlowerDataset(Dataset):
    """
    Standard PyTorch dataset for flower classification.
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to flower images
            labels: List of corresponding class indices
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (image tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label


class FlowerRLLoader():
    """
    RL-based data loader for flower classification.
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], random: bool = False) -> None:
        """
        Initialize the RL data loader.
        
        Args:
            image_paths: List of paths to flower images
            labels: List of corresponding class indices
            random: If True, returns items randomly; if False, returns sequentially
        """
        self.random = random
        self.image_paths = image_paths
        self.labels = labels
        self.current_index = 0
        self.prompt = FLOWER_PROMPT
        
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __iter__(self) -> 'FlowerRLLoader':
        return self
        
    def __next__(self) -> Dict[str, Any]:
        """
        Get the next item in the dataset.
        
        Returns:
            Dictionary containing:
                - prompt: The prompt text
                - answer: The ground truth answer
                - answer_idx: The class index of the answer
                - image: The image tensor (normalized)
                - image_raw: The image tensor (not normalized)
        """
        if self.current_index >= len(self.image_paths):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.image_paths) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        img_path = self.image_paths[idx]
        label_idx = self.labels[idx]
        label_name = FLOWER_CLASSES[label_idx]
        
        return img_path, label_name    


    def reset(self):
        """Reset the iterator to the beginning."""
        self.current_index = 0


def build_flower_dataloaders(
    train_samples_per_class: int = 50,
    test_samples_per_class: int = 10,
    seed: int = 1994,
    batch_size: int = 32
) -> Tuple[FlowerRLLoader, FlowerRLLoader, DataLoader, DataLoader]:
    """
    Build both RL and standard PyTorch dataloaders for the flower dataset.
    
    Args:
        train_samples_per_class: Number of training samples to use per class
        test_samples_per_class: Number of test samples to use per class
        seed: Random seed for reproducibility
        batch_size: Batch size for PyTorch DataLoader
        
    Returns:
        Tuple of (train_rl_loader, test_rl_loader, train_loader, test_loader)
    """
    # Set random seed for reproducibility
    utils.seed_everything(seed)
    
    # Download the dataset if needed
    utils.download_flower_ds()
    
    # Path to the train directory
    train_dir = "data/flowers/train"
    
    # Collect all image paths and labels
    all_image_paths = []
    all_labels = []
    
    for class_name in FLOWER_CLASSES:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            raise ValueError(f"Class directory not found: {class_dir}")
            
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Add paths and labels
        for img_file in image_files:
            all_image_paths.append(os.path.join(class_dir, img_file))
            all_labels.append(FLOWER_CLASS_TO_IDX[class_name])
    
    # Shuffle the data
    indices = list(range(len(all_image_paths)))
    random.shuffle(indices)
    all_image_paths = [all_image_paths[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split into train and test sets
    train_image_paths = []
    train_labels = []
    test_image_paths = []
    test_labels = []
    
    # Group by class
    class_indices = {cls: [] for cls in FLOWER_CLASSES}
    for i, label in enumerate(all_labels):
        class_name = FLOWER_CLASSES[label]
        class_indices[class_name].append(i)
    
    # Select samples for each class
    for class_name in FLOWER_CLASSES:
        class_idx_list = class_indices[class_name]
        
        # Ensure we have enough samples
        if len(class_idx_list) < train_samples_per_class + test_samples_per_class:
            print(f"Warning: Not enough samples for class {class_name}. "
                  f"Using all {len(class_idx_list)} available samples.")
            train_count = min(train_samples_per_class, len(class_idx_list))
            test_count = min(test_samples_per_class, len(class_idx_list) - train_count)
        else:
            train_count = train_samples_per_class
            test_count = test_samples_per_class
        
        # Select indices for train and test
        train_class_indices = class_idx_list[:train_count]
        test_class_indices = class_idx_list[train_count:train_count + test_count]
        
        # Add to train and test lists
        train_image_paths.extend([all_image_paths[i] for i in train_class_indices])
        train_labels.extend([all_labels[i] for i in train_class_indices])
        test_image_paths.extend([all_image_paths[i] for i in test_class_indices])
        test_labels.extend([all_labels[i] for i in test_class_indices])
    
    # Shuffle train and test sets
    train_indices = list(range(len(train_image_paths)))
    test_indices = list(range(len(test_image_paths)))
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    
    train_image_paths = [train_image_paths[i] for i in train_indices]
    train_labels = [train_labels[i] for i in train_indices]
    test_image_paths = [test_image_paths[i] for i in test_indices]
    test_labels = [test_labels[i] for i in test_indices]
    
    # Create RL dataloaders
    train_rl_loader = FlowerRLLoader(train_image_paths, train_labels, random=True)
    test_rl_loader = FlowerRLLoader(test_image_paths, test_labels, random=False)
    
    # Create standard PyTorch dataloaders
    train_dataset = FlowerDataset(train_image_paths, train_labels, transform=NORMALIZE_TRANSFORM)
    test_dataset = FlowerDataset(test_image_paths, test_labels, transform=NORMALIZE_TRANSFORM)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Created flower dataloaders:")
    print(f"  - Train set: {len(train_image_paths)} samples")
    print(f"  - Test set: {len(test_image_paths)} samples")
    
    return train_rl_loader, test_rl_loader, train_loader, test_loader


if __name__ == "__main__":
    # Test the dataloaders
    train_rl_loader, test_rl_loader, train_loader, test_loader = build_flower_dataloaders()
    
    # Test RL loader
    print("\nTesting RL loader:")
    sample = next(iter(train_rl_loader))
    print(f"Prompt: {sample['prompt'][:100]}...")
    print(f"Answer: {sample['answer']}")
    print(f"Answer index: {sample['answer_idx']}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Raw image shape: {sample['image_raw'].size}")
    
    # Test standard loader
    print("\nTesting standard loader:")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels[:30]}")