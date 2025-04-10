"""
Flower dataset implementation for both standard PyTorch training and RL-based training.
"""

import os
import random
import json
import numpy as np
import torch
import csv
import kagglehub
# Add shutil for file copying
import shutil 
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt


import utils

def _load_flower_102_class_names(data_dir="data/flowers102", kaggle_dataset="hobaak/oxford-102-flower-name-index", csv_filename="index_to_name.csv", index_header='Index', name_header='Name') -> Dict[int, str]:
    """Loads class names from a local CSV, downloading and copying it from Kaggle if not found. Returns a dict mapping 1-based index to name."""
    
    target_csv_path = os.path.join(data_dir, csv_filename)
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(target_csv_path):
        dataset_path = kagglehub.dataset_download(kaggle_dataset)
        source_csv_path = os.path.join(dataset_path, csv_filename)
        if not os.path.exists(source_csv_path):
            for file in os.listdir(dataset_path):
                if file.lower().endswith('.csv'):
                    source_csv_path = os.path.join(dataset_path, file)
                    break
        shutil.copy2(source_csv_path, target_csv_path)

    name_map = {}
    with open(target_csv_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            idx = int(row[index_header])
            name = row[name_header]
            name_map[idx] = name

    return name_map

name_map = _load_flower_102_class_names()
IDX_TO_FLOWER_NAME = name_map

print(IDX_TO_FLOWER_NAME)

FLOWER_CLASSES = [IDX_TO_FLOWER_NAME.get(i, f"Unknown Class {i+1}") for i in range(102)]
FLOWER_CLASS_TO_IDX = {name: idx for idx, name in enumerate(FLOWER_CLASSES)}
NUM_CLASSES = len(FLOWER_CLASSES)


# Updated prompt for 102 classes - now includes the class list
FLOWER_PROMPT = f"""
You will be shown an image of a flower. Your task is to identify which type of flower it is.

The options are: {', '.join(FLOWER_CLASSES)}

You must answer in the following format:
<reasoning>
Reason about the features of the flower that are relevant to identifying the flower.
</reasoning>
<answer>
[FLOWER_TYPE].
</answer>
Do not include any other text in your answer, or any other text after </answer>.

What flower is this? 
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


class FlowerRLLoader():
    """
    RL-based data loader for flower classification using Flowers102 dataset (or Subset).
    Provides data sequentially or randomly.
    """

    def __init__(self, dataset: Union[datasets.Flowers102, Subset], class_names: List[str], random: bool = False) -> None:
        """
        Initialize the RL data loader.

        Args:
            dataset: The initialized torchvision Flowers102 dataset instance or a Subset of it.
            class_names: List of class names corresponding to dataset labels (0-101).
            random: If True, returns items randomly; if False, returns sequentially.
        """
        self.dataset = dataset
        # Ensure class_names list matches the expected number of classes
        if len(class_names) != NUM_CLASSES:
            print(f"Warning: FlowerRLLoader received {len(class_names)} class names, but expected {NUM_CLASSES}.")
        self.class_names = class_names 
        self.random = random
        self.current_index = 0
        self.prompt = FLOWER_PROMPT
        # If using Subset, indices are relative to the Subset, not original dataset
        self._indices = list(range(len(self.dataset))) 

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> 'FlowerRLLoader':
        self.current_index = 0
        if self.random:
            random.shuffle(self._indices)
        return self

    def __next__(self) -> Tuple[str, str]:
        """
        Get the next item in the dataset.

        Returns:
            Tuple of (image_path, answer_string)
        """
        if not self.random and self.current_index >= len(self.dataset):
            raise StopIteration

        if self.random:
            idx_in_subset = random.choice(self._indices)
        else:
            idx_in_subset = self._indices[self.current_index]
            self.current_index += 1

        # Get image and label from the dataset/subset
        # Dataset returns (PIL Image, label_index 0..101)
        img_raw, label_idx = self.dataset[idx_in_subset] 

        if img_raw.mode != 'RGB':
            img_raw = img_raw.convert('RGB')

        img_normalized = NORMALIZE_TRANSFORM(img_raw)
        
        # Get label name using the 0-based index
        try:
            label_name = self.class_names[label_idx]
        except IndexError:
            print(f"Warning: Label index {label_idx} out of range for class names list (len {len(self.class_names)}).")
            label_name = f"Unknown Class {label_idx+1}" # Fallback

        # save as temp file
        img_resized = img_raw.resize((224, 224))
        img_resized.save(f"temp_image.png")
        return (f"temp_image.png", label_name)

    def reset(self):
        """Reset the iterator to the beginning."""
        self.current_index = 0
        if self.random:
            random.shuffle(self._indices)



def _get_subset_indices(dataset: Union[datasets.Flowers102, Subset], num_samples_per_class: int, num_classes: int) -> List[int]:
    """
    Performs stratified sampling to get indices for a subset.
    
    Args:
        dataset: The dataset to sample from.
        num_samples_per_class: Target number of samples for each class.
        num_classes: The total number of classes (e.g., 102).

    Returns:
        A list of indices selected for the subset.
    """
    print(f"Creating subset with {num_samples_per_class} samples per class...")
    labels = []
    print("Fetching labels for subsampling (this might take a moment)...")
    # Access labels efficiently. Subset stores original dataset and indices.
    if isinstance(dataset, Subset):
        # Get labels corresponding to the subset indices
        original_dataset = dataset.dataset
        # Ensure original dataset has _labels attribute or similar, common for torchvision image datasets
        # Flowers102 stores labels internally, need to access them via _labels
        if hasattr(original_dataset, '_labels'):
            original_labels = original_dataset._labels
            labels = [original_labels[i] for i in dataset.indices]
        else:
             # Fallback: iterate if no direct access (slower)
             print("Warning: Cannot directly access labels for subset, iterating...")
             labels = [label for _, label in tqdm(dataset, desc="Getting Subset Labels")]
    elif hasattr(dataset, '_labels'): # Access labels directly if available on main dataset
         labels = dataset._labels
    else:
        # Fallback: iterate through the full dataset (slower)
        print("Warning: Cannot directly access labels for dataset, iterating...")
        labels = [label for _, label in tqdm(dataset, desc="Getting Full Dataset Labels")]
    
    print(f"Found {len(labels)} total labels for potential inclusion.")
    
    indices_by_class = {i: [] for i in range(num_classes)}
    for i, label in enumerate(labels):
        if 0 <= label < num_classes:
             indices_by_class[label].append(i) # Store index *within the current dataset/subset*
        else:
             print(f"Warning: Encountered unexpected label {label} at index {i}.")

    subset_indices = []
    for class_idx in range(num_classes):
        available_indices = indices_by_class[class_idx]
        if not available_indices:
            print(f"Warning: No samples found for class {class_idx}.")
            continue

        if len(available_indices) < num_samples_per_class:
            print(f"Warning: Class {class_idx} has only {len(available_indices)} samples, requesting {num_samples_per_class}. Using all available.")
            selected = available_indices
        else:
            selected = random.sample(available_indices, num_samples_per_class)
        
        subset_indices.extend(selected)

    random.shuffle(subset_indices)
    print(f"Selected {len(subset_indices)} indices for the subset.")
    return subset_indices


def build_flower_dataloaders(
    data_dir: str = "data/flowers102",
    batch_size: int = 32,
    seed: int = 1994,
    # Add subsampling parameters
    train_samples_per_class: Optional[int] = 10, 
    val_samples_per_class: Optional[int] = 2,
) -> Tuple[FlowerRLLoader, FlowerRLLoader, DataLoader, DataLoader]:
    """
    Build both RL and standard PyTorch dataloaders for the Flowers102 dataset.
    Allows specifying the number of samples per class for train and validation sets.

    Args:
        data_dir: Directory where the dataset is or will be downloaded.
        batch_size: Batch size for standard PyTorch DataLoader.
        seed: Random seed for reproducibility.
        train_samples_per_class: Number of training samples per class. If None, use all.
        val_samples_per_class: Number of validation samples per class. If None, use all.

    Returns:
        Tuple of (train_rl_loader, test_rl_loader, train_loader, test_loader)
    """
    utils.seed_everything(seed)
    os.makedirs(data_dir, exist_ok=True)
    
    # --- Load FULL Datasets first ---
    print("Loading full Flowers102 datasets...")
    full_train_dataset_raw = datasets.Flowers102(root=data_dir, split='train', download=True, transform=None)
    full_val_dataset_raw = datasets.Flowers102(root=data_dir, split='val', download=True, transform=None) # Use 'val' for testing

    full_train_dataset_std = datasets.Flowers102(root=data_dir, split='train', download=False, transform=NORMALIZE_TRANSFORM)
    full_val_dataset_std = datasets.Flowers102(root=data_dir, split='val', download=False, transform=NORMALIZE_TRANSFORM)
    
    print(f"Full dataset sizes: Train={len(full_train_dataset_raw)}, Val={len(full_val_dataset_raw)}")

    # --- Perform Subsampling ---
    train_indices = list(range(len(full_train_dataset_raw))) # Default to all if no sampling
    if train_samples_per_class is not None and train_samples_per_class > 0:
         # Flowers102 uses 0-101 indices, NUM_CLASSES should be 102
         train_indices = _get_subset_indices(full_train_dataset_raw, train_samples_per_class, NUM_CLASSES) 
    
    val_indices = list(range(len(full_val_dataset_raw))) # Default to all if no sampling
    if val_samples_per_class is not None and val_samples_per_class > 0:
        val_indices = _get_subset_indices(full_val_dataset_raw, val_samples_per_class, NUM_CLASSES)

    # --- Create Subset datasets ---
    print("Creating subset datasets...")
    train_subset_raw = Subset(full_train_dataset_raw, train_indices)
    val_subset_raw = Subset(full_val_dataset_raw, val_indices) # Use 'val' split indices
    train_subset_std = Subset(full_train_dataset_std, train_indices)
    val_subset_std = Subset(full_val_dataset_std, val_indices) # Use 'val' split indices

    print(f"Subset sizes: Train={len(train_subset_raw)}, Val/Test={len(val_subset_raw)}")

    # --- Create Dataloaders using Subsets ---
    print("Creating dataloaders...")
    # Pass the subset and the global FLOWER_CLASSES list
    train_rl_loader = FlowerRLLoader(train_subset_raw, FLOWER_CLASSES, random=True)
    # Use the VAL subset for the test RL loader
    test_rl_loader = FlowerRLLoader(val_subset_raw, FLOWER_CLASSES, random=False) 

    train_loader = DataLoader(train_subset_std, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Use the VAL subset for the standard test loader
    test_loader = DataLoader(val_subset_std, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Finished creating dataloaders.")
    print(f"  - RL Train set: {len(train_rl_loader)} samples")
    print(f"  - RL Test set (from Val split): {len(test_rl_loader)} samples")
    print(f"  - Standard Train set: {len(train_loader.dataset)} samples") # Access underlying subset size
    print(f"  - Standard Test set (from Val split): {len(test_loader.dataset)} samples")

    return train_rl_loader, test_rl_loader, train_loader, test_loader


if __name__ == "__main__":
    from matplotlib.backends.backend_pdf import PdfPages # Import for PDF saving
    import matplotlib.pyplot as plt # Already imported, but good practice
    from PIL import Image # Already imported

    print("Testing dataloader creation with default subsampling (10 train, 2 val per class)...")
    # Test with default subsampling
    train_rl, test_rl, train_std, test_std = build_flower_dataloaders(
         train_samples_per_class=10,
         val_samples_per_class=2
    )

    print(f"\nRL Train Loader length: {len(train_rl)}")
    print(f"RL Test Loader (Val) length: {len(test_rl)}")
    print(f"Standard Train Loader batches: {len(train_std)}")
    print(f"Standard Test Loader (Val) batches: {len(test_std)}")

    # --- Visual Inspection PDF Generation ---
    print("\nGenerating visual inspection PDF (first 5 samples from underlying subset)...")
    pdf_filename = "flower_inspection_direct.pdf" # New name to avoid confusion
    num_samples_to_show = 5
    source_dataset = train_rl.dataset # Access the underlying Subset object

    # Check if source_dataset has enough samples
    if len(source_dataset) < num_samples_to_show:
        print(f"Warning: Source dataset has only {len(source_dataset)} samples. Showing all.")
        num_samples_to_show = len(source_dataset)

    try:
        with PdfPages(pdf_filename) as pdf:
            for i in range(num_samples_to_show):
                try:
                    # Get data directly from the subset: (PIL Image, label_index 0..101)
                    img_raw, label_idx = source_dataset[i]
                    
                    # Ensure image is RGB
                    if img_raw.mode != 'RGB':
                        img_raw = img_raw.convert('RGB')

                    # Get label name using the 0-based index and the global mapping
                    try:
                        label_name = FLOWER_CLASSES[label_idx]
                    except IndexError:
                         print(f"    Warning: Label index {label_idx} out of range for FLOWER_CLASSES (len {len(FLOWER_CLASSES)}).")
                         label_name = f"Unknown Class {label_idx+1}" # Fallback
                    except Exception as name_err:
                         print(f"    Warning: Error getting label name for index {label_idx}: {name_err}")
                         label_name = f"Error getting name for {label_idx}"

                    print(f"  Processing sample {i} (Label Index: {label_idx}): {label_name}")

                    # Resize image for display (can use NO_NORMALIZE_TRANSFORM or manual resize)
                    img_display = img_raw.resize((224, 224))

                    # Create plot
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    ax.imshow(img_display)
                    ax.set_title(f"Idx: {label_idx} - {label_name}", fontsize=10) # Show index and name
                    ax.axis('off') # Hide axes
                    plt.tight_layout()

                    pdf.savefig(fig) # Save the current figure to the PDF
                    plt.close(fig) # Close the figure to free memory

                except Exception as e:
                     print(f"  Error processing sample index {i}: {e}")
                     # Attempt to close any open plot if an error occurred mid-plot
                     try:
                         plt.close(fig)
                     except NameError: # fig might not be defined if error happened early
                         pass
                     except Exception:
                         pass # Ignore errors during cleanup closing


        print(f"Successfully generated direct visual inspection PDF: {pdf_filename}")

    except Exception as e:
        print(f"  Error during PDF generation: {e}")

    # --- Original Loader Iteration Tests (Optional) ---
    # Test RL loader iteration
    print("\nTesting RL loader (Train) iteration (first sample only):")
    try:
        # Get the first sample: (image_path, answer_string)
        train_rl.reset() # Reset iterator if needed
        img_path, answer = next(iter(train_rl))
        print(f"  Sample retrieved successfully.")
        print(f"  Image Path: {img_path}")
        print(f"  Answer: {answer}")
        # Test getting a second sample
        # img_path_2, answer_2 = next(iter(train_rl)) # Commented out to avoid confusion with PDF part
        # print(f"  Second Sample: ({img_path_2}, {answer_2})")
    except Exception as e:
        print(f"  Error iterating through RL train loader: {e}")

    print("\nTesting standard loader (Train) iteration:")
    try:
        images, labels = next(iter(train_std))
        print(f"  Batch retrieved successfully.")
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels (first 10): {labels[:10].tolist()}")
        # Map labels to names for verification
        label_names = [FLOWER_CLASSES[l] for l in labels[:10].tolist()]
        print(f"  Label Names (first 10): {label_names}")
    except Exception as e:
         print(f"  Error iterating through standard train loader: {e}")

