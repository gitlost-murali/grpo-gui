"""
Data loader for generating analog clock images and times.
"""

import random
import os
from typing import Tuple, Any
from abc import ABC, abstractmethod
from clock_generator import TimeObj, ClockGen


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    This class defines the interface that all dataset loaders should implement.
    Specific dataset loaders should inherit from this class and implement the
    required methods.
    
    Attributes:
        random (bool): If True, returns items randomly; if False, returns sequentially
        current_index (int): Current position for sequential access
    """
    
    def __init__(self, random: bool = False) -> None:
        self.random = random
        self.current_index = 0
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        pass
        
    @abstractmethod
    def __iter__(self) -> 'DataLoader':
        """Return self as iterator."""
        return self
        
    @abstractmethod
    def __next__(self) -> Any:
        """Return the next item(s) in the dataset."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the iterator to the beginning."""
        pass


# Define the prompt for the Clock dataset
CLOCK_PROMPT = f"""
You will be shown an image of an analog clock. Your task is to determine the time shown.

You must answer in the following format exactly:
<reasoning>
Reason about the positions of the hour, minute, and second hands to determine the time.
</reasoning>
<answer>
[HH:MM:SS]
</answer>
Replace HH, MM, and SS with the two-digit hour (01-12), minute (00-59), and second (00-59).
Do not include any other text in your answer, or any other text after </answer>.

What time is shown on this clock?
"""


class ClockDataLoader(DataLoader):
    """
    A data loader that generates random analog clock images and their corresponding times.
    
    For training, it generates infinitely. For testing, it generates a fixed number.
    
    Attributes:
        dataset_size (int): The nominal size of the dataset (used for testing length).
        is_train (bool): Flag indicating if this is a training loader.
        prompt (str): The instruction prompt for the language model.
        temp_image_path (str): Path where the temporary clock image is saved.
    """
    
    def __init__(self, dataset_size: int = 50, is_train: bool = True) -> None:
        super().__init__(random=True) # Always generates random times
        self.dataset_size = dataset_size
        self.is_train = is_train
        self.prompt = CLOCK_PROMPT
        self.temp_image_path = "temp_clock.png" # Fixed path for the temporary image
        
    def __len__(self) -> int:
        # Return the specified size, mainly relevant for the test set iteration count
        return self.dataset_size
        
    def __iter__(self) -> 'ClockDataLoader':
        self.current_index = 0
        return self
        
    def __next__(self) -> Tuple[str, str]:
        # Stop iteration for the test set after reaching dataset_size
        if not self.is_train and self.current_index >= self.dataset_size:
            raise StopIteration
        
        self.current_index += 1
        
        # Generate a random time
        time_obj = TimeObj()
        
        # Generate the clock image
        clock_gen = ClockGen(time_obj)
        clock_gen.generate_clock(filename=self.temp_image_path)

        # Get the time string in [HH:MM:SS] format
        time_string = str(time_obj)
            
        return self.temp_image_path, time_string

    def reset(self):
        self.current_index = 0 


def get_dataloaders(dataset_name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load ('clock' currently supported).
        **kwargs: Additional arguments for specific data loaders (e.g., dataset_size).

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported.
    """
    dataset_size = kwargs.get('dataset_size', 50) # Default size for test set

    if dataset_name.lower() == 'clock':
        # Training loader generates infinitely (conceptually), test loader has fixed size
        trainloader = ClockDataLoader(dataset_size=dataset_size * 100, is_train=True) # Large nominal size for train
        testloader = ClockDataLoader(dataset_size=dataset_size, is_train=False)
        return trainloader, testloader
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Currently supported: 'clock'")


if __name__ == "__main__": 
    train_loader, test_loader = get_dataloaders('clock', dataset_size=5) # Use smaller size for testing
    print(f"Train loader prompt: {train_loader.prompt}")
    print(f"Test loader length: {len(test_loader)}")

    print("\n--- Training Loader Samples (first 2) ---")
    count = 0
    for img_path, time_str in train_loader:
        print(f" Image Path: {img_path}, Time: {time_str}")
        count += 1
        if count >= 2:
            break

    print("\n--- Test Loader Samples (all) ---")
    test_loader.reset()
    for img_path, time_str in test_loader:
        print(f" Image Path: {img_path}, Time: {time_str}")
    
    print("\nTest loader iteration finished.")