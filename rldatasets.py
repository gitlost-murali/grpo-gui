"""
Data loader for generating analog clock images and times.
"""

import random
import os
from typing import Tuple, Any
from abc import ABC, abstractmethod
from clock_generator import TimeObj, ClockGen
from correlation_generator import generate_correlation_plot # Import the correlation generator


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


# --- Correlation Scatter Dataset (using temp file) --- 

CORRELATION_PROMPT = f"""
You will be shown a scatter plot image displaying a relationship between two variables.
Your task is to estimate the Pearson correlation coefficient (R) depicted in the plot.
The correlation R is a value between 0.00 and 1.00, where 0.00 indicates no correlation and 1.00 indicates perfect positive correlation.

You must answer in the following format exactly:
<reasoning>
Analyze the scatter plot. Consider how closely the points follow a linear pattern. A tighter cluster along a line indicates a higher correlation.
</reasoning>
<answer>
X.XX
</answer>
Replace X.XX with your estimated correlation coefficient, formatted to two decimal places (e.g., 0.75, 0.00, 1.00).
Do not include any other text in your answer, or any other text after </answer>.

What is the estimated correlation R shown in this scatter plot?
"""

class CorrelationScatterDataLoader(DataLoader):
    """
    A data loader that generates random correlation scatter plot images and their corresponding R-values,
    saving the image to a temporary file.
    
    Attributes:
        dataset_size (int): The nominal size of the dataset (used for testing length).
        is_train (bool): Flag indicating if this is a training loader.
        prompt (str): The instruction prompt for the language model.
        temp_image_path (str): Path where the temporary correlation image is saved.
        num_points (int): Number of points for the scatter plot.
        image_size (tuple): Size of the generated image.
    """
    
    def __init__(self, dataset_size: int = 50, is_train: bool = True) -> None:
        super().__init__(random=True) # Always generates random correlations
        self.dataset_size = dataset_size
        self.is_train = is_train
        self.prompt = CORRELATION_PROMPT
        self.temp_image_path = "temp_correlation.png" # Fixed path for the temporary image
        self.num_points = 75 # Standard number of points for correlation plots
        self.image_size = (224, 224) # Standard size
        
    def __len__(self) -> int:
        return self.dataset_size
        
    def __iter__(self) -> 'CorrelationScatterDataLoader':
        self.current_index = 0
        return self
        
    def __next__(self) -> Tuple[str, str]: # Returns path and label string
        if not self.is_train and self.current_index >= self.dataset_size:
            raise StopIteration
            
        self.current_index += 1
        
        # Generate a random R value between 0.00 and 1.00
        r_value = random.uniform(0.0, 1.0)
        
        # Generate the scatter plot image and save it to the temp path
        generate_correlation_plot(
            r_value=r_value,
            num_points=self.num_points,
            filename=self.temp_image_path, # Save to temp path
            img_size=self.image_size,
        )

        # Format the R value string to X.XX
        r_string = f"{r_value:.2f}"
            
        return self.temp_image_path, r_string # Return path and label

    def reset(self):
        self.current_index = 0


# --- Factory Function --- 

def get_dataloaders(dataset_name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset ('clock' or 'correlation').
        **kwargs: Additional arguments for specific data loaders (e.g., dataset_size).

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported.
    """
    dataset_size = kwargs.get('dataset_size', 50) # Default size for test set
    dataset_name = dataset_name.lower() # Normalize name

    if dataset_name == 'clock':
        # Training loader generates infinitely (conceptually), test loader has fixed size
        trainloader = ClockDataLoader(dataset_size=dataset_size * 100, is_train=True) # Large nominal size for train
        testloader = ClockDataLoader(dataset_size=dataset_size, is_train=False)
        return trainloader, testloader
    elif dataset_name == 'correlation':
        trainloader = CorrelationScatterDataLoader(dataset_size=dataset_size * 100, is_train=True)
        testloader = CorrelationScatterDataLoader(dataset_size=dataset_size, is_train=False)
        return trainloader, testloader
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Supported: 'clock', 'correlation'")


if __name__ == "__main__": 
    # Test Clock Loader
    print("--- Testing Clock Loader ---")
    try:
        train_loader_c, test_loader_c = get_dataloaders('clock', dataset_size=2)
        print(f"Clock Train loader prompt: {train_loader_c.prompt[:80]}...")
        print(f"Clock Test loader length: {len(test_loader_c)}")

        print("  Train Sample:")
        img_path_c, label_c = next(train_loader_c)
        print(f"    Image Path: {img_path_c}, Label: {label_c}")

        print("  Test Samples:")
        test_loader_c.reset()
        for img_path_c, label_c in test_loader_c:
            print(f"    Image Path: {img_path_c}, Label: {label_c}")
        print("  Clock Test loader iteration finished.")
    except Exception as e:
        print(f"Error testing clock loader: {e}")

    # Test Correlation Loader
    print("\n--- Testing Correlation Loader ---")
    try:
        train_loader_r, test_loader_r = get_dataloaders('correlation', dataset_size=2)
        print(f"Correlation Train loader prompt: {train_loader_r.prompt[:80]}...")
        print(f"Correlation Test loader length: {len(test_loader_r)}")

        print("  Train Sample:")
        img_path_r, label_r = next(train_loader_r)
        print(f"    Image Path: {img_path_r}, Label: {label_r}")

        print("  Test Samples:")
        test_loader_r.reset()
        for img_path_r, label_r in test_loader_r:
            print(f"    Image Path: {img_path_r}, Label: {label_r}")
        print("  Correlation Test loader iteration finished.")
    except Exception as e:
        print(f"Error testing correlation loader: {e}")