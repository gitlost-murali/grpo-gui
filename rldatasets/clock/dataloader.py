from .clock_generator import ClockGen, TimeObj
from ..base_loader import DataLoader
from typing import Tuple
import random

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
        super().__init__(random=True)  # Always generates random times
        self.dataset_size = dataset_size
        self.is_train = is_train
        self.prompt = CLOCK_PROMPT
        self.temp_image_path = "temp_clock.png"  # Fixed path for the temporary image

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