from .correlation_generator import generate_correlation_plot
from ..base_loader import DataLoader


import random
from typing import Tuple

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