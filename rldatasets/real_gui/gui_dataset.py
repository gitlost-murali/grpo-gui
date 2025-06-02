import random
from PIL import Image
import datasets
from torch.utils.data import Dataset
from typing import cast

from pydantic import BaseModel


# Dynamic prompt, base template here. {image_width},{image_height},{target_object_name} will be replaced.
GUI_PROMPT_TEMPLATE = """
You will be shown an image of a graphical user interface (GUI).
The image is {image_width}x{image_height} pixels.
Your task is to identify and provide the coordinates to click the target UI element: '{target_object_name}'.

You must answer in the following format exactly:
<reasoning>
Briefly describe your reasoning for choosing the click location based on the target UI element's appearance and position.
</reasoning>
<answer>
x,y
</answer>
Replace x and y with the integer pixel coordinates (e.g., 123,45) where you would click for the '{target_object_name}'. The coordinates must be within the 0-{image_width} range for x and 0-{image_height} range for y.
Do not include any other text in your answer, or any other text after </answer>.

Where would you click to interact with the '{target_object_name}'?
"""


class BoundingBox(BaseModel):
    ymin: int
    xmin: int
    ymax: int
    xmax: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def to_dict(self) -> dict[str, int]:
        return {
            "ymin": self.ymin,
            "xmin": self.xmin,
            "ymax": self.ymax,
            "xmax": self.xmax,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "BoundingBox":
        return cls(
            ymin=data["ymin"],
            xmin=data["xmin"],
            ymax=data["ymax"],
            xmax=data["xmax"],
        )

    @classmethod
    def from_tuple(cls, data: tuple[float, float, float, float]) -> "BoundingBox":
        assert len(data) == 4, "Bounding box must be a tuple of 4 integers"
        assert data[0] < data[2], "xmin must be less than xmax"
        assert data[1] < data[3], "ymin must be less than ymax"
        return cls(
            xmin=int(data[0]),
            ymin=int(data[1]),
            xmax=int(data[2]),
            ymax=int(data[3]),
        )

    def return_center(self) -> tuple[int, int]:
        return int((self.xmin + self.xmax) / 2), int((self.ymin + self.ymax) / 2)


class GUIElement:
    """Represents a single element in the GUI scene."""

    def __init__(
        self,
        name: str,
        center_x: int,
        center_y: int,
        bounding_box: tuple[int, int, int, int],
    ):
        self.name = name
        self.center_x = center_x
        self.center_y = center_y
        self.bounding_box = bounding_box  # (x_min, y_min, x_max, y_max)

    def to_dict(self):
        """Converts the element to a dictionary for JSON serialization."""
        return {
            "name": self.name,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "bounding_box": self.bounding_box,
        }


class TrainingSample(BaseModel):
    name: str
    bounding_box: tuple[int, int, int, int]
    center_x: int
    center_y: int
    dynamic_prompt: str
    is_hard: bool


class RealGUIDataset(Dataset):
    """Generates synthetic GUI scenes with various elements."""

    def __init__(
        self,
        seed=None,
        hf_dataset_name: str = "agentsea/wave-ui-25k",
        split: str = "train",
        train_test_split: float = 0.8,
        dataset_size: int = 9999999,
    ):
        self.elements: list[GUIElement] = []
        if seed is not None:
            random.seed(seed)

        # Load the full dataset (only train split is available in the HF dataset)
        full_dataset = cast(
            datasets.Dataset, datasets.load_dataset(hf_dataset_name, split="train")
        )

        # Calculate split sizes
        train_size = min(dataset_size, int(len(full_dataset) * train_test_split))
        test_size = min(dataset_size, int(len(full_dataset) * (1 - train_test_split)))

        if split == "train":
            self.dataset = full_dataset.select(range(train_size))
        else:  # test split
            self.dataset = full_dataset.select(
                range(train_size, train_size + test_size)
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Image.Image, TrainingSample]:
        """Returns the item at the given index."""
        dataset_sample = self.dataset[index]
        bbox = BoundingBox.from_tuple(tuple(dataset_sample["bbox"]))
        center_x, center_y = bbox.return_center()
        w, h = dataset_sample["resolution"]
        dynamic_prompt = GUI_PROMPT_TEMPLATE.format(
            image_width=w, image_height=h, target_object_name=dataset_sample["name"]
        )

        return dataset_sample["image"], TrainingSample(
            name=str(dataset_sample["name"]),
            bounding_box=bbox.to_tuple(),
            center_x=center_x,
            center_y=center_y,
            dynamic_prompt=dynamic_prompt,
            is_hard=False,  # TODO: add is_hard flag
        )
