from typing import Any
from ..base_loader import DataLoader
from .gui_dataset import RealGUIDataset
from PIL import Image


class RealGUIDataLoader(DataLoader):
    def __init__(
        self, is_train: bool, train_test_split: float = 0.8, dataset_size: int = 9999999
    ):
        super().__init__(random=is_train)
        split = "train" if is_train else "test"
        self.dataset = RealGUIDataset(
            dataset_size=dataset_size, split=split, train_test_split=train_test_split
        )
        self.dataset_size = dataset_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> "RealGUIDataLoader":
        self.current_index = 0
        return self

    def __next__(self) -> tuple[Image.Image, dict[str, Any]]:
        if not self.is_train and self.current_index >= len(self.dataset):
            raise StopIteration

        if self.random:
            import random

            index = random.randint(0, len(self.dataset) - 1)
        else:
            index = self.current_index

        self.current_index += 1
        image, target_details_obj = self.dataset[index]
        return image, target_details_obj.model_dump()

    def reset(self):
        self.current_index = 0
