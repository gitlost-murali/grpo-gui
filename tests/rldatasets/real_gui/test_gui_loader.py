import pytest
import rldatasets
from PIL import Image


@pytest.fixture
def gui_hard_dataset():
    return rldatasets.get_dataloaders(dataset_name="gui_hard")


@pytest.fixture
def gui_dataset():
    return rldatasets.get_dataloaders(dataset_name="gui")


def test_output_type():
    hard_train_loader, _ = rldatasets.get_dataloaders(dataset_name="gui_hard")

    train_loader, _ = rldatasets.get_dataloaders(dataset_name="gui")

    for _ in range(2):
        gui_hard_batch = next(hard_train_loader)
        gui_batch = next(train_loader)

        image, gui_hard_target_details_dict = gui_hard_batch
        assert isinstance(image, Image.Image)
        assert isinstance(gui_hard_target_details_dict, dict)

        image_path, gui_target_details_dict = gui_batch
        assert isinstance(image_path, str)
        assert isinstance(gui_target_details_dict, dict)

        assert gui_hard_target_details_dict.keys() == gui_target_details_dict.keys()
