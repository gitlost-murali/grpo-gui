from rldatasets.real_gui.gui_dataset import RealGUIDataset, TrainingSample
from PIL import Image


def test_real_gui_generator():
    generator = RealGUIDataset()
    item = generator[0]
    image, target_details_obj = item
    assert isinstance(image, Image.Image)
    assert isinstance(target_details_obj, TrainingSample)
    assert isinstance(target_details_obj.name, str)
    assert isinstance(target_details_obj.bounding_box, tuple)
    assert isinstance(target_details_obj.center_x, int)
    assert isinstance(target_details_obj.center_y, int)
    assert isinstance(target_details_obj.dynamic_prompt, str)
    assert isinstance(target_details_obj.is_hard, bool)
