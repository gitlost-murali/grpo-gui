import json
from .gui_generator import GUIGenerator
from ..base_loader import DataLoader
import random
from typing import Any, Dict, Tuple

# Dynamic prompt, base template here. {target_object_name} will be replaced.
GUI_PROMPT_TEMPLATE = """
You will be shown an image of a graphical user interface (GUI).
The image is 224x224 pixels.
Your task is to identify and provide the coordinates to click the target object: '{target_object_name}'.

You must answer in the following format exactly:
<reasoning>
Briefly describe your reasoning for choosing the click location based on the target object's appearance and position.
</reasoning>
<answer>
x,y
</answer>
Replace x and y with the integer pixel coordinates (e.g., 123,45) where you would click for the '{target_object_name}'. The coordinates must be within the 0-223 range for both x and y.
Do not include any other text in your answer, or any other text after </answer>.

Where would you click to interact with the '{target_object_name}'?
"""


class GUIDataLoader(DataLoader):
    """
    A data loader that generates GUI scenes, selects a target object, and provides a prompt to click it.

    Attributes:
        dataset_size (int): Nominal size (for testing).
        is_train (bool): Training or testing mode.
        gui_generator (GUIGenerator): Instance of the GUI scene generator.
        temp_image_path (str): Path for the temporary GUI image.
        prompt_template (str): Base prompt string with placeholder for target object name.
        # `self.prompt` will be dynamically set in __next__ for this loader
    """
    def __init__(self, dataset_size: int = 50, is_train: bool = True,
                 image_width: int = 224, image_height: int = 224,
                 hard_mode_prob: float = 0.1):
        super().__init__(random=True)
        self.dataset_size = dataset_size
        self.is_train = is_train
        self.gui_generator = GUIGenerator(width=image_width, height=image_height)
        self.temp_image_path = "temp_gui_scene.png"
        self.prompt_template = GUI_PROMPT_TEMPLATE
        self.prompt = ""
        self.hard_mode_prob = hard_mode_prob

    def __len__(self) -> int:
        return self.dataset_size

    def __iter__(self) -> 'GUIDataLoader':
        self.current_index = 0
        return self

    def __next__(self) -> Tuple[str, Dict[str, Any]]:
        """
        Returns:
            Tuple[str, Dict[str, Any]]: 
                - image_path (str): Path to the saved GUI image.
                - target_info (Dict[str, Any]): Dictionary containing target details:
                    {'name': str, 'bounding_box': tuple, 'center_x': int, 'center_y': int, 'dynamic_prompt': str, 'is_hard': bool}
        """
        if not self.is_train and self.current_index >= self.dataset_size:
            raise StopIteration

        self.current_index += 1

        # Determine if this example should be hard mode based on the stored probability
        use_hard_mode = random.random() < self.hard_mode_prob

        target_info = None
        max_retries = 5
        for _ in range(max_retries):
            gui_image, _, temp_target_info = self.gui_generator.generate_scene_with_target(generate_hard_mode=use_hard_mode)
            if temp_target_info:
                target_info = temp_target_info
                break

        if not target_info:
            all_elements = json.loads(_)
            if all_elements and any(el['name'] == 'window' for el in all_elements):
                target_info = next(el for el in all_elements if el['name'] == 'window')
            else:
                raise ValueError("Failed to generate a GUI scene with any identifiable target element after multiple retries.")

        gui_image.save(self.temp_image_path)

        target_object_name_for_prompt = target_info['name'].replace("_", " ")
        self.prompt = self.prompt_template.format(target_object_name=target_object_name_for_prompt)

        # Add the 'is_hard' flag to the answer dictionary
        answer_for_evaluator = {
            "name": target_info["name"],
            "bounding_box": target_info["bounding_box"],
            "center_x": target_info["center_x"],
            "center_y": target_info["center_y"],
            "dynamic_prompt": self.prompt,
            "is_hard": use_hard_mode
        }

        return self.temp_image_path, answer_for_evaluator

    def reset(self):
        self.current_index = 0