import math
from evaluator.base_evaluator import RewardEvaluator
from rldatasets.gui.gui_generator import IMAGE_HEIGHT, IMAGE_WIDTH

MAX_POSSIBLE_DISTANCE_GUI = math.sqrt(IMAGE_WIDTH**2 + IMAGE_HEIGHT**2) # Diagonal of the image

import torch


import math
import re
from typing import Any, Dict, List, Optional, Tuple


class GUIEvaluator(RewardEvaluator):
    """
    Reward evaluator for the GUI Interaction (click prediction) task.

    Implements reward functions for:
    - Strict XML formatting (<reasoning>/<answer>x,y</answer>)
    - Click Hit (whether the click is within the target bounding box)
    - Distance to Center (Euclidean distance from click to target center, scaled)
    """

    def __init__(self):
        self.num_reward_functions = 3 # XML Format, Click Hit, Distance to Center
        # Regex to extract "x,y" coordinates, allowing for spaces around comma
        self.coord_extract_pattern = re.compile(r"(\d+)\s*,\s*(\d+)")
        # Regex for the strict overall XML format, ensuring x,y in answer
        # This pattern is more specific for the answer part to guide the LLM.
        self.strict_xml_pattern = re.compile(r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n\d+\s*,\s*\d+\n</answer>\n$", re.DOTALL)
        # Define names of known circular elements
        self.circular_elements = {"window_close_button", "window_minimize_button", "window_maximize_button"}

    def _extract_coordinates(self, text: str) -> Optional[Tuple[int, int]]:
        """Extract x,y coordinates from the <answer> tag."""
        try:
            answer_content = text.split("<answer>")[-1].split("</answer>")[0].strip()
            match = self.coord_extract_pattern.search(answer_content)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                # Basic check for coordinates being within typical image bounds (e.g., 0-1024, can be refined based on actual image size if needed)
                # For now, GUIGenerator uses 224x224. We can add stricter checks if x/y are way off.
                if 0 <= x <= IMAGE_WIDTH*2 and 0 <= y <= IMAGE_HEIGHT*2: # Allow some leeway beyond 224 for robustness
                    return x, y
            return None
        except (IndexError, ValueError):
            return None # Tags not found, incorrect structure, or int conversion failed

    def _is_click_in_bbox(self, click_xy: Optional[Tuple[int, int]], target_bbox: Tuple[int, int, int, int], target_name: Optional[str] = None) -> bool:
        """Check if the click (x,y) is within the target area (bbox or circle radius)."""
        if click_xy is None:
            return False
        x, y = click_xy
        x_min, y_min, x_max, y_max = target_bbox

        # Check for known circular elements
        if target_name and target_name in self.circular_elements:
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            radius = (x_max - x_min) / 2 # Assume bbox tightly bounds the circle
            distance_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            return distance_from_center <= radius
        else:
            # Default rectangular bounding box check
            return x_min <= x <= x_max and y_min <= y <= y_max

    def _strict_xml_format_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """Reward for strict <reasoning>...</reasoning><answer>x,y</answer> format."""
        responses = [comp[0]['content'] for comp in completions]
        # Check overall structure and if coordinates can be extracted (implies x,y format in answer is somewhat met)
        rewards = []
        for r in responses:
            xml_match = bool(self.strict_xml_pattern.match(r))
            coords_extracted = self._extract_coordinates(r) is not None
            if xml_match and coords_extracted:
                rewards.append(0.5)
            elif coords_extracted and not xml_match: # Has x,y but not full XML structure
                rewards.append(0.1) # Small partial credit for at least getting coordinates
            else:
                rewards.append(0.0)
        return rewards

    def _click_hit_reward(self, extracted_coords: List[Optional[Tuple[int, int]]],
                          target_bboxes: List[Tuple[int, int, int, int]],
                          target_names: List[str]) -> List[float]:
        """Reward +3 if click is inside target area (bbox or circle), 0 otherwise."""
        rewards = []
        for click_xy, bbox, name in zip(extracted_coords, target_bboxes, target_names):
            # Pass target_name to the check function
            if self._is_click_in_bbox(click_xy, bbox, name):
                rewards.append(3.0)
            else:
                rewards.append(0.0)
        return rewards

    def _distance_to_center_reward(self, extracted_coords: List[Optional[Tuple[int, int]]],
                                   target_centers: List[Tuple[int,int]],
                                   target_bboxes: List[Tuple[int,int,int,int]]) -> Tuple[List[float], List[float]]:
        """Scaled reward based on Euclidean distance to target center (+2 to -2).
           Returns rewards and the raw distance errors.
        """
        rewards = []
        distance_errors = [] # Store raw distance errors for metrics

        max_reward = 2.0
        min_reward = -2.0 # Furthest possible click, or unparseable

        for click_xy, target_center_xy, target_bbox in zip(extracted_coords, target_centers, target_bboxes):
            if click_xy is None: # Coordinate couldn't be parsed
                rewards.append(min_reward)
                distance_errors.append(MAX_POSSIBLE_DISTANCE_GUI) # Max possible error
                continue

            pred_x, pred_y = click_xy
            true_center_x, true_center_y = target_center_xy

            distance = math.sqrt((pred_x - true_center_x)**2 + (pred_y - true_center_y)**2)
            distance_errors.append(distance)

            # Scale reward: +2 for 0 distance, down to -2 for MAX_POSSIBLE_DISTANCE_GUI
            # Reward = MaxReward - (Distance / MaxDistance) * (MaxReward - MinReward)
            # Ensure distance doesn't exceed MAX_POSSIBLE_DISTANCE_GUI for scaling
            clamped_distance = min(distance, MAX_POSSIBLE_DISTANCE_GUI)
            scaled_reward = max_reward - (clamped_distance / MAX_POSSIBLE_DISTANCE_GUI) * (max_reward - min_reward)
            rewards.append(scaled_reward)

        return rewards, distance_errors

    def compute_rewards(
        self,
        prompts: Optional[List[List[Dict[str, str]]]], # Prompts may not be directly used here but are part of API
        completions: List[List[Dict[str, str]]],
        answers: List[Dict[str, Any]], # List of target_info dicts from GUIDataLoader
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the GUI click task."""
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        extracted_coords_list = [self._extract_coordinates(comp[0]['content']) for comp in completions]
        target_bboxes_list = [ans['bounding_box'] for ans in answers]
        target_centers_list = [(ans['center_x'], ans['center_y']) for ans in answers]
        target_names_list = [ans['name'] for ans in answers] # Get target names

        # 1. XML Format Reward
        xml_format_scores = self._strict_xml_format_reward(completions)

        # 2. Click Hit Reward (pass target names)
        click_hit_scores = self._click_hit_reward(extracted_coords_list, target_bboxes_list, target_names_list)

        # 3. Distance to Center Reward
        dist_rewards_scores, raw_distance_errors = self._distance_to_center_reward(extracted_coords_list, target_centers_list, target_bboxes_list)

        all_component_scores = [
            xml_format_scores,
            click_hit_scores,
            dist_rewards_scores
        ]

        for i, scores_component in enumerate(all_component_scores):
            rewards_per_func[:, i] = torch.tensor(scores_component, dtype=torch.float32, device=device)

        # --- Metrics --- 
        # Mean for each reward component
        mean_rewards_per_component = rewards_per_func.mean(dim=0)

        # Click Hit Rate (Accuracy)
        num_hits = sum(1 for score in click_hit_scores if score > 0) # count non-zero scores
        click_hit_rate = num_hits / num_completions if num_completions > 0 else 0.0

        # Mean Distance Error
        distance_errors_tensor = torch.tensor(raw_distance_errors, dtype=torch.float32, device=device)
        mean_dist_error = distance_errors_tensor.mean().item()

        # Total reward mean
        total_reward_mean = rewards_per_func.sum(dim=1).mean().item()

        metrics = {
            "rewards/xml_format_reward": mean_rewards_per_component[0].item(),
            "rewards/click_hit_reward": mean_rewards_per_component[1].item(),
            "rewards/distance_to_center_reward": mean_rewards_per_component[2].item(),
            "reward": total_reward_mean, # Overall mean reward
            "metrics/click_hit_rate": click_hit_rate,
            "metrics/mean_distance_to_center_error": mean_dist_error
        }
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        if reward_scores.ndim == 1 and len(reward_scores) == self.num_reward_functions:
            return {
                'xml_format': reward_scores[0].item(),
                'click_hit': reward_scores[1].item(),
                'distance_to_center': reward_scores[2].item(),
            }
        elif reward_scores.ndim == 2 and reward_scores.shape[1] == self.num_reward_functions:
            # For batch tensor, return for the first item (or mean if preferred)
            return {
                'xml_format': reward_scores[0, 0].item(),
                'click_hit': reward_scores[0, 1].item(),
                'distance_to_center': reward_scores[0, 2].item(),
            }
        else:
            # print(f"Warning: Unexpected shape for reward_scores in GUIEvaluator.get_reward_breakdown: {reward_scores.shape}")
            return {'xml_format': 0.0, 'click_hit': 0.0, 'distance_to_center': 0.0}