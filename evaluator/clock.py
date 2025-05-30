# Maximum possible time difference on a 12-hour clock in seconds (6 hours)
MAX_DIFF_SECONDS = 6 * 3600

from .base_evaluator import RewardEvaluator
from rldatasets.clock.clock_generator import TimeObj
import torch


import re
from typing import Dict, List, Tuple


class ClockEvaluator(RewardEvaluator):
    """
    Reward evaluator for the Analog Clock time prediction task.

    Implements reward functions for:
    - Time correctness (based on seconds difference, scaled from +3 to -3)
    - HH:MM:SS format correctness
    - Strict XML formatting (<reasoning>/<answer> tags)
    """

    def __init__(self, accuracy_tolerance_seconds: int = 60):
        self.num_reward_functions = 3  # Correctness, Time Format, XML Format
        self.accuracy_tolerance_seconds = accuracy_tolerance_seconds
        # Regex to extract HH:MM:SS, ensuring it's not part of a larger number
        self.time_extract_pattern = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")
        # Regex for the strict overall XML format
        self.strict_xml_pattern = re.compile(
            r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$", re.DOTALL
        )

    def _extract_time_string(self, text: str) -> str | None:
        """Extract HH:MM:SS time string from the <answer> tag."""
        try:
            # Isolate content within <answer> tags
            answer_content = text.split("<answer>")[-1].split("</answer>")[0].strip()
            # Match the HH:MM:SS pattern within the answer content
            match = self.time_extract_pattern.search(answer_content)
            if match:
                # Return the matched time string directly
                return match.group(0)  # Return the exact match found
            return None
        except IndexError:
            return None  # Tags not found or incorrect structure

    def _time_format_reward(self, extracted_times: List[str | None]) -> List[float]:
        """Reward for having the correct [HH:MM:SS] format extracted.
        Awards 0.5 if the format was successfully extracted, 0 otherwise.
        """
        # The extraction itself validates the format based on the regex
        return [0.5 if time_str is not None else 0.0 for time_str in extracted_times]

    def _correctness_reward(
        self, extracted_times: List[str | None], ground_truth_answers: List[str]
    ) -> Tuple[List[float], List[float]]:
        """Reward based on time difference in seconds, scaled from +3 to -3.
        Returns a tuple: (list of reward scores, list of absolute errors in seconds)
        """
        rewards = []
        abs_errors = []
        max_reward = 3.0
        min_reward = -3.0

        for pred_time_str, true_time_str in zip(extracted_times, ground_truth_answers):
            true_time_obj = TimeObj.from_string(true_time_str)
            pred_time_obj = (
                TimeObj.from_string(pred_time_str) if pred_time_str else None
            )

            if pred_time_obj is None:
                # Prediction is invalid or couldn't be parsed
                rewards.append(min_reward)
                abs_errors.append(
                    float(MAX_DIFF_SECONDS)
                )  # Assign max error if format is wrong
            else:
                # Both times are valid, calculate difference
                diff_seconds, _ = true_time_obj.subtract(pred_time_obj)
                abs_errors.append(float(diff_seconds))

                # Scale reward linearly from +3 (0 diff) to -3 (MAX_DIFF_SECONDS diff)
                scaled_reward = max_reward - (max_reward - min_reward) * (
                    diff_seconds / MAX_DIFF_SECONDS
                )
                rewards.append(scaled_reward)

        return rewards, abs_errors

    def _strict_xml_format_reward(
        self, completions: List[List[Dict[str, str]]]
    ) -> List[float]:
        """Reward for strict <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n format."""
        responses = [comp[0]["content"] for comp in completions]
        matches = [bool(self.strict_xml_pattern.match(r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answers: List[
            str
        ],  # Expecting a list of ground truth time strings "[HH:MM:SS]"
        device: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the clock task."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(
            num_completions, self.num_reward_functions, device=device
        )

        # Extract predicted time strings
        extracted_times = [
            self._extract_time_string(comp[0]["content"]) for comp in completions
        ]

        # Compute reward components
        correctness_scores, abs_error_seconds = self._correctness_reward(
            extracted_times, answers
        )
        time_format_scores = self._time_format_reward(extracted_times)
        xml_format_scores = self._strict_xml_format_reward(completions)

        all_scores = [correctness_scores, time_format_scores, xml_format_scores]

        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(
                scores, dtype=torch.float32, device=device
            )

        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        abs_error_tensor = torch.tensor(
            abs_error_seconds, dtype=torch.float32, device=device
        )
        mean_abs_error = abs_error_tensor.mean().item()

        # Calculate accuracy (within tolerance)
        num_accurate = (
            (abs_error_tensor <= self.accuracy_tolerance_seconds).sum().item()
        )
        accuracy = num_accurate / num_completions if num_completions > 0 else 0.0

        mean_abs_error_minutes = mean_abs_error / 60.0
        mean_abs_error_hours = mean_abs_error / 3600.0

        metrics = {
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/time_format_reward_func": reward_per_func[1].item(),
            "rewards/strict_xml_format_reward_func": reward_per_func[2].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),  # Total reward mean
            "metrics/mean_abs_error_seconds": mean_abs_error,
            "metrics/mean_abs_error_minutes": mean_abs_error_minutes,
            "metrics/mean_abs_error_hours": mean_abs_error_hours,
            "metrics/accuracy": accuracy,  # Accuracy within tolerance
        }
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        # Ensure reward_scores is a 1D tensor with expected length
        if reward_scores.ndim == 1 and len(reward_scores) == self.num_reward_functions:
            return {
                "correctness": reward_scores[0].item(),
                "time_format": reward_scores[1].item(),
                "strict_xml_format": reward_scores[2].item(),
            }
        elif (
            reward_scores.ndim == 2
            and reward_scores.shape[1] == self.num_reward_functions
        ):
            # If passed the whole batch tensor, return breakdown for the first element
            # Or consider averaging? Returning first for now.
            print(
                "Warning: get_reward_breakdown received batch tensor, returning breakdown for first item."
            )
            return {
                "correctness": reward_scores[0, 0].item(),
                "time_format": reward_scores[0, 1].item(),
                "strict_xml_format": reward_scores[0, 2].item(),
            }
        else:
            print(
                f"Warning: Unexpected shape for reward_scores in get_reward_breakdown: {reward_scores.shape}"
            )
            # Return default/empty breakdown
            return {
                "correctness": 0.0,
                "time_format": 0.0,
                "strict_xml_format": 0.0,
            }
