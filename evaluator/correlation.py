from .base_evaluator import RewardEvaluator
import torch
import re
from typing import Dict, List, Tuple


class CorrelationEvaluator(RewardEvaluator):
    """
    Reward evaluator for the Correlation Scatter Plot estimation task.

    Implements reward functions for:
    - Correlation correctness (based on absolute difference, scaled 0 to 1)
    - X.XX format correctness
    - Strict XML formatting (<reasoning>/<answer> tags)
    """

    def __init__(self):
        self.num_reward_functions = 3  # Correctness, Value Format, XML Format
        # Regex to extract X.XX format (0.00 to 1.00)
        self.correlation_extract_pattern = re.compile(r"\b([01]\.\d{2})\b")
        # Regex for the strict overall XML format (same as clock task)
        self.strict_xml_pattern = re.compile(
            r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\\n</answer>\n$", re.DOTALL
        )

    def _extract_correlation_value(self, text: str) -> float | None:
        """Extract X.XX correlation value from the <answer> tag."""
        try:
            # Isolate content within <answer> tags
            answer_content = text.split("<answer>")[-1].split("</answer>")[0].strip()
            # Match the X.XX pattern within the answer content
            match = self.correlation_extract_pattern.search(answer_content)
            if match:
                # Return the matched correlation value as float
                value = float(match.group(1))
                # Ensure value is within [0.0, 1.0] range (due to regex \b1\.00\b is allowed)
                if 0.0 <= value <= 1.0:
                    return value
            return None
        except (IndexError, ValueError):
            return (
                None  # Tags not found, incorrect structure, or float conversion failed
            )

    def _correlation_format_reward(
        self, extracted_values: List[float | None]
    ) -> List[float]:
        """Reward for having the correct X.XX format extracted.
        Awards 0.5 if the format was successfully extracted, 0 otherwise.
        """
        # The extraction itself validates the format based on the regex and range check
        return [0.5 if value is not None else 0.0 for value in extracted_values]

    def _correctness_reward(
        self, extracted_values: List[float | None], ground_truth_answers: List[str]
    ) -> Tuple[List[float], List[float]]:
        """Reward based on absolute difference, scaled linearly from +1 (0 diff) to 0 (1.0 diff).
        Returns a tuple: (list of reward scores, list of absolute errors)
        """
        rewards = []
        abs_errors = []
        max_reward = 1.0
        min_reward = 0.0
        max_possible_error = 1.0

        for pred_val, true_r_str in zip(extracted_values, ground_truth_answers):
            try:
                true_r = float(true_r_str)  # Ground truth is already "X.XX"
            except ValueError:
                # Should not happen if dataloader is correct
                print(f"Warning: Could not parse ground truth R value: {true_r_str}")
                rewards.append(min_reward)
                abs_errors.append(max_possible_error)
                continue

            if pred_val is None:
                # Prediction is invalid or couldn't be parsed
                rewards.append(min_reward)
                abs_errors.append(
                    max_possible_error
                )  # Assign max error if format is wrong
            else:
                # Both values are valid floats between 0 and 1
                diff = abs(true_r - pred_val)
                abs_errors.append(diff)

                # Scale reward linearly: Reward = MaxReward - (Diff / MaxError) * (MaxReward - MinReward)
                # Since MaxReward=1, MinReward=0, MaxError=1, this simplifies:
                scaled_reward = max_reward - diff
                rewards.append(scaled_reward)

        return rewards, abs_errors

    def _strict_xml_format_reward(
        self, completions: List[List[Dict[str, str]]]
    ) -> List[float]:
        """Reward for strict <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n format."""
        responses = [comp[0]["content"] for comp in completions]
        matches = [bool(self.strict_xml_pattern.match(r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]  # Award 0.5 for correct XML

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answers: List[str],  # Expecting a list of ground truth R strings "X.XX"
        device: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the correlation task."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(
            num_completions, self.num_reward_functions, device=device
        )

        # Extract predicted correlation values
        extracted_values = [
            self._extract_correlation_value(comp[0]["content"]) for comp in completions
        ]

        # Compute reward components
        correctness_scores, abs_error_values = self._correctness_reward(
            extracted_values, answers
        )
        correlation_format_scores = self._correlation_format_reward(extracted_values)
        xml_format_scores = self._strict_xml_format_reward(completions)

        all_scores = [
            correctness_scores,  # Scaled 0-1
            correlation_format_scores,  # 0 or 0.5
            xml_format_scores,  # 0 or 0.5
        ]

        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(
                scores, dtype=torch.float32, device=device
            )

        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        abs_error_tensor = torch.tensor(
            abs_error_values, dtype=torch.float32, device=device
        )
        mean_abs_error = abs_error_tensor.mean().item()

        # Total reward is sum of components (max possible is 1.0 + 0.5 + 0.5 = 2.0)
        total_reward_mean = rewards_per_func.sum(dim=1).mean().item()

        metrics = {
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/correlation_format_reward_func": reward_per_func[1].item(),
            "rewards/strict_xml_format_reward_func": reward_per_func[2].item(),
            "reward": total_reward_mean,  # Total reward mean
            "metrics/mean_abs_correlation_error": mean_abs_error,
        }
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        # Ensure reward_scores is a 1D tensor with expected length
        if reward_scores.ndim == 1 and len(reward_scores) == self.num_reward_functions:
            return {
                "correctness": reward_scores[0].item(),
                "correlation_format": reward_scores[1].item(),
                "strict_xml_format": reward_scores[2].item(),
            }
        elif (
            reward_scores.ndim == 2
            and reward_scores.shape[1] == self.num_reward_functions
        ):
            # Handle batch tensor case (return first item's breakdown)
            print(
                "Warning: get_reward_breakdown received batch tensor, returning breakdown for first item."
            )
            return {
                "correctness": reward_scores[0, 0].item(),
                "correlation_format": reward_scores[0, 1].item(),
                "strict_xml_format": reward_scores[0, 2].item(),
            }
        else:
            print(
                f"Warning: Unexpected shape for reward_scores in get_reward_breakdown: {reward_scores.shape}"
            )
            # Return default/empty breakdown
            return {
                "correctness": 0.0,
                "correlation_format": 0.0,
                "strict_xml_format": 0.0,
            }
