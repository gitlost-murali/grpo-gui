"""
reward computation in RL training.

"""

import math
import numpy as np  # Added numpy
from typing import Optional

from .base_evaluator import RewardEvaluator
from .clock import ClockEvaluator
from .correlation import CorrelationEvaluator
from .gui import GUIEvaluator


def get_evaluator(name: str) -> RewardEvaluator:
    """
    Get the appropriate reward evaluator for a given task.

    Args:
        name: Name of the task/dataset to get evaluator for

    Returns:
        RewardEvaluator instance for the specified task

    Raises:
        NotImplementedError: If evaluator for given task is not implemented
    """
    if name.lower() == "clock":
        return ClockEvaluator()
    elif name.lower() == "correlation":
        return CorrelationEvaluator()
    elif name.lower() == "gui":
        return GUIEvaluator()
    else:
        raise NotImplementedError(
            f"No evaluator implemented for {name}. Supported: 'clock', 'correlation', 'gui'"
        )
