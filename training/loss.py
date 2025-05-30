import os
import evaluator
from llms import generate_completions
from shutil import copyfile
from typing import Any
import utils


import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


import argparse


def compute_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel,
    prompt_completion_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    args: argparse.Namespace,
    img_path: str,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the GRPO loss between current and base model.

    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        prompt_completion_ids: Combined prompt and completion token IDs
        prompt_ids: Token IDs for just the prompt
        completion_ids: Token IDs for just the completion
        attention_mask: Attention mask for the full sequence
        completion_mask: Mask indicating which tokens are from the completion
        advantages: Advantage values for each sequence
        args: Training arguments

    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing additional metrics like KL divergence
    """

    # Only need the generated tokens' logits
    logits_to_keep = completion_ids.size(1)

    # Get reference model logits
    with torch.inference_mode():
        ref_per_token_logps = utils.get_per_token_logps_vl(
            base_model,
            prompt_completion_ids,
            attention_mask,
            img_path,
            tokenizer,
            logits_to_keep,
            prompt,
        )

    # Get training model logits
    per_token_logps = utils.get_per_token_logps_vl(
        model,
        prompt_completion_ids,
        attention_mask,
        img_path,
        tokenizer,
        logits_to_keep,
        prompt,
    )

    # Compute KL divergence
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps)
        - (ref_per_token_logps - per_token_logps)
        - 1
    )

    # Compute loss with advantages
    per_token_loss = torch.exp(
        per_token_logps - per_token_logps.detach()
    ) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - args.kl_weight_beta * per_token_kl)
    loss = (
        (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
    ).mean()

    # Additional metrics
    metrics = {}
    response_length = completion_mask.sum(1).float().mean().item()
    metrics["response_length"] = response_length
    mean_kl = (
        (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
    ).mean()
    metrics["kl"] = mean_kl.item()

    return loss, metrics


def score_completions(
    completions_text: list[str],
    prompt_data: Any,  # For GUI, this will be target_details_dict including dynamic_prompt
    # For Clock/Corr, this will be the static prompt string
    image_path: str,
    answer_data: Any,  # For GUI, this is target_details_dict (name, bbox, etc.)
    # For Clock/Corr, this is the answer string (time/R value)
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict]:
    """
    Score model completions and compute advantages for training.
    """
    log_data = {"prompt": {}, "generations": []}

    if args.dataset_type == "gui":
        # prompt_data is target_details_dict from GUIDataLoader
        # answer_data is also target_details_dict
        log_data["prompt"] = {
            "text": prompt_data["dynamic_prompt"],  # The actual prompt shown to LLM
            "target_object_name": answer_data["name"],
            "target_object_bbox": answer_data["bounding_box"],
            "image_path": image_path,
        }
        # The `answers` expected by GUIEvaluator.compute_rewards is a list of these answer_data dicts
        eval_answers_list = [answer_data] * len(completions_text)
    else:
        # prompt_data is the static prompt string
        # answer_data is the single answer string (time or R value)
        log_data["prompt"] = {
            "text": prompt_data,
            "answer": answer_data,
            "image_path": image_path,
        }
        eval_answers_list = [answer_data] * len(completions_text)

    # Format inputs as expected by evaluator
    mock_completions_for_eval = [
        [{"content": completion}] for completion in completions_text
    ]

    rewards_per_func, metrics = eval_class.compute_rewards(
        prompts=None,  # Eval class might not need full prompt structure, depends on its impl.
        completions=mock_completions_for_eval,
        answers=eval_answers_list,  # Pass list of answers (target_details for GUI, string for others)
        device=device,
    )

    rewards = rewards_per_func.sum(dim=1)

    for i, (completion, reward_scores_single_chain) in enumerate(
        zip(completions_text, rewards_per_func)
    ):
        generation_data = {
            "response": completion,
            "scores": {
                **eval_class.get_reward_breakdown(
                    reward_scores_single_chain
                ),  # Pass scores for one chain
                "total_reward": rewards[i].item(),
            },
        }
        log_data["generations"].append(generation_data)

    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
        args.num_chains, dim=0
    )
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = (
        std_grouped_rewards.mean().item() if args.num_chains > 1 else 0.0
    )

    log_data["summary_stats"] = {
        "mean_rewards_per_group": mean_grouped_rewards.tolist(),
        "std_rewards_per_group": std_grouped_rewards.tolist(),
        "advantages": advantages.tolist(),
    }
    return rewards, advantages, rewards_per_func, metrics, log_data


def grpo_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    # For GUI, prompt_info_or_static_prompt will be target_details_dict
    # For Clock/Corr, it will be the static prompt string from the loader
    prompt_info_or_static_prompt: Any,
    img_path: str,
    # For GUI, answer_details_or_string will be target_details_dict
    # For Clock/Corr, it will be the answer string
    answer_details_or_string: Any,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    round_num: int,
    training_log_dir: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], list[str], torch.Tensor, torch.Tensor]:
    """
    Compute GRPO loss for a batch.
    Returns: loss, metrics, completions_text, rewards_per_func, advantages
    """
    if args.dataset_type == "gui":
        # prompt_info_or_static_prompt is target_details_dict
        # answer_details_or_string is also target_details_dict
        current_prompt_text = prompt_info_or_static_prompt["dynamic_prompt"]
        current_answer_data = (
            answer_details_or_string  # This is the dict for GUIEvaluator
        )
    else:
        current_prompt_text = prompt_info_or_static_prompt
        current_answer_data = (
            answer_details_or_string  # This is the string for Clock/Corr
        )

    (
        prompt_completion_ids,
        prompt_ids,
        completion_ids,
        attention_mask,
        completions_text,
        _,
    ) = generate_completions(
        model, tokenizer, img_path, current_prompt_text, device, args
    )

    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text,
        prompt_info_or_static_prompt,  # Pass the original prompt info (dict for GUI, str for others)
        img_path,
        current_answer_data,  # Pass target_details for GUI, answer_str for others
        eval_class,
        device,
        args,
    )

    log_file_name = f"round_{round_num}_batch_generations.txt"  # More descriptive name
    log_file = os.path.join(training_log_dir, log_file_name)
    utils.write_generation_log(log_data, log_file)

    image_log_dir = os.path.join(training_log_dir, "images")
    os.makedirs(image_log_dir, exist_ok=True)
    image_log_path = os.path.join(
        image_log_dir, f"image_round_{round_num}.png"
    )  # One image per round for now
    if not os.path.exists(
        image_log_path
    ):  # Save image only if it hasn't been saved for this round yet
        copyfile(img_path, image_log_path)

    completion_mask = attention_mask[:, prompt_ids.size(1) :]
    # Ensure current_prompt_text is used for get_per_token_logps_vl if it relies on it
    loss, loss_metrics = compute_loss(
        model,
        base_model,
        prompt_completion_ids,
        prompt_ids,
        completion_ids,
        attention_mask,
        completion_mask,
        advantages,
        args,
        img_path,
        tokenizer,
        current_prompt_text,
    )
    metrics.update(loss_metrics)

    # Return the required values
    return loss, metrics, completions_text, rewards_per_func, advantages
