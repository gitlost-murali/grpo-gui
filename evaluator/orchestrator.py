from .base_evaluator import RewardEvaluator
import rldatasets as rldatasets
import utils
from llms import generate_completions
from rldatasets.gui.gui_generator import GUIGenerator


import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


import argparse
import os
from collections import defaultdict


def eval_on_test_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_loader: rldatasets.DataLoader,
    eval_class: RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int,
) -> tuple[dict[str, float], float]:
    """
    Evaluate model performance on test set with improved logging and PDF reports.
    (Orchestrates evaluation using helper functions).
    Returns tuple: (aggregated_avg_scores_dict, main_error_metric_value)
    """
    print("Running evaluation on test set...")

    # Initialize accumulators for overall, normal, and hard subsets
    num_examples = 0
    num_chains_processed_overall = 0
    num_chains_processed_normal = 0
    num_chains_processed_hard = 0
    aggregated_metrics_sum_overall = defaultdict(float)
    aggregated_metrics_sum_normal = defaultdict(float)
    aggregated_metrics_sum_hard = defaultdict(float)

    # Ensure temp_vis directory exists for saving images with clicks for PDF
    eval_temp_vis_dir = os.path.join(args.output_dir, "eval_logs", "temp_vis")
    os.makedirs(eval_temp_vis_dir, exist_ok=True)

    _, pdf_dir, json_dir = utils._setup_eval_directories(args.output_dir)
    # Sanitize experiment name for PDF filename (if present in args.output_dir)
    exp_name_sanitized = os.path.basename(os.path.normpath(args.output_dir)).replace(
        " ", "_"
    )
    pdf_filename = f"eval_results_round_{round_num}_exp_{exp_name_sanitized}.pdf"
    pdf_path = os.path.join(pdf_dir, pdf_filename)
    doc, styles, story = utils._setup_pdf(pdf_path)

    test_loader.reset()

    # Determine the correct error metric key based on dataset type
    if args.dataset_type == "correlation":
        main_error_metric_key_for_return = "metrics/mean_abs_correlation_error"
    elif args.dataset_type == "clock":
        main_error_metric_key_for_return = "metrics/mean_abs_error_seconds"
    elif args.dataset_type == "gui":
        main_error_metric_key_for_return = (
            "metrics/mean_distance_to_center_error"  # Or click_hit_rate if preferred
        )
    else:
        main_error_metric_key_for_return = "reward"  # Fallback, should be defined

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating on test set")):
        is_hard_example = False  # Default for non-GUI tasks
        if args.dataset_type == "gui":
            img_path, target_details_dict = batch
            prompt_to_use = target_details_dict["dynamic_prompt"]
            answer_for_eval_and_pdf = target_details_dict
            is_hard_example = target_details_dict.get("is_hard", False)  # Get the flag
        else:
            img_path, answer_string = batch
            prompt_to_use = test_loader.prompt  # Static prompt for clock/correlation
            answer_for_eval_and_pdf = answer_string
        # Generate completions
        # Ensure generate_completions uses the correct prompt (dynamic for GUI)
        _, _, _, _, completions_text, _ = generate_completions(
            model, tokenizer, img_path, prompt_to_use, device, args, eval=True
        )

        # Add example header to PDF (prompt_to_use will have target name for GUI)
        utils._add_example_header_to_pdf(
            story,
            styles,
            img_path,
            prompt_to_use,
            answer_for_eval_and_pdf,
            num_examples,
            args.dataset_type,
            is_hard=is_hard_example,
        )

        for completion_idx, completion_text in enumerate(completions_text):
            # Path for saving image with plotted click (unique per completion)
            vis_image_path_for_pdf = os.path.join(
                eval_temp_vis_dir,
                f"round{round_num}_ex{batch_idx}_comp{completion_idx}.png",
            )

            # Process single completion also handles plotting for GUI task
            metrics_single = utils._process_single_completion_for_eval(
                completion_text=completion_text,
                eval_class=eval_class,
                answer_data=answer_for_eval_and_pdf,  # This is target_details for GUI
                device=device,
                story=story,
                styles=styles,
                completion_idx=completion_idx,
                dataset_type=args.dataset_type,
                original_image_path=img_path,  # Pass original image path for GUI task
                vis_image_path_for_pdf=vis_image_path_for_pdf,  # Path to save visualization
                gui_plotter=GUIGenerator.plot_predictions
                if args.dataset_type == "gui"
                else None,
            )

            num_chains_processed_overall += 1
            if metrics_single and isinstance(metrics_single, dict):
                # Aggregate overall metrics
                for metric_name, value in metrics_single.items():
                    if isinstance(value, (int, float)):
                        aggregated_metrics_sum_overall[metric_name] += value
                    elif torch.is_tensor(value) and value.numel() == 1:
                        aggregated_metrics_sum_overall[metric_name] += value.item()

                # Aggregate normal/hard metrics
                if is_hard_example:
                    num_chains_processed_hard += 1
                    for metric_name, value in metrics_single.items():
                        if isinstance(value, (int, float)):
                            aggregated_metrics_sum_hard[metric_name] += value
                        elif torch.is_tensor(value) and value.numel() == 1:
                            aggregated_metrics_sum_hard[metric_name] += value.item()
                else:
                    num_chains_processed_normal += 1
                    for metric_name, value in metrics_single.items():
                        if isinstance(value, (int, float)):
                            aggregated_metrics_sum_normal[metric_name] += value
                        elif torch.is_tensor(value) and value.numel() == 1:
                            aggregated_metrics_sum_normal[metric_name] += value.item()

        num_examples += 1

    # --- Calculate final averages ---
    avg_scores_overall = {}
    avg_scores_normal = {}
    avg_scores_hard = {}

    if num_chains_processed_overall > 0:
        for metric_name, total_value in aggregated_metrics_sum_overall.items():
            avg_scores_overall[f"avg_overall_{metric_name}"] = (
                total_value / num_chains_processed_overall
            )
    if num_chains_processed_normal > 0:
        for metric_name, total_value in aggregated_metrics_sum_normal.items():
            avg_scores_normal[f"avg_normal_{metric_name}"] = (
                total_value / num_chains_processed_normal
            )
    if num_chains_processed_hard > 0:
        for metric_name, total_value in aggregated_metrics_sum_hard.items():
            avg_scores_hard[f"avg_hard_{metric_name}"] = (
                total_value / num_chains_processed_hard
            )

    # Combine all averages into one dictionary for logging
    all_avg_scores = {**avg_scores_overall, **avg_scores_normal, **avg_scores_hard}

    # Get the overall average value of the primary error metric for return
    final_avg_main_metric_value = avg_scores_overall.get(
        f"avg_overall_{main_error_metric_key_for_return}", float("nan")
    )

    doc.build(story)
    print(f"PDF report saved to {pdf_path}")

    utils._calculate_and_log_final_metrics(
        all_avg_scores, json_dir, round_num, args.verbose
    )

    return (
        all_avg_scores,
        final_avg_main_metric_value,
    )  # Return all avg scores and the specific main error metric value
