import re
import os
import json
import html
import torch
import random
import numpy as np
import torch.nn.functional as F
from reportlab.lib.units import inch
from typing import Any, Dict, Optional
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PlatypusImage,
    PageBreak,
    Table,
    TableStyle,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from PIL import Image as PILImage  # To avoid conflict with ReportLabImage
from qwen_vl_utils import process_vision_info

import evaluator

MAX_COMPLETIONS_PER_PAGE_PDF = 2
MAX_PROMPT_LENGTH_PDF = 300  # Add the missing constant definition
MAX_ANSWER_LENGTH_PDF = 200
MAX_COMPLETION_LENGTH_PDF = 500

####################
## MISC FUNCTIONS ##
####################


def clean_spaces_preserve_newlines(text):
    # Replace multiple spaces with a single space, but preserve newlines
    lines = text.split("\n")  # Split by newlines
    cleaned_lines = [
        " ".join(re.split(r"\s+", line)).strip() for line in lines
    ]  # Remove extra spaces in each line
    return "\n".join(cleaned_lines)  # Join the lines back with newlines


def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility across multiple libraries.

    This function sets consistent random seeds for Python's random module,
    NumPy, PyTorch (both CPU and CUDA), and configures CUDNN for deterministic
    operation. This ensures reproducible results across multiple runs.

    Args:
        seed: The random seed to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Add set_seed (alias for seed_everything)
set_seed = seed_everything


def write_generation_log(log_data: Dict[str, Any], log_file: str) -> None:
    """
    Write generation log data to a text file.

    Args:
        log_data: Dictionary containing prompt and generation data
        log_file: Path to output log file
    """
    with open(log_file, "a") as f:  # Append mode
        f.write(json.dumps(log_data, indent=2) + "\n---\n")  # Add separator


####################################################################################
## Copied Directly from TRL -> generate log probs per token                 ########
## https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py ########
####################################################################################


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(
            logits, index
        ):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
    ).logits
    logits = logits[
        :, :-1, :
    ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(
        logits, input_ids
    )  #  compute logprobs for the input tokens


def get_per_token_logps_vl(
    model, input_ids, attention_mask, image_path, tokenizer, logits_to_keep, prompt
):
    """
    We have the input ids - all the correct tokens including all chate templates/special tokens etc
    We just need to include the image - and have the same sort of obj to pass to the model to generate
    the logits

    So lets generate with the image


    resulting to a very non-generic way to do this - TODO: make this better
    """

    conversation = [
        {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group. You are an expert image analyst.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False, padding_side="left"
    )
    image_inputs, video_inputs = process_vision_info(conversation)

    prompt_inputs = (
        tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )
        .to(model.device)
        .to(model.dtype)
    )

    # Repeat input tensors for batch generation
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(
                input_ids.shape[0], *([1] * (value.dim() - 1))
            )
        else:
            # Handle non-tensor items if necessary, otherwise just copy
            batched_prompt_inputs[key] = value

    batched_prompt_inputs["input_ids"] = input_ids
    batched_prompt_inputs["attention_mask"] = attention_mask

    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    # logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = model(**batched_prompt_inputs).logits
    logits = logits[
        :, :-1, :
    ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(
        logits, input_ids
    )  #  compute logprobs for the input tokens


########################
## PDF/HTML/TEXT STUFF ##
########################

# Constants for PDF generation
PDF_IMAGE_WIDTH = 400  # Max width for images in points (adjust as needed)
PDF_MAX_IMAGE_HEIGHT = 550  # Max height for images in points (adjust as needed)


def _extract_tagged_content(text: str, tag: str) -> Optional[str]:
    """Extracts content within a specific XML-like tag."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _setup_training_log_directory(output_dir: str) -> str:
    """Creates and returns the path for the training log directory."""
    training_log_dir = os.path.join(output_dir, "training_logs")
    os.makedirs(training_log_dir, exist_ok=True)
    return training_log_dir


def _setup_eval_directories(base_output_dir: str) -> tuple[str, str, str]:
    """Creates and returns paths for evaluation log directories (PDF and JSON)."""
    logs_dir = os.path.join(base_output_dir, "eval_logs")
    pdf_dir = os.path.join(logs_dir, "pdfs")
    json_dir = os.path.join(logs_dir, "json_reports")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    return logs_dir, pdf_dir, json_dir


def _setup_pdf(pdf_path: str) -> tuple[SimpleDocTemplate, dict, list]:
    """Initializes the PDF document, styles, and story list."""
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    styles["Code"].fontSize = 10
    styles.add(
        ParagraphStyle(name="Bold", parent=styles["Normal"], fontName="Helvetica-Bold")
    )
    # Add a justified style for longer text blocks
    styles.add(
        ParagraphStyle(name="Justified", parent=styles["Normal"], alignment=TA_JUSTIFY)
    )
    return doc, styles, story


def _truncate_text(text: str, max_length: int) -> str:
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def _add_example_header_to_pdf(
    story: list,
    styles: dict,
    img_or_image_path: str | PILImage.Image,
    prompt_text: str,
    answer_data: Any,
    example_num: int,
    dataset_type: str,
    is_hard: bool = False,
):
    title = f"Example {example_num + 1}"
    if is_hard:
        title += " (Hard Subset)"
    story.append(Paragraph(title, styles["h2"]))

    # Display image if path is valid
    if isinstance(img_or_image_path, PILImage.Image) or (
        isinstance(img_or_image_path, str) and os.path.exists(img_or_image_path)
    ):
        try:
            if isinstance(img_or_image_path, str):
                img = PILImage.open(img_or_image_path)
            else:
                img = img_or_image_path
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            display_width = 2.5 * inch  # Max width for the image in PDF
            display_height = display_width * aspect
            # Cap height to prevent overly tall images
            if display_height > 3.5 * inch:
                display_height = 3.5 * inch
                display_width = display_height / aspect

            story.append(
                PlatypusImage(
                    img_or_image_path, width=display_width, height=display_height
                )
            )
            story.append(Spacer(1, 0.1 * inch))
        except Exception as e:
            story.append(
                Paragraph(
                    f"Error loading image: {img_or_image_path}. Error: {e}",
                    styles["BodyText"],
                )
            )
    else:
        story.append(
            Paragraph(f"Image not found: {img_or_image_path}", styles["BodyText"])
        )

    # Display full prompt without truncation, escape HTML tags
    story.append(Paragraph("<b>Prompt:</b>", styles["BodyText"]))
    escaped_prompt = html.escape(prompt_text).replace("\n", "<br/>")
    story.append(Paragraph(escaped_prompt, styles["Code"]))

    # Display answer/target information based on dataset type
    if dataset_type == "gui":
        target_name = answer_data.get("name", "N/A")
        target_bbox = answer_data.get("bounding_box", "N/A")
        story.append(
            Paragraph(f"<b>Target Object:</b> {target_name}", styles["BodyText"])
        )
        story.append(
            Paragraph(f"<b>Target BBox:</b> {str(target_bbox)}", styles["BodyText"])
        )
    else:  # Clock, Correlation
        story.append(
            Paragraph(
                f"<b>Ground Truth Answer:</b> {_truncate_text(str(answer_data), MAX_ANSWER_LENGTH_PDF)}",
                styles["BodyText"],
            )
        )
    story.append(Spacer(1, 0.2 * inch))


def _add_completion_to_pdf(
    story: list,
    styles: dict,
    completion_text: str,
    metrics: Optional[Dict[str, Any]],
    completion_idx: int,
    dataset_type: str,
    image_path_for_completion_pdf: Optional[str] = None,
):
    story.append(Paragraph(f"Completion {completion_idx + 1}", styles["h3"]))

    # Display image for this completion (e.g., with click for GUI)
    if image_path_for_completion_pdf and os.path.exists(image_path_for_completion_pdf):
        try:
            img = PILImage.open(image_path_for_completion_pdf)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            display_width = 2.0 * inch  # Slightly smaller for completion image
            display_height = display_width * aspect
            if display_height > 3.0 * inch:
                display_height = 3.0 * inch
                display_width = display_height / aspect

            story.append(
                PlatypusImage(
                    image_path_for_completion_pdf,
                    width=display_width,
                    height=display_height,
                )
            )
            story.append(Spacer(1, 0.05 * inch))
        except Exception as e:
            story.append(
                Paragraph(
                    f"Error loading completion image: {image_path_for_completion_pdf}. Error: {e}",
                    styles["Italic"],
                )
            )

    # Display full completion text (escaped)
    story.append(Paragraph("<b>Full Model Output:</b>", styles["BodyText"]))
    escaped_completion = html.escape(completion_text).replace("\n", "<br/>")
    story.append(Paragraph(escaped_completion, styles["Code"]))
    story.append(Spacer(1, 0.05 * inch))

    # Extract and display reasoning and answer separately
    reasoning = _extract_tagged_content(completion_text, "reasoning")
    answer = _extract_tagged_content(completion_text, "answer")

    story.append(Paragraph("<b>Extracted Reasoning:</b>", styles["BodyText"]))
    story.append(
        Paragraph(html.escape(reasoning) if reasoning else "<i>N/A</i>", styles["Code"])
    )
    story.append(Spacer(1, 0.05 * inch))

    story.append(Paragraph("<b>Extracted Answer:</b>", styles["BodyText"]))
    story.append(
        Paragraph(html.escape(answer) if answer else "<i>N/A</i>", styles["Code"])
    )
    story.append(Spacer(1, 0.1 * inch))

    if metrics:
        # Create a more structured table for metrics if many, or simple paragraphs
        metrics_data = [["Metric", "Value"]]
        for name, value in metrics.items():
            # Optionally shorten name for display
            display_name = name.replace("rewards/", "").replace("metrics/", "")
            if isinstance(value, float):
                metrics_data.append([display_name, f"{value:.4f}"])
            else:
                metrics_data.append([display_name, str(value)])

        if len(metrics_data) > 1:
            # Adjusted colWidths: more space for metric name
            table = Table(metrics_data, colWidths=[2.8 * inch, 1.2 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "MIDDLE",
                        ),  # Improve vertical alignment
                        ("WORDWRAP", (0, 1), (0, -1)),  # Allow metric names to wrap
                    ]
                )
            )
            story.append(table)
    story.append(Spacer(1, 0.15 * inch))


def _process_single_completion_for_eval(
    completion_text: str,
    eval_class: evaluator.RewardEvaluator,
    answer_data: Any,  # This is target_details for GUI, answer_str for clock/correlation
    device: str,
    story: Optional[list] = None,  # Make it optional
    styles: Optional[dict] = None,  # Make it optional
    completion_idx: int = 0,
    dataset_type: str = "gui_hard",
) -> Optional[dict[str, float]]:
    """
    Processes a single completion text for evaluation and optionally adds it to a PDF story.
    Returns a dictionary of metric scores for this single completion.
    """
    if dataset_type == "gui" or dataset_type == "gui_hard":
        # For GUI, answer_data is the target_details_dict
        # The evaluator's compute_rewards expects a list of answers.
        # For a single completion, we wrap answer_data in a list.
        current_answers_list = [answer_data]
        # The evaluator also expects completions in a specific nested list structure.
        current_completions_list = [[{"content": completion_text}]]
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported")

    # Get rewards and metrics for this single completion
    # The evaluator's compute_rewards should return rewards_per_func and metrics
    # rewards_per_func will be a tensor for this one completion, e.g., shape [1, num_reward_components]
    # metrics will be a dict like {'metric_name': value_tensor}
    rewards_per_func_single, metrics_single_dict_tensors = eval_class.compute_rewards(
        prompts=None,  # Not always needed by evaluator if context is in answer_data
        completions=current_completions_list,
        answers=current_answers_list,
        device=device,  # device might be used by evaluator internally
    )

    # Convert metric tensors to scalar floats and ensure all are included
    # Also, get the reward breakdown for PDF logging if story is present
    processed_metrics_for_return = {}
    if metrics_single_dict_tensors and isinstance(metrics_single_dict_tensors, dict):
        for k, v_tensor in metrics_single_dict_tensors.items():
            if torch.is_tensor(v_tensor) and v_tensor.numel() == 1:
                processed_metrics_for_return[k] = v_tensor.item()
            elif isinstance(v_tensor, (float, int)):
                processed_metrics_for_return[k] = v_tensor
            # else: might be other types of metrics not directly plottable/averageable

    # Get reward breakdown if needed (e.g., for PDF)
    # rewards_per_func_single should be for one sample, e.g., shape [1, num_reward_components]
    # We need to pass the actual reward scores for this one completion to get_reward_breakdown
    reward_scores_for_breakdown = rewards_per_func_single[
        0
    ]  # Get the tensor for the first (only) sample

    # Add overall reward to the metrics
    total_reward_single = reward_scores_for_breakdown.sum().item()
    processed_metrics_for_return["reward"] = (
        total_reward_single  # Ensure 'reward' key exists
    )

    return processed_metrics_for_return


def _calculate_and_log_final_metrics(
    all_avg_scores: dict, json_dir: str, round_num: int, verbose: bool
):
    """Saves combined average scores (overall, normal, hard) to JSON and prints if verbose."""
    # Save average scores to a JSON file
    avg_scores_path = os.path.join(json_dir, f"average_scores_round_{round_num}.json")
    with open(avg_scores_path, "w") as f:
        # Log all average scores (overall, normal, hard)
        json.dump(all_avg_scores, f, indent=4)

    if verbose:
        print(f"\n--- Evaluation Results (Round {round_num}) ---")
        print(f"Average scores saved to {avg_scores_path}")
        print("Average Scores Breakdown:")
        # Nicely print the different groups if they exist
        print("  --- Overall --- ")
        for name, value in all_avg_scores.items():
            if name.startswith("avg_overall_"):
                print(f"    {name.replace('avg_overall_', ''):<35}: {value:.4f}")
        print("  --- Normal Subset --- ")
        for name, value in all_avg_scores.items():
            if name.startswith("avg_normal_"):
                print(f"    {name.replace('avg_normal_', ''):<35}: {value:.4f}")
        if not any(k.startswith("avg_normal_") for k in all_avg_scores):
            print("    (No normal examples in this evaluation)")
        print("  --- Hard Subset --- ")
        for name, value in all_avg_scores.items():
            if name.startswith("avg_hard_"):
                print(f"    {name.replace('avg_hard_', ''):<35}: {value:.4f}")
        if not any(k.startswith("avg_hard_") for k in all_avg_scores):
            print("    (No hard examples in this evaluation)")
        print("-" * 40)


def _add_training_completion_to_pdf(
    story: list,
    styles: dict,
    completion_text: str,
    reward_breakdown: Dict[str, float],
    advantage: float,
    completion_idx: int,
    image_path_for_completion_pdf: Optional[str] = None,
):
    """Adds details for a single training completion (including scores and advantage) to the PDF story."""
    story.append(Paragraph(f"Training Completion {completion_idx + 1}", styles["h3"]))

    # Display visualized image (e.g., with click for GUI)
    if image_path_for_completion_pdf and os.path.exists(image_path_for_completion_pdf):
        try:
            img = PILImage.open(image_path_for_completion_pdf)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            display_width = 2.0 * inch
            display_height = display_width * aspect
            if display_height > 3.0 * inch:
                display_height = 3.0 * inch
                display_width = display_height / aspect

            story.append(
                PlatypusImage(
                    image_path_for_completion_pdf,
                    width=display_width,
                    height=display_height,
                )
            )
            story.append(Spacer(1, 0.05 * inch))
        except Exception as e:
            story.append(
                Paragraph(
                    f"Error loading training completion image: {image_path_for_completion_pdf}. Error: {e}",
                    styles["Italic"],
                )
            )

    # Display full completion text (escaped)
    story.append(Paragraph("<b>Full Model Output:</b>", styles["BodyText"]))
    escaped_completion = html.escape(completion_text).replace("\n", "<br/>")
    story.append(Paragraph(escaped_completion, styles["Code"]))
    story.append(Spacer(1, 0.05 * inch))

    # Display Reward Breakdown and Advantage
    story.append(Paragraph("<b>Scores & Advantage:</b>", styles["BodyText"]))
    data_table = [["Component", "Value"]]
    total_reward = 0
    if reward_breakdown:
        for name, value in reward_breakdown.items():
            data_table.append([name, f"{value:.4f}"])
            total_reward += value
    data_table.append(["Total Reward (sum)", f"{total_reward:.4f}"])
    data_table.append(["Advantage", f"{advantage:.4f}"])

    table = Table(data_table, colWidths=[2.0 * inch, 1.5 * inch])  # Adjusted width
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("WORDWRAP", (0, 1), (0, -1)),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))
