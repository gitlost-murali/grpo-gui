import re
import os
import json
import html
import torch
import random
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from reportlab.lib.units import inch
from typing import Any, Dict, Optional
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_JUSTIFY

from qwen_vl_utils import process_vision_info

import evaluator

####################
## MISC FUNCTIONS ##
####################

def clean_spaces_preserve_newlines(text):
    # Replace multiple spaces with a single space, but preserve newlines
    lines = text.split("\n")  # Split by newlines
    cleaned_lines = [" ".join(re.split(r"\s+", line)).strip() for line in lines]  # Remove extra spaces in each line
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
    with open(log_file, 'w') as f:
        # Write prompt section
        f.write("###### ORIGINAL PROMPT #####\n\n")
        f.write(log_data['prompt']['text'] + "\n\n")
        f.write("#### ANS ####\n\n")
        f.write(str(log_data['prompt']['answer']) + "\n")

        # Write each generation
        for i, gen in enumerate(log_data['generations'], 1):
            f.write(f"#### GENERATION {i} RESPONSE ####\n\n")
            f.write(gen['response'] + "\n\n")
            f.write(f"#### GENERATION {i} SCORES ####\n")
            
            # Write all available scores dynamically
            # Check if scores exist and are a dictionary
            if 'scores' in gen and isinstance(gen['scores'], dict):
                for score_name, score_value in gen['scores'].items():
                    # Format score names nicely (optional but good)
                    formatted_name = score_name.replace('_', ' ').capitalize()
                    try:
                        # Attempt to format as float (e.g., total_reward), fallback to string
                        f.write(f"{formatted_name}: {float(score_value):.4f}\n") 
                    except (ValueError, TypeError):
                         f.write(f"{formatted_name}: {score_value}\n") # Fallback for non-numeric scores
            else:
                f.write("No scores available for this generation.\n") # Handle case where scores might be missing

            f.write("\n") # Add a newline after scores for better separation


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
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

def get_per_token_logps_vl(model, input_ids, attention_mask, image_path, tokenizer, logits_to_keep, prompt):
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
                {"type": "text", "text": prompt}

            ],
        },
    ]

    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False, padding_side="left")  
    image_inputs, video_inputs = process_vision_info(conversation)

    prompt_inputs = tokenizer(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        padding_side="left"
    ).to(model.device).to(model.dtype)

    # Repeat input tensors for batch generation
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(input_ids.shape[0], *([1] * (value.dim() - 1)))
        else:
            # Handle non-tensor items if necessary, otherwise just copy
            batched_prompt_inputs[key] = value 

    batched_prompt_inputs["input_ids"] = input_ids
    batched_prompt_inputs["attention_mask"] = attention_mask

    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    # logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = model(**batched_prompt_inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens


########################
## PDF/HTML/TEXT STUFF ##
########################

# Constants for PDF generation
PDF_IMAGE_WIDTH = 400  # Max width for images in points (adjust as needed)
PDF_MAX_IMAGE_HEIGHT = 550 # Max height for images in points (adjust as needed)

def _extract_tagged_content(text: str, tag: str) -> Optional[str]:
    """Extracts content within a specific XML-like tag."""
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def _setup_training_log_directory(output_dir: str) -> str:
    """Creates and returns the path for the training log directory."""
    training_log_dir = os.path.join(output_dir, "training_logs")
    os.makedirs(training_log_dir, exist_ok=True)
    return training_log_dir

def _setup_eval_directories(output_dir: str) -> tuple[str, str, str]:
    """Creates and returns paths for evaluation log directories (PDF and JSON)."""
    eval_logs_dir = os.path.join(output_dir, "eval_logs")
    pdf_dir = os.path.join(eval_logs_dir, "pdfs")
    json_dir = os.path.join(eval_logs_dir, "json")
    for dir_path in [pdf_dir, json_dir]:
        os.makedirs(dir_path, exist_ok=True)
    return eval_logs_dir, pdf_dir, json_dir

def _setup_pdf(pdf_path: str) -> tuple[SimpleDocTemplate, dict, list]:
    """Initializes the PDF document, styles, and story list."""
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [] 
    styles['Code'].fontSize = 10 
    styles.add(ParagraphStyle(name='Bold', parent=styles['Normal'], fontName='Helvetica-Bold'))
    # Add a justified style for longer text blocks
    styles.add(ParagraphStyle(name='Justified', parent=styles['Normal'], alignment=TA_JUSTIFY))
    return doc, styles, story

def _add_example_header_to_pdf(
    story: list, 
    styles: dict, 
    img_path: str, 
    prompt: str, 
    answer: str, 
    example_num: int
):
    """Adds the image, question, and ground truth for an example to the PDF story."""
    story.append(Paragraph(f"Evaluation Example #{example_num}", styles['h2']))
    try:
        img = PlatypusImage(img_path)
        img_width = img.imageWidth
        img_height = img.imageHeight
        
        if not img_width or not img_height:
             raise ValueError("Image dimensions are zero or invalid.")

        aspect = img_height / float(img_width)
        
        # Calculate proposed width/height based on max width
        proposed_width = PDF_IMAGE_WIDTH
        proposed_height = proposed_width * aspect

        # If proposed height is too large, recalculate based on max height
        if proposed_height > PDF_MAX_IMAGE_HEIGHT:
            proposed_height = PDF_MAX_IMAGE_HEIGHT
            proposed_width = proposed_height / aspect
            # Ensure width doesn't exceed max width after height adjustment
            if proposed_width > PDF_IMAGE_WIDTH:
                proposed_width = PDF_IMAGE_WIDTH
                proposed_height = proposed_width * aspect

        # Set final dimensions
        img.drawWidth = proposed_width
        img.drawHeight = proposed_height
        
        story.append(img)
    except Exception as e:
        print(f"Warning: Could not add image {img_path} to PDF. Error: {e}")
        story.append(Paragraph(f"<i>Error loading/scaling image: {os.path.basename(img_path)}</i>", styles['Italic']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Question:", styles['Bold']))
    story.append(Paragraph(html.escape(prompt), styles['Code'])) # Escape prompt too
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Ground Truth:", styles['Bold']))
    story.append(Paragraph(html.escape(answer), styles['Code'])) # Escape answer too
    story.append(Spacer(1, 0.2 * inch))

def _process_single_completion_for_eval(
    completion_text: str,
    eval_class: evaluator.RewardEvaluator,
    answer: str,
    device: str,
    story: list,
    styles: dict,
    completion_idx: int
) -> tuple[dict, int, int]:
    """ 
    Computes rewards, formats PDF content, and returns metrics/counts for one completion.
    Adapts to ClockEvaluator metrics while preserving original return signature.
    """
    # --- Add completion details to PDF (Existing logic) ---
    story.append(Paragraph(f"Response #{completion_idx + 1}:", styles['Bold']))
    story.append(Paragraph("Full Response:", styles['Italic']))
    escaped_completion_text = html.escape(completion_text) 
    story.append(Paragraph(escaped_completion_text, styles['Code']))

    reasoning = _extract_tagged_content(completion_text, 'reasoning')
    extracted_answer = _extract_tagged_content(completion_text, 'answer')

    story.append(Paragraph("Extracted Reasoning:", styles['Italic']))
    story.append(Paragraph(html.escape(reasoning) if reasoning else "<i>Couldn't extract reasoning tag.</i>", styles['Code'] if reasoning else styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))
        
    story.append(Paragraph("Extracted Answer:", styles['Italic']))
    story.append(Paragraph(html.escape(extracted_answer) if extracted_answer else "<i>Couldn't extract answer tag.</i>", styles['Code'] if extracted_answer else styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Metrics:", styles['Italic']))
    # --- End Add completion details to PDF ---

    # Evaluate the single completion using ClockEvaluator
    mock_completion = [[{'role': 'assistant', 'content': completion_text}]]
    # Ensure answers is a list, expected by ClockEvaluator
    answers_list = [answer] if not isinstance(answer, list) else answer 
    rewards_per_func, metrics_single = eval_class.compute_rewards(
        prompts=None, 
        completions=mock_completion, 
        answers=answers_list, # Pass list of answers
        device=device
    )
    # Add calculated metrics to PDF
    if metrics_single:
        metric_items = [f"- {metric}: {value:.4f}" for metric, value in metrics_single.items()]
        for item in metric_items:
             story.append(Paragraph(item, styles['Normal']))
    else:
         story.append(Paragraph("- <i>Metrics computation failed or N/A</i>", styles['Normal']))

    # Calculate total score for this completion (sum of reward components)
    total_score = rewards_per_func.sum().item() if rewards_per_func is not None else 0.0
    story.append(Paragraph(f"Total Score: {total_score:.4f}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

            
    return metrics_single

def _calculate_and_log_final_metrics(
    avg_scores: dict[str, float],
    json_dir: str,
    round_num: int,
    verbose: bool
) -> None:
    """ Saves JSON, prints verbose output, and returns results."""
    
    final_metrics_data = {
        'average_metrics_per_example': avg_scores
    }
    
    metrics_path = os.path.join(json_dir, f'eval_metrics_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics_data, f, indent=4, default=str) 

    if verbose:
        print("\n--- Evaluation Results ---")
        for avg_metric_name, value in avg_scores.items():
            clean_name = avg_metric_name.replace('avg_', '').replace('metrics/', '').replace('rewards/', '').replace('_reward_func','').replace('_', ' ').capitalize()
            print(f"- {clean_name:<30}: {value:.4f}") 
        print("-" * 30) 




