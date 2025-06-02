"""
Module for loading LLMs and their tokenizers from huggingface.

"""

import argparse
import torch
from transformers import (
    GenerationConfig, #type: ignore
    PreTrainedModel, #type: ignore
    PreTrainedTokenizerBase, #type: ignore
)

from transformers import (
    Qwen2_5_VLForConditionalGeneration, #type: ignore
    AutoProcessor, #type: ignore
)
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig #type: ignore
from PIL import Image


def get_llm_tokenizer(
    model_name: str, device: str, quantized: bool = False
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load and configure a language model and its tokenizer.

    Args:
        model_name: Name or path of the pretrained model to load
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        tuple containing:
            - The loaded language model
            - The configured tokenizer for that model
    """
    nf4_config = None
    if quantized:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        quantization_config=nf4_config,
    )

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    # default processer
    processor = AutoProcessor.from_pretrained(model_name)

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    processor.tokenizer.padding_side = "left"
    processor.padding_side = "left"

    # This fixed ~'need to set the pdadding.left' but even if you do that nothing works
    model.config.use_cache = False

    return model, processor


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    img_or_image_path: str | Image.Image,
    prompt: str,
    device: str,
    args: argparse.Namespace,
    eval: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """
    Generate multiple completion sequences for a given prompt using a language model.

    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        image_path: The input image path to generate completions for
        prompt: The input prompt to generate completions for
        device: Device to run generation on ('cpu' or 'cuda')
        args: Namespace containing generation parameters

    Returns:
        prompt_completion_ids: Tensor containing the full sequence of prompt + completion token IDs
        prompt_ids: Tensor containing just the prompt token IDs
        completion_ids: Tensor containing just the completion token IDs
        attention_mask: Attention mask tensor for the full sequence
        completions_text: List of decoded completion texts
        prompt_text: The full formatted prompt text
    """

    conversation = [
        {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group. You are an expert image analyst.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_or_image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    image_inputs, video_inputs = process_vision_info(conversation) #type: ignore

    # Ensure left padding for tokenizer/processor before tokenizing
    prompt_inputs = (
        tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        .to(model.device)
        .to(model.dtype)
    )

    # Repeat input tensors for batch generation
    if eval:
        num_chains = args.num_chains_eval
    else:
        num_chains = args.num_chains
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(
                num_chains, *([1] * (value.dim() - 1))
            )
        else:
            # Handle non-tensor items if necessary, otherwise just copy
            batched_prompt_inputs[key] = value

    # Original prompt_ids/mask are needed for splitting later
    original_prompt_ids = prompt_inputs["input_ids"]

    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        temperature=args.temperature,
        pad_token_id=tokenizer.tokenizer.pad_token_id,
    )

    # Generate all completions at once
    prompt_completion_ids = model.generate(
        **batched_prompt_inputs, generation_config=generation_config
    )

    # Extract completion ids
    # Use the original prompt length before repeating
    prompt_length = original_prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[
        :, :prompt_length
    ]  # These are the batched prompt IDs
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Do masking
    is_eos = completion_ids == tokenizer.tokenizer.eos_token_id
    eos_idx = torch.full(
        (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device
    )
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(
        is_eos.size(0), -1
    )
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    # Create attention mask based on original prompt mask repeated and completion mask
    prompt_mask = batched_prompt_inputs["attention_mask"]  # Use the repeated mask
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    # Decode completions
    completions_text = tokenizer.batch_decode(
        completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return (
        prompt_completion_ids,
        prompt_ids,
        completion_ids,
        attention_mask,
        completions_text,
        prompt,
    )
