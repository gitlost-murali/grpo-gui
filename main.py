"""
Implementation of GRPO, DeepSeek style training without external libraries
"""

import os
import json
import torch
import argparse
from tqdm import tqdm

# Import necessary modules and constants for PDF generation in main
from evaluator.orchestrator import eval_on_test_set
from training.loss import grpo_loss

import llms
import utils
import evaluator
import rldatasets as rldatasets
from utils.pdf_logger import generate_pdf_log, pdf_log_info


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training arguments")

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model identifier for main and base model",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="clock",
        choices=["clock", "correlation", "gui"],
        help="Type of dataset to use ('clock', 'correlation', or 'gui')",
    )

    # Output and logging
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save outputs"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=3000,
        help="Save a resumable checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_iterations",
        type=int,
        default=50,
        help="Number of iterations for evaluation",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )

    # Optimization hyperparameters
    parser.add_argument(
        "--learning_rate", type=float, default=5e-6, help="Learning rate"
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.1,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup_percent",
        type=float,
        default=0.18,
        help="Percentage of total steps for warmup",
    )
    parser.add_argument(
        "--update_ref_model",
        action="store_true",
        help="Whether to update reference model",
    )
    parser.add_argument(
        "--update_ref_model_freq",
        type=int,
        default=200,
        help="How often to update reference model",
    )
    parser.add_argument(
        "--ref_model_mixup_alpha",
        type=float,
        default=0.1,
        help="Alpha parameter for reference model mixup",
    )

    # Generation parameters
    parser.add_argument(
        "--temperature", type=float, default=0.9, help="Sampling temperature"
    )
    parser.add_argument(
        "--num_chains",
        type=int,
        default=16,
        help="Number of parallel generation chains",
    )
    parser.add_argument(
        "--num_chains_eval",
        type=int,
        default=2,
        help="Number of parallel generation chains for evaluation",
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=256, help="Maximum prompt length"
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=786,
        help="Maximum completion length",
    )

    # Training parameters
    parser.add_argument(
        "--num_train_iters",
        type=int,
        default=3000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--kl_weight_beta", type=float, default=0.04, help="KL penalty weight"
    )
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    utils.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision("high")

    model, tokenizer = llms.get_llm_tokenizer(args.model_name_or_path, device)
    base_model, _ = llms.get_llm_tokenizer(args.model_name_or_path, device)

    print(f"Loading dataset: {args.dataset_type}")
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_type)

    print(f"Loading evaluator for: {args.dataset_type}")
    eval_class = evaluator.get_evaluator(args.dataset_type)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    args_path = os.path.join(args.output_dir, "args.json")
    if not args.resume_from_checkpoint or not os.path.exists(args_path):
        args_dict = vars(args)
        with open(args_path, "w") as f:
            json.dump(args_dict, f, indent=4)

    eval_log_dir, eval_pdf_dir, eval_json_dir = utils._setup_eval_directories(
        args.output_dir
    )
    train_log_dir = utils._setup_training_log_directory(args.output_dir)
    # Setup dirs for training PDF logs
    train_pdf_dir = os.path.join(train_log_dir, "pdfs")
    train_temp_vis_dir = os.path.join(train_log_dir, "temp_vis")
    os.makedirs(train_pdf_dir, exist_ok=True)
    os.makedirs(train_temp_vis_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    warmup_steps = int(args.warmup_percent * args.num_train_iters)

    def get_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    start_round = 0
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "base_model_state_dict" in checkpoint:
                base_model.load_state_dict(checkpoint["base_model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_round = checkpoint["round_num"] + 1
            print(f"Loaded checkpoint. Resuming from round {start_round}")
            temp_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: get_lr(step + start_round)
            )
            scheduler = temp_scheduler
        else:
            print(
                f"Warning: Checkpoint file not found at {args.resume_from_checkpoint}. Starting training from scratch."
            )

    accumulated_loss_val = 0
    optimizer.zero_grad()
    train_metrics_total = {}
    if os.path.exists(os.path.join(train_log_dir, "train_logs.json")):
        with open(os.path.join(train_log_dir, "train_logs.json"), "r") as f:
            try:
                train_metrics_total = json.load(f)
                train_metrics_total = {
                    int(k): v for k, v in train_metrics_total.items()
                }
            except json.JSONDecodeError:
                print(
                    "Warning: Could not parse existing train_logs.json. Starting with empty logs."
                )
                train_metrics_total = {}

    # Variables to store data for the round to be logged in PDF
    pdf_log_round_data = {}

    for round_num in tqdm(
        range(start_round, args.num_train_iters),
        initial=start_round,
        total=args.num_train_iters,
        desc="Training Progress",
    ):
        # --- Evaluation Step --- (Run periodically)
        if round_num % args.eval_iterations == 0:
            eval_avg_scores, eval_main_metric_val = eval_on_test_set(
                model=model,
                tokenizer=tokenizer,
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num,
            )
            metrics_path = os.path.join(
                eval_json_dir, f"metrics_round_{round_num}.json"
            )  # Use eval_json_dir
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "avg_scores": eval_avg_scores,
                        "main_error_metric_value": eval_main_metric_val,
                    },
                    f,
                    indent=4,
                )

        # --- Reference Model Update --- (Run periodically)
        if args.update_ref_model and (round_num + 1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(
                    model.parameters(), base_model.parameters()
                ):
                    ref_param.data = (
                        args.ref_model_mixup_alpha * param.data
                        + (1 - args.ref_model_mixup_alpha) * ref_param.data
                    )

        # --- Training Step --- (Run every round)
        batch = next(train_loader)
        img_path, data_for_grpo = batch

        # Perform GRPO step and get data needed for potential PDF logging
        loss, train_metrics, completions_text, rewards_per_func, advantages = grpo_loss(
            model,
            base_model,
            tokenizer,
            data_for_grpo,
            img_path,
            data_for_grpo,
            eval_class,
            device,
            round_num,
            train_log_dir,
            args,
        )

        # Store data for potential PDF logging this round
        # This replaces previous round's data, so only the latest is kept
        pdf_log_round_data = pdf_log_info(
            round_num,
            img_path,
            data_for_grpo,
            completions_text,
            rewards_per_func,
            advantages,
        )

        # Gradient accumulation and optimizer step
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        accumulated_loss_val += loss.item()
        scheduler.step()
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # --- Checkpoint Saving & Training PDF Logging --- (Run periodically)
        if (round_num + 1) % args.save_steps == 0:
            # Save checkpoint
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_round_{round_num}.pt"
            )
            torch.save(
                {
                    "round_num": round_num,
                    "model_state_dict": model.state_dict(),
                    "base_model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": args,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

        # --- Generate Training PDF Log for this round ---
        generate_pdf_log(
            round_num,
            pdf_log_round_data,
            train_pdf_dir,
            args,
            train_temp_vis_dir,
            eval_class,
        )

        # --- Logging Training Metrics --- (Run every round)
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = (
            loss.item() * args.gradient_accumulation_steps
        )  # Log un-normalized loss for the step
        # Calculate grad norm *before* zeroing gradients
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        # Calculate grad norm *after* clipping but *before* optimizer step might be more informative if clipping happens
        # Or calculate before clipping to see raw norm. Let's keep it simple for now:
        # Calculate after backward, before optimizer step/zeroing
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm**0.5
        train_metrics["grad_norm"] = grad_norm
        train_metrics_total[round_num] = train_metrics
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)

        # Clear cache periodically
        if round_num % 50 == 0:
            torch.cuda.empty_cache()

        # Add after each major operation in the training loop
        torch.cuda.empty_cache()
