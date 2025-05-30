import os
import evaluator
import utils
from utils import MAX_COMPLETIONS_PER_PAGE_PDF
from reportlab.platypus import PageBreak  # Import PageBreak
from PIL import Image as PILImage
from rldatasets.gui.gui_generator import GUIGenerator


def pdf_log_info(
    round_num, img_path, data_for_grpo, completions_text, rewards_per_func, advantages
):
    return {
        "round_num": round_num,
        "img_path": img_path,
        "prompt_info": data_for_grpo,  # Contains target_details for GUI, prompt_str otherwise
        "completions_text": completions_text,
        "rewards_per_func": rewards_per_func,
        "advantages": advantages,
    }


def generate_pdf_log(
    round_num, pdf_log_round_data, train_pdf_dir, args, train_temp_vis_dir, eval_class
):
    print(f"Generating training PDF log for round {round_num}...")
    # Use the data stored from the *last* training step of this logging interval
    log_data = pdf_log_round_data
    if log_data.get("round_num") == round_num:  # Ensure data is from the correct round
        pdf_train_filename = f"training_log_round_{round_num}.pdf"
        pdf_train_path = os.path.join(train_pdf_dir, pdf_train_filename)
        doc_train, styles_train, story_train = utils._setup_pdf(pdf_train_path)

        # Determine prompt text and answer/target data based on type
        if args.dataset_type == "gui":
            prompt_text_for_pdf = log_data["prompt_info"]["dynamic_prompt"]
            answer_data_for_pdf = log_data["prompt_info"]  # The dict itself
        else:
            prompt_text_for_pdf = log_data["prompt_info"]  # The static prompt string
            answer_data_for_pdf = log_data[
                "prompt_info"
            ]  # The answer string (same as prompt_info here)

        # Add PDF Header (Image, Prompt, Target Info)
        utils._add_example_header_to_pdf(
            story_train,
            styles_train,
            log_data["img_path"],
            prompt_text_for_pdf,
            answer_data_for_pdf,
            round_num,
            args.dataset_type,
        )

        # Add Completions
        num_chains_logged = min(
            len(log_data["completions_text"]), 8
        )  # Log max 8 chains to keep PDF size manageable
        for compl_idx in range(num_chains_logged):
            completion_text = log_data["completions_text"][compl_idx]
            reward_scores = log_data["rewards_per_func"][compl_idx]
            advantage = log_data["advantages"][compl_idx].item()  # Get scalar value
            reward_breakdown = eval_class.get_reward_breakdown(
                reward_scores
            )  # Get dict

            vis_train_img_path = os.path.join(
                train_temp_vis_dir, f"train_round{round_num}_comp{compl_idx}.png"
            )

            img_path_for_pdf_entry = None
            # Plot click for GUI task
            if args.dataset_type == "gui":
                try:
                    if isinstance(eval_class, evaluator.GUIEvaluator):
                        parsed_click = eval_class._extract_coordinates(completion_text)
                        if parsed_click:
                            pil_img = PILImage.open(log_data["img_path"])
                            plot_data = [
                                {
                                    "name": "VLM Click",
                                    "center_x": parsed_click[0],
                                    "center_y": parsed_click[1],
                                    "is_truth": False,
                                }
                            ]
                            img_w_click = GUIGenerator.plot_predictions(
                                pil_img, plot_data, pred_color="red"
                            )
                            img_w_click.save(vis_train_img_path)
                            img_path_for_pdf_entry = vis_train_img_path
                        else:
                            img_path_for_pdf_entry = log_data[
                                "img_path"
                            ]  # Use original if click not parsed
                    else:
                        img_path_for_pdf_entry = log_data["img_path"]
                except Exception as plot_err:
                    print(
                        f"  Warning: Error plotting training click for PDF (Round {round_num}, Comp {compl_idx}): {plot_err}"
                    )
                    img_path_for_pdf_entry = log_data["img_path"]
            else:
                img_path_for_pdf_entry = log_data[
                    "img_path"
                ]  # Use original image for non-GUI tasks

            # Add completion details to story
            utils._add_training_completion_to_pdf(
                story=story_train,
                styles=styles_train,
                completion_text=completion_text,
                reward_breakdown=reward_breakdown,
                advantage=advantage,
                completion_idx=compl_idx,
                image_path_for_completion_pdf=img_path_for_pdf_entry,
            )
            # Add PageBreak if needed (e.g., after every 2 completions)
            if (compl_idx + 1) % MAX_COMPLETIONS_PER_PAGE_PDF == 0:
                story_train.append(PageBreak())

        # Build the PDF
        doc_train.build(story_train)
        print(f"Training PDF log saved to {pdf_train_path}")
    else:
        print(
            f"Warning: Skipping training PDF log for round {round_num} as stored data mismatch."
        )
