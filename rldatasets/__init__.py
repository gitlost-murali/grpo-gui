"""
Data loader for generating analog clock images and times.
"""

from typing import Tuple, Any

from . import real_gui
from .gui.dataloader import GUIDataLoader
from .real_gui.dataloader import RealGUIDataLoader
from .base_loader import DataLoader

__all__ = ["real_gui"]


# --- Factory Function ---
def get_dataloaders(dataset_name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.

    Args:
        dataset_name (str): Name of the dataset ('clock' or 'correlation' or 'gui').
        **kwargs: Additional arguments for specific data loaders (e.g., dataset_size, image_width, image_height).

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders

    Raises:
        ValueError: If dataset_name is not supported.
    """
    dataset_size = kwargs.get("dataset_size", 50)
    train_size = kwargs.get("train_size", dataset_size)
    test_size = kwargs.get("test_size", dataset_size)
    image_width = kwargs.get("image_width", 224)
    image_height = kwargs.get("image_height", 224)
    # Get hard mode probability from kwargs, default to 0.1 for training
    hard_mode_prob_train = kwargs.get("hard_mode_prob_train", 0.1)
    # Default hard mode prob for testing is 0.0 unless specified
    hard_mode_prob_test = kwargs.get(
        "hard_mode_prob_test", 0.1
    )  # Let's use same prob for test too

    dataset_name = dataset_name.lower()

    if dataset_name == "gui":
        trainloader = GUIDataLoader(
            dataset_size=dataset_size * 100,
            is_train=True,
            image_width=image_width,
            image_height=image_height,
            hard_mode_prob=hard_mode_prob_train,
        )
        testloader = GUIDataLoader(
            dataset_size=dataset_size,
            is_train=False,
            image_width=image_width,
            image_height=image_height,
            hard_mode_prob=hard_mode_prob_test,
        )
        return trainloader, testloader
    elif dataset_name == "gui_hard":
        trainloader = RealGUIDataLoader(is_train=True, train_test_split=0.8)
        testloader = RealGUIDataLoader(is_train=False, train_test_split=0.2)
        return trainloader, testloader
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Supported: 'gui', 'gui_hard'"
        )


if __name__ == "__main__":
    # Test GUI Loader
    print("\n--- Testing GUI Loader ---")
    try:
        gui_train_loader, gui_test_loader = get_dataloaders("gui", dataset_size=2)
        print(f"GUI Test loader length: {len(gui_test_loader)}")

        print("  Train Sample (GUI):")
        img_path_gui, target_details_gui = next(
            gui_train_loader
        )  # Prompt is now part of target_details_gui
        print(f"    Image Path: {img_path_gui}")
        print(f"    Target Name: {target_details_gui['name']}")
        print(f"    Target BBox: {target_details_gui['bounding_box']}")
        print(
            f"    Dynamic Prompt Snippet: {target_details_gui['dynamic_prompt'][:150]}..."
        )

        print("  Test Samples (GUI):")
        gui_test_loader.reset()
        for i, (img_path_gui, target_details_gui) in enumerate(gui_test_loader):
            print(f"    Test Sample {i + 1}:")
            print(f"      Image Path: {img_path_gui}")
            print(f"      Target Name: {target_details_gui['name']}")
            # print(f"      Target BBox: {target_details_gui['bounding_box']}")
            print(
                f"      Dynamic Prompt Snippet: {target_details_gui['dynamic_prompt'][:150]}..."
            )
            if i == 0:  # Plot the first test image with its target for visual check
                from PIL import Image
                from rldatasets.gui.gui_generator import GUIGenerator  # For plotting

                example_img = Image.open(img_path_gui)
                plot_data = [
                    {
                        "name": "TARGET_" + target_details_gui["name"],
                        "bounding_box": target_details_gui["bounding_box"],
                        "is_truth": True,
                    }
                ]
                img_with_target_plot = GUIGenerator.plot_predictions(
                    example_img, plot_data, truth_color="blue"
                )
                target_plot_filename = "rldatasets_gui_test_sample_with_target.png"
                img_with_target_plot.save(target_plot_filename)
                print(
                    f"      Saved first test sample with target plotted to: {target_plot_filename}"
                )

        print("  GUI Test loader iteration finished.")
    except Exception as e:
        print(f"Error testing GUI loader: {e}")
        import traceback

        traceback.print_exc()
