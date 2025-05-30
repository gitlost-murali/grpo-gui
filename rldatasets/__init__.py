"""
Data loader for generating analog clock images and times.
"""

from typing import Tuple, Any
from .clock.dataloader import ClockDataLoader
from .correlation.dataloader import CorrelationScatterDataLoader
from .gui.dataloader import GUIDataLoader
from .real_gui.dataloader import RealGUIDataLoader
from .gui.gui_generator import GUIGenerator
from .base_loader import DataLoader


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
    image_width = kwargs.get("image_width", 224)
    image_height = kwargs.get("image_height", 224)
    # Get hard mode probability from kwargs, default to 0.1 for training
    hard_mode_prob_train = kwargs.get("hard_mode_prob_train", 0.1)
    # Default hard mode prob for testing is 0.0 unless specified
    hard_mode_prob_test = kwargs.get(
        "hard_mode_prob_test", 0.1
    )  # Let's use same prob for test too

    dataset_name = dataset_name.lower()

    if dataset_name == "clock":
        trainloader = ClockDataLoader(dataset_size=dataset_size * 100, is_train=True)
        testloader = ClockDataLoader(dataset_size=dataset_size, is_train=False)
        return trainloader, testloader
    elif dataset_name == "correlation":
        trainloader = CorrelationScatterDataLoader(
            dataset_size=dataset_size * 100, is_train=True
        )
        testloader = CorrelationScatterDataLoader(
            dataset_size=dataset_size, is_train=False
        )
        return trainloader, testloader
    elif dataset_name == "gui":
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
        trainloader = RealGUIDataLoader(dataset_size=dataset_size * 100, is_train=True)
        testloader = RealGUIDataLoader(dataset_size=dataset_size, is_train=False)
        return trainloader, testloader

    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Supported: 'clock', 'correlation', 'gui'"
        )


if __name__ == "__main__":
    # Test Clock Loader
    print("--- Testing Clock Loader ---")
    try:
        train_loader_c, test_loader_c = get_dataloaders("clock", dataset_size=2)
        print(f"Clock Train loader prompt: {train_loader_c.prompt[:80]}...")
        print(f"Clock Test loader length: {len(test_loader_c)}")

        print("  Train Sample:")
        img_path_c, label_c = next(train_loader_c)
        print(f"    Image Path: {img_path_c}, Label: {label_c}")

        print("  Test Samples:")
        test_loader_c.reset()
        for img_path_c, label_c in test_loader_c:
            print(f"    Image Path: {img_path_c}, Label: {label_c}")
        print("  Clock Test loader iteration finished.")
    except Exception as e:
        print(f"Error testing clock loader: {e}")

    # Test Correlation Loader
    print("\n--- Testing Correlation Loader ---")
    try:
        train_loader_r, test_loader_r = get_dataloaders("correlation", dataset_size=2)
        print(f"Correlation Train loader prompt: {train_loader_r.prompt[:80]}...")
        print(f"Correlation Test loader length: {len(test_loader_r)}")

        print("  Train Sample:")
        img_path_r, label_r = next(train_loader_r)
        print(f"    Image Path: {img_path_r}, Label: {label_r}")

        print("  Test Samples:")
        test_loader_r.reset()
        for img_path_r, label_r in test_loader_r:
            print(f"    Image Path: {img_path_r}, Label: {label_r}")
        print("  Correlation Test loader iteration finished.")
    except Exception as e:
        print(f"Error testing correlation loader: {e}")

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
