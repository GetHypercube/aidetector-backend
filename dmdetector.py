"""
DM Detector: A module for running inference on single images using pre-trained models.
Supports different models and applies necessary transformations before prediction.

Prints out the logits returned by each model and the final label based on these logits.
"""

import argparse
import time
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import networks.resnet_mod as resnet_mod
from utils import compress_and_resize_image, print_memory_usage

models_config = {
    "Grag2021_progan": {
        "model_path": "./weights/diffusion/Grag2021_progan/model_epoch_best.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
    },
    "Grag2021_latent": {
        "model_path": "./weights/diffusion/Grag2021_latent/model_epoch_best.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
    },
}


def load_and_process_model(model_name, device, debug):
    """
    Loads and processes the model from the config.
    Args:
        model_name (str): Name of the model to load.
        device (torch.device): Device to use for the model.
    Returns:
        torch.nn.Module: Loaded and processed model.
    """
    config = models_config[model_name]
    model = resnet_mod.resnet50(num_classes=1, gap_size=1, stride0=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config["model_path"], map_location=device)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise KeyError("No model state_dict found in the checkpoint file")

    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    if debug:
        print(f"Model {model_name} loaded")
        print_memory_usage()

    return model


def get_transformations(norm_type):
    """
    Gets the transformations based on the normalization type.
    Args:
        norm_type (str): Type of normalization to apply.
    Returns:
        torchvision.transforms.Compose: Transformations to apply.
    """
    if norm_type == "resnet":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def process_image(image_path, debug, preloaded_models=None):
    """
    Runs inference on a single image using specified models and weights.
    """
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if debug:
        if torch.cuda.is_available():
            print("Running on GPU")
        else:
            print("Running on CPU")
    logits = {}

    processed_image_path = compress_and_resize_image(image_path)
    img = Image.open(processed_image_path).convert("RGB")

    for model_name, config in models_config.items():
        model = (
            preloaded_models.get(model_name)
            if preloaded_models
            else load_and_process_model(model_name, device, debug)
        )

        transform = get_transformations(config["norm_type"])

        with torch.no_grad():
            transformed_img = transform(img)
            transformed_img = transformed_img.unsqueeze(0).to(device)
            logit = model(transformed_img).cpu().numpy()
            logits[model_name] = np.mean(logit, (2, 3)).item()

        if not preloaded_models:  # Only delete model if it was not preloaded
            del model
            torch.cuda.empty_cache()
            if debug:
                print(f"Model {model_name} unloaded")
                print_memory_usage()

    execution_time = time.time() - start_time
    label = "False" if any(value < 0 for value in logits.values()) else "True"

    detection_output = {
        "model": "diffusion-model-detector",
        "inferenceResults": {
            "logits": logits,
            "isDiffusionImage": label,
            "executionTime": execution_time,
        },
    }

    return detection_output


def main():
    """
    Command-line interface for the GAN detector.
    """
    parser = argparse.ArgumentParser(
        description="DM Detector Inference on a Single Image"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    return process_image(args.image_path, args.debug)


if __name__ == "__main__":
    output = main()
    print(json.dumps(output, indent=4))
