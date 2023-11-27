"""
GAN Detector: Inference on a single image using pre-trained models for GAN detection.
It prints out the logits returned by each model and the final label based on these logits.
"""

import time
import argparse
import json
import torch
import numpy as np
from PIL import Image
from networks.resnet50nodown import resnet50nodown
from utils import compress_and_resize_image, print_memory_usage

models_config = {
    "gandetection_resnet50nodown_progan": {
        "model_path": "weights/gan/gandetection_resnet50nodown_progan.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
        "patch_size": None,
    },
    "gandetection_resnet50nodown_stylegan2": {
        "model_path": "weights/gan/gandetection_resnet50nodown_stylegan2.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
        "patch_size": None,
    },
}


def load_model(model_name, device, debug):
    """
    Loads the model from the config.
    """
    model_config = models_config[model_name]
    model = resnet50nodown(device, model_config["model_path"])
    if debug:
        print(f"Model {model_name} loaded")
        print_memory_usage()
    return model


def process_image(image_path, debug, preloaded_models=None):
    """
    Runs inference on a single image using specified models and weights.
    Args:
        image_path (str): Path to the image file for inference.
        debug (bool): Flag to enable debug mode.
        preloaded_models (dict, optional): Dictionary of preloaded models.
    Returns:
        dict: JSON object with detection results and execution time.
    """
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if debug:
        if torch.cuda.is_available():
            print("DEBUG: Running on GPU")
        else:
            print("DEBUG: Running on CPU")
    logits = {}

    processed_image_path = compress_and_resize_image(image_path)

    print("DEBUG: Image compressed and resized")

    img = Image.open(processed_image_path).convert("RGB")
    img.load()

    print("DEBUG: Before looping each model")

    for model_name, model_config in models_config.items():

        print(f"DEBUG: Processing model {model_name}")

        model = (
            preloaded_models.get(model_name)
            if preloaded_models
            else load_model(model_name, device, debug)
        )

        logit = model.apply(img)
        logits[model_name] = logit.item() if isinstance(logit, np.ndarray) else logit

        print(f"DEBUG: Calculated the logits of model {model_name}")

        if not preloaded_models:  # Only delete model if it was not preloaded
            del model
            torch.cuda.empty_cache()
            if debug:
                print(f"Model {model_name} unloaded")
                print_memory_usage()

    execution_time = time.time() - start_time

    label = "True" if any(value > 0 for value in logits.values()) else "False"

    detection_output = {
        "model": "gan-model-detector",
        "inferenceResults": {
            "logits": logits,
            "isGANImage": label,
            "executionTime": execution_time,
        },
    }

    return detection_output


def main():
    """
    Command-line interface for the GAN detector.
    """
    parser = argparse.ArgumentParser(description="GAN Detector script.")
    parser.add_argument(
        "--image_path",
        "-i",
        type=str,
        required=True,
        help="Input image path (PNG or JPEG)",
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    return process_image(args.image_path, args.debug)


if __name__ == "__main__":
    output = main()
    print(json.dumps(output, indent=4))
