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
        "model_path": "./weights/gan/gandetection_resnet50nodown_progan.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
        "patch_size": None,
    },
    "gandetection_resnet50nodown_stylegan2": {
        "model_path": "./weights/gan/gandetection_resnet50nodown_stylegan2.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
        "patch_size": None,
    },
}

def load_model(model_name, device):
    """
    Loads the model from the config.
    """
    model_config = models_config[model_name]
    model = resnet50nodown(device, model_config["model_path"])
    return model

def process_image(image_path, debug):
    """
    Runs inference on a single image using specified models and weights.
    Args:
        image_path (str): Path to the image file for inference.
        debug (bool): Flag to enable debug mode.
    Returns:
        dict: JSON object with detection results and execution time.
    """
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logits = {}

    processed_image_path = compress_and_resize_image(image_path)
    img = Image.open(processed_image_path).convert("RGB")
    img.load()

    for model_name in models_config:
        if debug:
            print(f"Model {model_name} processed")
            print_memory_usage()

        model = load_model(model_name, device)
        logit = model.apply(img)
        logits[model_name] = logit.item() if isinstance(logit, np.ndarray) else logit

        del model
        torch.cuda.empty_cache()

        if debug:
            print_memory_usage()

    execution_time = time.time() - start_time
    label = "True" if any(value < 0 for value in logits.values()) else "False"

    output = {
        "product": "gan-model-detector",
        "detection": {
            "logit": logits,
            "IsGANImage?": label,
            "ExecutionTime": execution_time,
        },
    }

    return output

def main():
    """
    Command-line interface for the GAN detector.
    """
    parser = argparse.ArgumentParser(description="GAN Detector script.")
    parser.add_argument("--image_path", "-i", type=str, required=True, help="Input image path (PNG or JPEG)")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()

    output = process_image(args.image_path, args.debug)
    
    return output

if __name__ == "__main__":
    output = main()
    print(json.dumps(output, indent=4))
