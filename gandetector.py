"""
This module is used for running inference on a single image using pre-trained
models. It supports different models and applies necessary transformations to the
input image before feeding it to the models for prediction.

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
    """Loads the model from the config."""
    model_config = models_config[model_name]
    model = resnet50nodown(device, model_config["model_path"])
    return model


def process_image(model, img):
    """Passes the image through the model to get the logit."""
    return model.apply(img)


def main():
    """
    The main function of the script. It parses command-line arguments and runs the inference test.

    The function expects three command-line arguments:
    - `--image_path`: The path to the image file on which inference is to be performed.
    - `--debug`: Show memory usage or not

    After parsing the arguments, it calls the `run_single_test` function to perform inference
    on the specified image using the provided model weights.
    """
    parser = argparse.ArgumentParser(
        description="This script tests the network on a single image."
    )
    parser.add_argument(
        "--image_path",
        "-i",
        type=str,
        required=True,
        help="input image path (PNG or JPEG)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="enables debug mode to print memory usage",
    )
    config = parser.parse_args()
    image_path = config.image_path
    debug_mode = config.debug

    start_time = time.time()
    logits = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    processed_image_path = compress_and_resize_image(image_path)
    img = Image.open(processed_image_path).convert("RGB")
    img.load()

    for model_name in models_config:
        if debug_mode:
            print(f"Model {model_name} processed")
            print_memory_usage()


        model = load_model(model_name, device)
        logit = process_image(model, img)

        logits[model_name] = logit.item() if isinstance(logit, np.ndarray) else logit

        # Unload model from memory
        del model
        torch.cuda.empty_cache()

        if debug_mode:
            print_memory_usage()

    execution_time = time.time() - start_time

    label = "True" if any(value < 0 for value in logits.values()) else "False"

    # Construct output JSON
    output = {
        "product": "gan-model-detector",
        "detection": {
            "logit": logits,
            "IsGANImage?": label,
            "ExecutionTime": execution_time,
        },
    }

    return output

if __name__ == "__main__":
    output = main()
    print(json.dumps(output, indent=4))
