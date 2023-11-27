"""
Diffusor detector: Inference on a single image using pre-trained models for GAN detection.
It prints out the logits returned by each model and the final label based on these logits.
"""

import argparse
import time
import json
import logging
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import networks.resnet_mod as resnet_mod
from utils import setup_logger, compress_and_resize_image, print_memory_usage

logger = setup_logger(__name__, logging.INFO)  # Default level can be INFO

models_config = {
    "Grag2021_progan": {
        "model_path": "weights/diffusion/Grag2021_progan/model_epoch_best.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
    },
    "Grag2021_latent": {
        "model_path": "weights/diffusion/Grag2021_latent/model_epoch_best.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
    },
}


def load_model(model_name, device):
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

    logger.info("Model %s loaded", model_name)
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


def process_image(image_path, preloaded_models=None):
    """
    Runs inference on a single image using specified models and weights.
    Args:
        image_path (str): Path to the image file for inference.
        preloaded_models (dict, optional): Dictionary of preloaded models.
    Returns:
        dict: JSON object with detection results and execution time.
    """
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info("Running on GPU")
    else:
        logger.info("Running on CPU")
    logits = {}

    processed_image_path = compress_and_resize_image(image_path)

    logger.info("Image %s resized and compressed", image_path)

    img = Image.open(processed_image_path).convert("RGB")
    img.load()

    logger.debug("Before looping all models")

    for model_name, config in models_config.items():

        logger.info("Processing the model: %s", model_name)

        model = (
            preloaded_models.get(model_name)
            if preloaded_models
            else load_model(model_name, device)
        )

        transform = get_transformations(config["norm_type"])

        logger.debug("Executed get_transformations")

        with torch.no_grad():
            transformed_img = transform(img)
            transformed_img = transformed_img.unsqueeze(0).to(device)
            logit = model(transformed_img).cpu().numpy()
            logits[model_name] = np.mean(logit, (2, 3)).item()

        logger.info("Calculated the logit of model: %s", model_name)

        if not preloaded_models:  # Only delete model if it was not preloaded
            del model
            torch.cuda.empty_cache()
            logger.info("Model %s unloaded", model_name)
            print_memory_usage()

    execution_time = time.time() - start_time

    label = "True" if any(value > 0 for value in logits.values()) else "False"

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
    Command-line interface for the Diffusor detector.
    """
    parser = argparse.ArgumentParser(
        description="Diffusion detector inference on a single image"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    args = parser.parse_args()
    # Configure logger
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    setup_logger(
        __name__, log_levels.get(args.log_level.upper(), logging.INFO)
    )

    return process_image(args.image_path)


if __name__ == "__main__":
    output = main()
    print(json.dumps(output, indent=4))
