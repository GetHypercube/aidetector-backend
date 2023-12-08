"""
GAN Detector: Inference on a single image using pre-trained models for GAN
detection. It prints out the logits returned by each model and the final
label based on these logits.

Based on: https://github.com/grip-unina/GANimageDetection

"""
import argparse
import time
import json
import logging
import torch
import numpy as np
from PIL import Image
from models.networks.resnet50nodown import resnet50nodown
from utils import (
    setup_logger,
    memory_usage,
    calculate_sigmoid_probabilities,
)

logger = setup_logger(__name__)

models_config = {
    "gandetection_resnet50nodown_progan": {
        "model_path": "weights/gan/gandetection_resnet50nodown_progan.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
    },
    "gandetection_resnet50nodown_stylegan2": {
        "model_path": "weights/gan/gandetection_resnet50nodown_stylegan2.pth",
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
    model_config = models_config[model_name]
    model = resnet50nodown(device, model_config["model_path"])

    logger.info("Model %s loaded", model_name)
    logger.info(memory_usage())

    return model


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

    logger.info("Image %s resized and compressed", image_path)

    img = Image.open(image_path).convert("RGB")
    img.load()

    for model_name, model_config in models_config.items():
        logger.info("Processing the model: %s", model_name)

        model = (
            preloaded_models.get(model_name)
            if preloaded_models
            else load_model(model_name, device)
        )

        logit = model.apply(img)
        logits[model_name] = logit.item() if isinstance(logit, np.ndarray) else logit

        logger.info("Calculated the logit of model: %s", model_name)

        if not preloaded_models:  # Only delete model if it was not preloaded
            del model
            torch.cuda.empty_cache()
            logger.info("Model %s unloaded", model_name)
            logger.info(memory_usage())

    execution_time = time.time() - start_time

    # Calculate if the image is fake or not

    threshold = 0.2

    sigmoid_probs = calculate_sigmoid_probabilities(logits)

    logger.debug("Calculated the sigmoid probabilities")

    for prob in sigmoid_probs.values():
        if prob >= threshold:
            is_gan_image = True  # Image is classified as fake
            break
        else:
            is_gan_image = False

    # Fuse both models outputs

    fused_logit = np.mean(list(logits.values()))
    fused_sigmoid_prob = np.mean(list(sigmoid_probs.values()))

    detection_output = {
        # "model": "gan-model-detector",
        "inferenceResults": {
            "logits": logits,
            "probabilities": sigmoid_probs,
            "fusedLogit": fused_logit,
            "fusedProbability": fused_sigmoid_prob,
            "isGanImage": is_gan_image,
            "executionTime": execution_time,
        },
    }

    return detection_output


def main():
    """
    Command-line interface for the GAN detector.
    """
    parser = argparse.ArgumentParser(
        description="GAN detector inference on a single image"
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
    setup_logger(__name__, log_levels.get(args.log_level.upper(), logging.INFO))
    return process_image(args.image_path)


if __name__ == "__main__":
    output = main()
    print(json.dumps(output, indent=4))
