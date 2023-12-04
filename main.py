"""
Main Module for Command-Line Image Detection

This module provides a command-line interface for running image detection
using Diffusion Model (DM) Detector and Generative Adversarial Network (GAN)
Detector. It allows users to specify an image file for detection and provides
an option for enabling debug mode for more verbose output.

The module processes the provided image through both DM and GAN detectors and
prints the results to the console.

Functions:
    main(): Parses command-line arguments and runs image detection using DM
    and GAN detectors.
"""
import os
import argparse
import json
import logging
import glob
from time import time
import torch
from utils import (
    setup_logger,
    validate_image_file,
    write_to_csv,
    compress_and_resize_image,
    memory_usage,
)
from dmdetector import (
    process_image as dm_process_image,
    load_model as load_dm_model,
    models_config as dm_models_config,
)
from gandetector import (
    process_image as gan_process_image,
    load_model as load_gan_model,
    models_config as gan_models_config,
)
from exifdetector import (
    process_image as exif_process_image,
)
from explainability import craft_explanation

logger = setup_logger(__name__)

# Global variables to store loaded models
dm_loaded_models = {}
gan_loaded_models = {}


def preload_models():
    """
    Preloads models into memory for faster inference.
    """
    logger.info("Starting model preloading...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Preload DM models
    for model_name in dm_models_config:
        dm_loaded_models[model_name] = load_dm_model(model_name, device)

        logger.info("Loaded DM model: %s", model_name)
        logger.info(memory_usage())

    # Preload GAN models
    for model_name in gan_models_config:
        gan_loaded_models[model_name] = load_gan_model(model_name, device)

        logger.info("Loaded GAN model: %s", model_name)
        logger.info(memory_usage())

    logger.info("Model preloading complete!")


def process_folder(folder_path, models):
    """
    Processes all JPEG images in a specified folder (and its subfolders) using given models.

    This function searches for all image files within the given folder and its subdirectories.
    Each image found is processed using the specified models.

    Args:
        folder_path (str): The path to the folder containing images.
        models (list): A list of models to be used for processing each image.

    Returns:
        list: A list of tuples, each containing the file path and the image processing results.
    """
    results = []
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")
    for extension in valid_extensions:
        search_pattern = os.path.join(folder_path, "**", "*" + extension)
        for filepath in glob.iglob(search_pattern, recursive=True):
            image_results = process_image(filepath, models)
            image_data = {"filepath": filepath}
            image_data.update(image_results)
            results.append(image_data)
    return results



def process_image(image_path, models):
    """
    Processes a single image using specified models.

    Validates the given image file and applies a series of models to it, including
    DM Detector, GAN Detector, EXIF Detector, and optionally explainability analysis.

    Args:
        image_path (str): The path to the image file.
        models (list): A list of models to be used for image processing.

    Returns:
        dict: A dictionary containing the results from each applied model, keyed by model name.
    """
    # Validate the image file
    try:
        validate_image_file(image_path)
        processed_image_path = compress_and_resize_image(
            image_path, max_size=(1024, 1024), output_path="tmp"
        )
        logger.info("Image %s validated, resized and compressed", image_path)

    except ValueError as e:
        return logger.error("Image %s is not valid: %s", image_path, e)

    image_results = {}
    # image_results["path"] = processed_image_path
    if "dMDetectorResults" in models:
        logger.info("Starting DM detection on %s", processed_image_path)
        image_results["dMDetectorResults"] = dm_process_image(
            processed_image_path, preloaded_models=dm_loaded_models
        )
    if "gANDetectorResults" in models:
        logger.info("Starting GAN detection on %s", processed_image_path)
        image_results["gANDetectorResults"] = gan_process_image(
            processed_image_path, preloaded_models=gan_loaded_models
        )
    if "exifDetectorResults" in models:
        logger.info("Starting EXIF detection on %s", image_path)
        image_results["exifDetectorResults"] = exif_process_image(image_path)

    # Only add to preliminary_results if the key exists
    preliminary_results = {}
    for model_key in ["dMDetectorResults", "gANDetectorResults", " "]:
        if model_key in image_results:
            preliminary_results[model_key] = image_results[model_key]

    if "explainabilityResults" in models:
        logger.info("Starting explainability detection on %s", processed_image_path)
        image_results["explainabilityResults"] = craft_explanation(
            processed_image_path, preliminary_results
        )
    return image_results


def main():
    """
    Command-line interface for the Diffusor and GAN detector
    """
    parser = argparse.ArgumentParser(
        description="GAN detector inference on a single image"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_path", type=str, help="Path to the image file")
    group.add_argument(
        "--folder_path", type=str, help="Path to the folder containing images"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to use for detection",
        default=[
            "dMDetectorResults",
            "gANDetectorResults",
            "exifDetectorResults",
            "explainabilityResults",
        ],
        choices=[
            "dMDetectorResults",
            "gANDetectorResults",
            "exifDetectorResults",
            "explainabilityResults",
        ],
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results.csv",
        help="Path to the output CSV file",
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
    logging_level = log_levels.get(args.log_level.upper(), logging.INFO)
    setup_logger(__name__, logging_level)

    # Start timing
    start_time = time()

    if args.folder_path:
        preload_models()
        folder_results = process_folder(args.folder_path, args.models)
        write_to_csv(folder_results, args.output_csv)
        # End timing
        end_time = time()
        total_execution_time = end_time - start_time
        results = {
            "folderResults": folder_results,
            "totalExecutionTime": total_execution_time,
        }
    else:
        image_results = process_image(args.image_path, args.models)

        # End timing
        end_time = time()
        total_execution_time = end_time - start_time

        # Combine results
        results = {
            **image_results,
            "totalExecutionTime": total_execution_time,
        }

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
