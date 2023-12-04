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
from utils import setup_logger, validate_image_file, write_to_csv
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
            results.append((filepath, image_results))
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
    except ValueError as e:
        return logger.error("Image %s is not valid: %s", image_path, e)
    logger.info("Image %s is valid", image_path)

    image_results = {}
    image_results["path"] = image_path
    if "dMDetectorResults" in models:
        logger.info("Starting DM detection on %s", image_path)
        image_results["dMDetectorResults"] = dm_process_image(image_path)
    if "gANDetectorResults" in models:
        logger.info("Starting GAN detection on %s", image_path)
        image_results["gANDetectorResults"] = gan_process_image(image_path)
    if "exifDetectorResults" in models:
        logger.info("Starting EXIF detection on %s", image_path)
        image_results["exifDetectorResults"] = exif_process_image(image_path)
    preliminary_results = {
        "dMDetectorResults": image_results["dMDetectorResults"],
        "gANDetectorResults": image_results["gANDetectorResults"],
        "exifDetectorResults": image_results["exifDetectorResults"],
    }
    if "explainabilityResults" in models:
        logger.info("Starting explainability detection on %s", image_path)
        image_results["explainabilityResults"] = craft_explanation(
            image_path, preliminary_results
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
    setup_logger(__name__, log_levels.get(args.log_level.upper(), logging.DEBUG))

    # Start timing
    start_time = time()

    if args.folder_path:
        folder_results = process_folder(args.folder_path, args.models)
        write_to_csv(folder_results, args.output_csv)
        print(json.dumps(folder_results, indent=4))
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
