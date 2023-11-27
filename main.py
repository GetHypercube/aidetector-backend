"""
Main Module for Command-Line Image Detection

This module provides a command-line interface for running image detection 
using Diffusion Model (DM) Detector and Generative Adversarial Network (GAN) Detector. 
It allows users to specify an image file for detection and provides an option 
for enabling debug mode for more verbose output.

The module processes the provided image through both DM and GAN detectors and prints 
the results to the console.

Functions:
    main(): Parses command-line arguments and runs image detection using DM and GAN detectors.
"""

import argparse
import json
import logging
from time import time
from utils import setup_logger
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

logger = setup_logger(__name__, logging.INFO)  # Default level can be INFO

def main():
    """
    Command-line interface for the Diffusor and GAN detector
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
    setup_logger(
        __name__, log_levels.get(args.log_level.upper(), logging.INFO)
    )

    # Start timing
    start_time = time()

    # Run DM Detector
    dm_results = dm_process_image(
        args.image_path
    )

    logger.info("Starting GAN detection on %s", args.image_path)

    # Run GAN Detector
    gan_results = gan_process_image(
        args.image_path
    )

    # End timing
    end_time = time()
    total_execution_time = end_time - start_time

    # Combine results
    results = {
        "dMDetectorResults": dm_results,
        "gANDetectorResults": gan_results,
        "totalExecutionTime": total_execution_time,
    }
    
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
