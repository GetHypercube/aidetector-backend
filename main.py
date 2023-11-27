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
from dmdetector import process_image as dm_process_image
from gandetector import process_image as gan_process_image
from utils import setup_logger

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

    # Run DM Detector
    dm_results = dm_process_image(args.image_path)
    print("DM Detector Results:")
    print(json.dumps(dm_results, indent=4))

    # Run GAN Detector
    gan_results = gan_process_image(args.image_path)
    print("GAN Detector Results:")
    print(json.dumps(gan_results, indent=4))


if __name__ == "__main__":
    main()
