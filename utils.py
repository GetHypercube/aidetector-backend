"""
Utility Functions for Image Processing and System Monitoring

This module provides a set of utility functions that are used across the image
detection application. These include functions for image compression and
resizing, file format validation, and system resource monitoring.

Functions:
    compress_and_resize_image(image_path, max_size):
        Compresses and resizes an image to a specified maximum size.
    validate_image_file(image_path):
        Validates the file format of an image and checks if it's a valid image
        file.
    print_memory_usage():
        Prints the current memory usage of the process to the console.
"""

import os
import base64
import logging
import csv
import numpy as np
import psutil
from PIL import Image, UnidentifiedImageError


def setup_logger(name, level=logging.DEBUG):
    """
    Sets up and returns a logger with the specified name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handler and formatter
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Avoid duplicate logging
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


# Configure logger for utils.py
logger = setup_logger(__name__)


def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a[0].upper() + a[1:])
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


def write_to_csv(results, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = None

        flattened_result = flatten_json(results)

        # Initialize CSV writer and write headers
        if writer is None:
            writer = csv.DictWriter(file, fieldnames=flattened_result.keys())
            writer.writeheader()

        # Write the row
        writer.writerow(flattened_result)


def calculate_sigmoid_probabilities(logits_dict):
    """
    Adds sigmoid probabilities to the logits dictionary.

    Parameters:
    logits_dict (dict): Dictionary containing logits.

    Returns:
    dict: Updated dictionary with sigmoid probabilities.
    """
    sigmoid_probs = {}
    for model, logit in logits_dict.items():
        sigmoid_prob = 1 / (1 + np.exp(-logit))  # Sigmoid function
        sigmoid_probs[model] = sigmoid_prob
    return sigmoid_probs


def encode_image(image_path):
    """
    Encodes an image to a base64 string.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    str: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def validate_image_file(image_path):
    """
    Validates that the image has a valid file extension and is a valid image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        bool: True if the file is a valid image, False otherwise.
    """
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")

    # Check file extension
    if not image_path.lower().endswith(valid_extensions):
        logger.warning("Unsupported file format. Accepts only JPEG, PNG, and WebP.")
        raise ValueError("Unsupported file format. Accepts only JPEG, PNG, and WebP.")

    # Validate with PIL
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verifies that an image can be opened
    except (UnidentifiedImageError, IOError) as exc:
        logger.warning("Invalid image file or path.")
        raise ValueError("Invalid image file or path.") from exc

    return True


def compress_and_resize_image(image_path, max_size=(1024, 1024)):
    """
    Compresses and resizes an image to a manageable size.

    Args:
        image_path (str): Path to the image file.
        max_size (tuple): Maximum width and height of the resized image.

    Returns:
        str: Path to the processed image.
    """
    # Open and process the image
    with Image.open(image_path) as img:
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            # Resize the image only if it's larger than max_size
            img.thumbnail(max_size, Image.LANCZOS)
        # Save the processed image in a lossless format
        processed_image_path = os.path.splitext(image_path)[0] + "_processed.png"
        img.save(processed_image_path, format="PNG", optimize=True)
        return processed_image_path


def print_memory_usage():
    """
    Prints the current memory usage of the process.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_used = mem_info.rss / (1024 * 1024)
    logger.info("Memory used: %.2f MB", memory_used)
