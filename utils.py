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
    memory_usage():
        Returns the current memory usage of the process to the console.
"""

import os
import base64
import logging
import csv
import numpy as np
import psutil
from PIL import Image, UnidentifiedImageError


def setup_logger(name, level=logging.INFO):
    """
    Sets up and returns a logger with the specified name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    module_logger = logging.getLogger(name)
    module_logger.setLevel(level)

    # Create handler and formatter
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Avoid duplicate logging
    if not module_logger.handlers:
        module_logger.addHandler(handler)

    return module_logger


def flatten_json(y):
    """
    Flattens a nested JSON object into a flat dictionary.

    This function takes a nested JSON object (which may contain nested dictionaries
    and lists) and converts it into a flat dictionary with compound keys representing
    the paths to each value in the original JSON structure.

    Args:
        y (dict): The JSON object to flatten.

    Returns:
        dict: A flattened dictionary with compound keys.
    """
    out = {}

    def flatten(x, name=""):
        if isinstance(x, dict):
            for a in x:
                # Add an underscore only if name is not empty
                next_key = name + a + ('_' if name else '')
                flatten(x[a], next_key)
        elif isinstance(x, list):
            i = 0
            for a in x:
                flatten(a, name + str(i) + "_")
                i += 1
        else:
            # Remove the trailing underscore (if any) before assigning the value
            clean_name = name[:-1] if name.endswith('_') else name
            out[clean_name] = x

    flatten(y)
    return out


def write_to_csv(results, output_file):
    """
    Writes a list of results to a CSV file, flattening any nested structures.

    This function takes a list of results, where each result is expected to be a dictionary
    (potentially with nested structures), and writes them to a CSV file. Each result is
    flattened into a single row in the CSV file.

    Args:
        results (list): A list of dictionaries to be written to the CSV file.
        output_file (str): The path of the output CSV file.

    Note:
        The CSV file's columns are dynamically determined based on the keys of the
        flattened dictionaries.
    """
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = None
        for result in results:
            flattened_result = flatten_json(result)

            # Initialize CSV writer and write headers only once
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=flattened_result.keys())
                writer.writeheader()

            # Write each row
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
        return base64.b64encode(image_file.read()).decode("utf-8")


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
        raise ValueError("Unsupported file format. Accepts only JPEG, PNG, and WebP.")

    # Validate with PIL
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verifies that an image can be opened
    except (UnidentifiedImageError, IOError) as exc:
        raise ValueError("Invalid image file or path.") from exc

    return True


def compress_and_resize_image(image_path, max_size=(1024, 1024), output_path=None):
    """
    Compresses and resizes an image to a manageable size.

    Args:
        image_path (str): Path to the image file.
        max_size (tuple): Maximum width and height of the resized image.
        output_path (str, optional): Directory where the processed image will be saved.
                                     If None, the image will be saved in the same 
                                     directory as the original.

    Returns:
        str: Path to the processed image.
    """
    # Convert relative path to absolute path
    image_path = os.path.abspath(image_path)
    output_path = (
        os.path.abspath(output_path) if output_path else os.path.dirname(image_path)
    )

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with Image.open(image_path) as img:
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.LANCZOS)

        processed_file_name = (
            os.path.splitext(os.path.basename(image_path))[0] + "_processed.png"
        )
        processed_image_path = os.path.join(output_path, processed_file_name)

        img.save(processed_image_path, format="PNG", optimize=True)
        return processed_image_path


def memory_usage():
    """
    Prints the current memory usage of the process.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_used = mem_info.rss / (1024 * 1024)
    return memory_used
