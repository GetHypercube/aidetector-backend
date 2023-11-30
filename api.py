"""
Flask API for Image Detection

This module provides a Flask web server with endpoints for detecting images using 
pre-trained models and for receiving user feedback on the detection results.
"""
import os
import uuid
from time import time
from flask_cors import CORS
from waitress import serve
from flask import Flask, request, jsonify
import torch
from utils import setup_logger, print_memory_usage, validate_image_file
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
from explainability import craft_explanation

# Setup logger for Flask application
logger = setup_logger(__name__)

app = Flask(__name__)
CORS(app)

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
        print_memory_usage()

    # Preload GAN models
    for model_name in gan_models_config:
        gan_loaded_models[model_name] = load_gan_model(model_name, device)

        logger.info("Loaded GAN model: %s", model_name)
        print_memory_usage()

    logger.info("Model preloading complete!")

@app.route("/debug/preload_models", methods=["GET"])
def debug_preload_models():
    """
    Responds with a list of preloaded models.
    """

    dm_models_loaded = list(dm_loaded_models.keys())
    gan_models_loaded = list(gan_loaded_models.keys())
    return (
        jsonify(
            {
                "DM_Models_Loaded": dm_models_loaded,
                "GAN_Models_Loaded": gan_models_loaded,
            }
        ),
        200,
    )

@app.route("/", methods=["GET"])
def hello_world():
    """
    Responds with a 'Hello world' message.
    """
    return jsonify({"message": "Hello world!"}), 200

@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Receives user feedback for an image detection.

    Expects a JSON payload with 'file_name' and 'feedback' keys.
    Prints the received data and returns a confirmation message.
    """

    logger.debug("Received feedback data: %s", request.data)

    data = request.json
    if not data or "file_name" not in data or "feedback" not in data:
        return jsonify({"error": "Missing file_name or feedback"}), 400

    return jsonify({"message": "Feedback received successfully"})

@app.route("/detect", methods=["POST"])
def detect():
    """
    Detects images using DM and GAN detectors.

    Expects a multipart/form-data request with a file.
    Saves the file to a temporary location and runs detection models on it.
    Returns the combined results of the detections.
    """

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Generate a random mnemonic filename with the original file extension
    _, ext = os.path.splitext(file.filename)
    random_name = f"{uuid.uuid4()}{ext}"
    file_path = f"/tmp/{random_name}"

    # Save the file to a temporary location
    file.save(file_path)

    logger.info("Image saved in temporal the location: %s", file_path)

    # Validate the image file
    try:
        validate_image_file(file_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    logger.info("Image %s is valid", file_path)

    # Start timing
    start_time = time()

    logger.info("Starting DM detection on %s", file_path)

    # Run DM Detector
    dm_results = dm_process_image(
        file_path, preloaded_models=dm_loaded_models
    )

    logger.info("Starting GAN detection on %s", file_path)

    # Run GAN Detector
    gan_results = gan_process_image(
        file_path, preloaded_models=gan_loaded_models
    )

    # Run explainability generator
    preliminary_results = {
        "dMDetectorResults": dm_results,
        "gANDetectorResults": gan_results,
    }
    craft = craft_explanation(file_path, preliminary_results)

    # End timing
    end_time = time()
    total_execution_time = end_time - start_time

    # Combine results
    results = {
        "dMDetectorResults": dm_results,
        "gANDetectorResults": gan_results,
        "explainabilityResults": craft['choices'][0]['message']['content'],
        "totalExecutionTime": total_execution_time,
    }

    return jsonify(results)

if __name__ == "__main__":
    preload_models()
    serve(app, host="0.0.0.0", port=80)
