"""
Flask API for Image Detection

This module provides a Flask web server with endpoints for detecting images
using pre-trained models and for receiving user feedback on the detection
results.
"""
import os
import uuid
import tempfile
from time import time
from datetime import datetime
from flask_cors import CORS
from waitress import serve
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import torch
from pymongo import MongoClient
from utils.general import (
    setup_logger,
    get_memory_usage,
    validate_image_file,
    compress_and_resize_image,
)
from utils.aws import (
    upload_image_to_s3
)
from utils.aws import (
    get_secret
)
from models.dmdetector import (
    process_image as dm_process_image,
    load_model as load_dm_model,
    models_config as dm_models_config,
)
from models.gandetector import (
    process_image as gan_process_image,
    load_model as load_gan_model,
    models_config as gan_models_config,
)
from models.exifdetector import (
    process_image as exif_process_image,
)
from models.explainability import (
    craft_explanation
)
logger = setup_logger(__name__)

# Load environment variables from .env file
load_dotenv()

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
        logger.info("Memory usage: %s", get_memory_usage())

    # Preload GAN models
    for model_name in gan_models_config:
        gan_loaded_models[model_name] = load_gan_model(model_name, device)

        logger.info("Loaded GAN model: %s", model_name)
        logger.info("Memory usage: %s", get_memory_usage())

    logger.info("Model preloading complete!")

def save_to_mongodb(image_path, inference_results):
    """
    Save results to MongoDB
    """

    mongodb_url = os.getenv("MONGODB_URL")
    if mongodb_url:
        pass
    else:
        mongodb_url = get_secret("MONGODB_URL")

    try:
        with MongoClient(mongodb_url) as client:
            db = client['test']
            collection = db['aidetectorresults']

            timestamp = datetime.now()
            document = {
                "image_path": image_path,
                "inference_results": inference_results,
                "created_at": timestamp
            }

            result = collection.insert_one(document)

            if result.inserted_id:
                response = {"message": "Saved successfully"}
                logger.info("Saved successfully to mongodb: %s", result.inserted_id)
            else:
                response = {"message": "The document could not be inserted"}

            return jsonify(response)

    except Exception as e:
        logger.error("Error to save to mongodb: %s", e)

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
        return jsonify({"error": "No selected image"}), 400

    logger.info("Received the image %s", file.filename)

    # Generate a random mnemonic filename with the original file extension
    _, ext = os.path.splitext(file.filename)
    random_name = f"{uuid.uuid4()}{ext}"

    # Save the file to a temporary location
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, random_name)
    file.save(image_path)

    logger.info("Image saved in temporal the location: %s", image_path)

    # Validate the image file
    try:
        validate_image_file(image_path)
        processed_image_path = compress_and_resize_image(image_path)
        logger.info("Image %s validated, resized and compressed", image_path)
    except ValueError as e:
        logger.error("Image %s is not valid: %s", image_path, e)
        return jsonify({"error": "The provided image format is not valid"}), 400

    logger.info("Uploading images to S3")

    # Upload original image
    upload_success, error_message = upload_image_to_s3(image_path, "aidetector-results")
    if not upload_success:
        logger.error("Error upload image to AWS: %s", error_message)
        return jsonify({"error": "There was an issue processing your image"}), 400

    # Upload processed image
    upload_success, error_message = upload_image_to_s3(processed_image_path, "aidetector-results")
    if not upload_success:
        logger.error("Error upload image to AWS: %s", error_message)
        return jsonify({"error": "There was an issue processing your image"}), 400
    
    # Start timing
    start_time = time()

    logger.info("Starting DM detection on %s", processed_image_path)

    # Run DM Detector
    dm_results = dm_process_image(processed_image_path, preloaded_models=dm_loaded_models)

    logger.info("Starting GAN detection on %s", processed_image_path)

    # Run GAN Detector
    gan_results = gan_process_image(processed_image_path, preloaded_models=gan_loaded_models)

    # Run EXIF detector
    original_filename = os.path.basename(file.filename)
    exif_results = exif_process_image(image_path, original_filename)

    # Run explainability generator
    preliminary_results = {
        "dMDetectorResults": dm_results,
        "gANDetectorResults": gan_results,
        "exifDetectorResults": exif_results,
    }

    logger.info("Starting explainability generator on %s", processed_image_path)

    craft_results = craft_explanation(processed_image_path, preliminary_results)

    # End timing
    end_time = time()
    total_execution_time = end_time - start_time

    # Combine results
    results = {
        "dMDetectorResults": dm_results,
        "gANDetectorResults": gan_results,
        "exifDetectorResults": exif_results,
        "explainabilityResults": craft_results,
        "totalExecutionTime": total_execution_time,
    }

    save_to_mongodb(image_path, results)

    return jsonify(results)


if __name__ == "__main__":

    preload_models()

    logger.info("Running on environment: %s", os.environ.get('DEV_ENV'))

    # Check if the 'DEV_ENV' environment variable is set to 'true'
    if os.environ.get('DEV_ENV') == 'true':
        # Run on port 8080 for development environment
        port = 8080
    else:
        # Run on port 80 otherwise
        port = 80

    # Log the port number
    logger.info("Successfully running the API on port %s", port)

    serve(app, host="0.0.0.0", port=port)

