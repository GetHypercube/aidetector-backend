"""
This module provides functionality to interact with GPT Vision from OpenAI for image analysis. 
It includes methods to retrieve API keys from AWS Secrets Manager, encode images, and 
send requests to OpenAI's API for crafting explanations of images based on analysis results.

Functions:
- get_secret: Retrieves an API key from AWS Secrets Manager.
- craft_explanation: Sends an image and its analysis results to GPT Vision and crafts an 
                        explanation.
"""

import requests
import boto3
from botocore.exceptions import ClientError
from utils.general import encode_image, setup_logger

logger = setup_logger(__name__)


def get_secret():
    """
    Retrieves the OpenAI API key stored in AWS Secrets Manager.

    This function fetches the secret named 'aidetector-prod/OPENAI_API_KEY' from AWS Secrets 
    Manager in the 'us-east-1' region. It handles any client errors during the retrieval 
    process.

    Returns:
        str: The secret string containing the OpenAI API key, or an exception message 
                if an error occurs.

    Raises:
        ClientError: An error thrown by boto3 if AWS Secrets Manager request fails.
    """

    secret_name = "aidetector-prod/OPENAI_API_KEY"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        # Decrypts secret using the associated KMS key.
        secret = get_secret_value_response["SecretString"]
        return secret
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        return e


# Hardcode the API key
# logger.debug("This is the secret for OPENAI %s", get_secret())
# API_KEY = "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"
# Load the API key from an environment variable
# API_KEY = os.getenv("OPENAI_API_KEY")
# Load the API from AWS Secret Manager
API_KEY = get_secret()


def craft_explanation(image_path, analysis_results):
    """
    Sends an image and its analysis results to GPT Vision for crafting an explanation.

    This function encodes the image located at 'image_path' to a base64 string, 
    constructs a prompt including the analysis results, and sends a request to OpenAI's 
    GPT-4 Vision model. The function handles the response and returns a crafted 
    explanation or an error message.

    Args:
        image_path (str): The file path of the image to be analyzed.
        analysis_results (str): A JSON string containing the results of previous image analysis.

    Returns:
        str: The explanation from GPT Vision if successful, or an error message.

    Exceptions:
        Exception: Generic exception for any errors during the request to OpenAI's API.
    """

    logger.debug("Inside the craft_explanation")

    # Getting the base64 string
    base64_image = encode_image(image_path)

    # Craft the prompt for the explanation
    prompt = f"""
    The following JSON results are from predicting whether the image is synthetic with 5 different models. 
    2 models finetuned to detect GANs synthetic images, 2 models are finetuned for diffusion models and 
    an analysis of the EXIF and metadata information of the image is made.
    For the GAN and diffusion models, if any prediction probabilities cross a 0.5 threshold, the image is 
    classifies the image as synthetic. 
    If specific meta data is encountered in the image, the EXIF detector classifies the image as synthetic.
    Please in less than 50 words describe what is the image the user provided and 
    explain the results of the detectors. Make it funny, but make it clear at the end if it is synthetic or not.
    
    {analysis_results}
    """

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        logger.debug("OPENAI response is %s", response)
        if response.status_code == 200:
            # Convert the response to JSON format
            data = response.json()
            logger.debug("Response choice: %s", data["choices"][0])
            result = data["choices"][0]["message"]["content"]
            return result
        else:
            logger.warning("Received an invalid response from OpenAI")
            return "Our explainability model is having some issues today"
    except Exception as e:
        logger.warning("Received an invalid response from OpenAI %s", e)
        return "Our explainability model is having some issues today"
