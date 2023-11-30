import os
import requests
import boto3
from botocore.exceptions import ClientError
from utils import encode_image

# Hardcode the API key
# api_key = "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"
# Load the API key from an environment variable
# api_key = os.getenv("OPENAI_API_KEY")

def get_secret():

    secret_name = "aidetector-prod/OPENAI_API_KEY"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    
    return secret

api_key = get_secret()

def craft_explanation(image_path, analysis_results):
    """
    Sends an image and its analysis results to GPT Vision for crafting an explanation.
    """

    # Getting the base64 string
    base64_image = encode_image(image_path)

    # Craft the prompt for the explanation
    prompt = f"""
    The following JSON results from predicting whether the image is synthetic with 4 models, 2 GAN, and 2 diffusion detection models.

    If any prediction probabilities cross a 0.5 threshold, the image is tagged as synthetic by either the GAN or the diffusion detector, or both.

    Please describe in less than 50 words the image provided, and do some explainability of why the models detected the image as AI-generated or not. Make it funny.
    
    {analysis_results}
    """

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            # Convert the response to JSON format
            data = response.json()
            return data
        else:
            return(f"Error: {response.status_code}")
    except Exception as e:
        return str(e)