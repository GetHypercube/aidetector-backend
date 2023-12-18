"""
This module provides functionality for interacting with AWS services such as S3 and Secrets Manager.
It includes functions for AWS session configuration, uploading images to S3, and retrieving secrets 
from AWS Secrets Manager.

Functions:
    upload_image_to_s3(image_path, bucket_name, object_name=None): Uploads an image file to the  
        specified S3 bucket. The object name in the bucket is optional and defaults to the image 
        file's basename. Returns True if the upload is successful, or an error message if it fails.
    
    get_secret(secret_name): Retrieves a specified secret from AWS Secrets Manager. The function is 
        preconfigured to work with a specific secret path ('aidetector-prod/'). It returns the 
        secret value or an error message in case of failure.

Exceptions:
    ClientError: An error thrown by boto3 if AWS service requests fail.
    NoCredentialsError: Raised by boto3 when AWS credentials are not provided or are invalid.
"""

import os
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def upload_image_to_s3(image_path, bucket_name, object_name=None):
    """
    Uploads an image to an AWS S3 bucket.

    Args:
        image_path (str): Path to the image file.
        bucket_name (str): S3 bucket name.
        object_name (str, optional): S3 object name. If not specified, image_path is used.

    Returns:
        bool: True if file was uploaded, else False.
    """
    if object_name is None:
        object_name = os.path.basename(image_path)

    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(image_path, bucket_name, object_name)
        return True, None
    except (ClientError, NoCredentialsError) as e:
        return False, str(e)


def get_secret(secret_name, region="us-east-1"):
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

    secret_name = "aidetector-prod/" + secret_name

    # Create a Secrets Manager client
    secret_client = boto3.client('secretsmanager', region)

    try:
        get_secret_value_response = secret_client.get_secret_value(SecretId=secret_name)

        # Decrypts secret using the associated KMS key.
        secret = get_secret_value_response["SecretString"]

        # Parse the JSON string and extract the specific secret value
        secret_dict = json.loads(secret)
        return secret_dict.get(secret_name.split('/')[-1], None)
    except (ClientError, NoCredentialsError) as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        return e
