# Use Python 3.11.6 as the base image
FROM python:3.11.6

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Install wget and unzip (if not already installed)
RUN apt-get update && apt-get install -y wget unzip

# Download and unzip the model weights
RUN wget -O weights.zip https://aidetector-models.s3.amazonaws.com/weights.zip \
    && unzip weights.zip -d /app/ \
    && rm weights.zip

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container at /app
COPY . /app

# Make port 8080 available outside this container
EXPOSE 8080

# Run api.py when the container launches
CMD ["python", "api.py"]
