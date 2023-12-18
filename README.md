# aidetector-backend

## Requirements

- Python 3.11.6

## Installation

Follow these steps to set up and use the DMimageDetection project:

1. Clone the repository:
```
git clone https://github.com/GetHypercube/aidetector-backend.git
```

2. Create a Python virtual environment:
```
python -m venv .venv
```

3. Activate the Python environment:
- For Windows:
  ```
  .venv\Scripts\activate
  ```
- For OSX:
  ```
  source .venv/bin/activate
  ```

4. Upgrade pip:
```
python -m pip install --upgrade pip
```

5. Install the necessary requirements:
```
pip install -r requirements.txt
```

6. If you have a GPU, make sure Pytorch is compiled with GPU support. Here is the example installation command for Windows:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Please go to the following [link](https://pytorch.org/get-started/locally/) to find the detailed instructions.

7. Download the weights from this [link](https://aidetector-models.s3.amazonaws.com/weights.zip) and place them under `/weights` directory in your project folder.

8. You need to create a `.env` file with the following variables:

```bash
AWS_ACCESS_KEY=XXX
AWS_SECRET_KEY=XXX
OPENAI_API_KEY=XXX
MONGODB_URL=XXX
DEV_ENV=true
```
9. If you are going to use the API, please make sure to have the AWS CLI installed and properly configured.

## Usage

To run the detector on an image:

`python main.py --image_path image.jpg` 

To run the detector on a folder:

`python main.py --image_folder image.jpg` 

To run the detector on a folder and a specific models:

`python main.py --image_folder image.jpg --model dMDetectorResults` 

If you want to use the API, first run the server:

`python api.py`

To create the CSV file neccesary to calculate evaluations of the models:

`python main.py --image_folder tests --true_label True`

And to calculate evaluations, open `evaluations.ipynb` notebook.

In Windows, run the CURL command:

`curl -X POST -F "file=@real_biggan.png" http://localhost:80/detect`

Please see `main.py` for additional options

## Docker usage

To build and run the container for development (CPU):

```
docker build -t aidetector-backend .
docker run --rm -p 8080:80 aidetector-backend
```

To build and run the container for development (GPU):

```
docker build -t aidetector-backend-gpu -f Dockerfile.GPU .
docker run --rm --gpus all -p 8080:80 aidetector-backend-gpu
```

