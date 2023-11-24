# aidetector-backend

## Requirements

- Python 3.11.6

## How to Install and Use

Follow these steps to set up and use the DMimageDetection project:

1. Clone the repository:
```
git clone https://github.com/grip-unina/DMimageDetection.git
```

2. Create a Python virtual environment:
```
python -m venv .env
```

3. Activate the Python environment:
- For Windows:
  ```
  .\.env\Scripts\activate
  ```
- For OSX:
  ```
  source env/bin/activate
  ```

4. Upgrade pip:
```
python -m pip install --upgrade pip
```

5. Install the necessary requirements**:
```
pip install -r requirements.txt
```

6. Download the weights from this [link](https://www.dropbox.com/s/pkj8p3v1gmm8t4p/weights.zip?dl=0) and place them under `.\weights` directory in your project folder.

## Usage

To run the detector on an image:

`python .\dmdetector.py --image_path example_images\diffusion\real\biggan_256\biggan_000_302875.png`

For just running the diffusion detector:

`python .\dmdetector.py --image_path example_images\diffusion\real\biggan_256\biggan_000_302875.png`

For just running the GAN detector:

`python .\gandetector.py --image_path example_images\gan\real\real_biggan.png`

If you want to see memory usage, you can use the `—debug` flag.
