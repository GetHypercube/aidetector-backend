import os
import sys
import csv

sys.path.append("/mnt/efs")
sys.path.append("/mnt/efs/Libs")

import torch
import pandas
import numpy as np
import tqdm
import glob
import torchvision.transforms as transforms
import argparse

from third_party_libs.normalization import CenterCropNoPad, get_list_norm

# from third_party_libs.normalization2 import PaddingWarp
from PIL import Image
from getMethod import get_method_here, def_model

basePath = "/mnt/efs/"


def init(filePath, resultsPath, weightsPath, fileName):
    outroot = "/tmp" + resultsPath
    csvfilename = fileName

    # Initial setup and create folders
    if not os.path.exists(outroot):
        os.makedirs(outroot)
        print("Created results folders: ", os.listdir("/tmp"))

    # Get the uploaded images names and create a list of them
    image_files = []
    for imageName in os.listdir(filePath):
        if imageName.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
            image_files.append(os.path.join(filePath, imageName))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    ### list of models

    models_list = {
        "Grag2021_progan": "Grag2021_progan",
        "Grag2021_latent": "Grag2021_latent",
    }

    models_dict = dict()
    transform_dict = dict()

    for model_name in models_list:
        _, model_path, arch, norm_type, patch_size = get_method_here(
            models_list[model_name], weights_path=weightsPath
        )

        model = def_model(arch, model_path, localize=False)
        model = model.to(device).eval()

        transform = list()
        if patch_size is not None:
            if isinstance(patch_size, tuple):
                transform.append(transforms.Resize(*patch_size))
            else:
                if patch_size > 0:
                    transform.append(CenterCropNoPad(patch_size))
                else:
                    transform.append(CenterCropNoPad(-patch_size))
                    transform.append(PaddingWarp(-patch_size))
            transform_key = "custom"
        else:
            transform_key = "none"

        transform = transform + get_list_norm(norm_type)
        transform = transforms.Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)

    # Process the image
    img = Image.open(image_files[0]).convert("RGB")
    logits = {}
    with torch.no_grad():
        for model_name in models_dict:
            transformed_img = transform_dict[models_dict[model_name][0]](img)
            transformed_img = transformed_img.unsqueeze(0).to(device)
            out_tens = models_dict[model_name][1](transformed_img).cpu().numpy()

            # Check the shape of out_tens and handle accordingly
            print(
                f"Output tensor shape for {model_name}: {out_tens.shape}"
            )  # Debugging line
            if out_tens.size == 1:
                logits[model_name] = out_tens.item()
            else:
                logits[model_name] = np.mean(out_tens, (2, 3)).item()

    # Print the results

    print("Logits:")
    for model_name, logit in logits.items():
        print(f"{model_name}: {logit}")
    print(
        "Label: " + ("True" if any(value > 0 for value in logits.values()) else "False")
    )
