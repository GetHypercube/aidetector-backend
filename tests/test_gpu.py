"""
This module provides utilities for working with CUDA-enabled GPUs using PyTorch.

It includes functions to get information about the available CUDA devices and to set
up the default device for PyTorch computations. The module checks for the availability
of CUDA-compatible GPUs and, if available, sets PyTorch to use the GPU. Otherwise,
it defaults to using the CPU.

The module also provides functionality to retrieve and display details about the
selected CUDA device, such as the device name, the amount of memory allocated, and
the amount of memory cached.

Functions:
    - torch.cuda.current_device(): Returns the index of the currently selected CUDA device.
    - torch.cuda.device_count(): Returns the number of available CUDA devices.
    - torch.cuda.get_device_name(): Gets the name of a specific CUDA device.
    - torch.cuda.memory_allocated(): Returns the amount of memory allocated 
      on the current CUDA device.
    - torch.cuda.memory_reserved(): Returns the amount of memory reserved on the 
      current CUDA device.
"""
import torch

# Get index of currently selected device
torch.cuda.current_device()  # Returns 0 in my case

# Get number of GPUs available
torch.cuda.device_count()  # Returns 1 in my case

# Get the name of the device
torch.cuda.get_device_name(0)  # Good old Tesla K80

# Setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

# Additional Info when using cuda
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
