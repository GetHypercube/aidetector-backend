# Imports are always needed
import torch

# Get index of currently selected device
torch.cuda.current_device()  # Returns 0 in my case

# Get number of GPUs available
torch.cuda.device_count()  # Returns 1 in my case

# Get the name of the device
torch.cuda.get_device_name(0)  # Good old Tesla K80

# Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024 ** 3, 1), 'GB')
