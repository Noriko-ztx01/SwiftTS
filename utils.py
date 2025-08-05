import torch
import torch.backends.cudnn as cudnn

import os
import time
import random
import pickle
import numpy as np


def save_pickle(file_name, data):
    """
    Save data to a pickle file.

    Args:
        file_name (str): The name of the file to save the data.
        data: The data to be saved.
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.DEFAULT_PROTOCOL)


def load_pickle(file_name):
    """
    Load data from a pickle file.

    Args:
        file_name (str): The name of the file to load the data from.

    Returns:
        The loaded data.
    """
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def gpu_state(gpu_id, get_return=False):
    """
    Display or return the current state of specified GPU(s).

    Args:
        gpu_id (str): Comma-separated GPU IDs to check (e.g., '0,1').
        get_return (bool): If True, returns a dictionary with available memory per GPU.
                           If False, prints the GPU state information.

    Returns:
        dict or None: Dictionary mapping GPU ID to available memory (in MB) if get_return is True.
                      Otherwise, prints the state and returns None.
    """
    qargs = ['index', 'gpu_name', 'memory.used', 'memory.total']
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))

    results = os.popen(cmd).readlines()
    gpu_id_list = gpu_id.split(",")
    gpu_space_available = {}
    for cur_state in results:
        cur_state = cur_state.strip().split(", ")
        for i in gpu_id_list:
            if i == cur_state[0]:
                if not get_return:
                    print(f'GPU {i} {cur_state[1]}: Memory-Usage {cur_state[2]} / {cur_state[3]}.')
                else:
                    # Extract numeric values from memory strings and calculate available memory
                    total_memory = int("".join(filter(str.isdigit, cur_state[3])))
                    used_memory = int("".join(filter(str.isdigit, cur_state[2])))
                    gpu_space_available[i] = total_memory - used_memory
    if get_return:
        return gpu_space_available


def set_gpu(x, space_hold=1000):
    """
    Set the visible GPU(s) and wait until sufficient GPU memory is available.

    Args:
        x (str): Comma-separated GPU IDs to make visible (e.g., '0,1').
        space_hold (int): Minimum total available GPU memory (in MB) required before proceeding.
    """
    assert torch.cuda.is_available(), "CUDA is not available."
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for improved performance

    gpu_available = 0
    while gpu_available < space_hold:
        gpu_space_available = gpu_state(x, get_return=True)
        gpu_available = 0
        for gpu_id, space in gpu_space_available.items():
            gpu_available += space
        if gpu_available < space_hold:
            gpu_available = 0
            time.sleep(1800)  # Wait for 30 minutes before checking again
    gpu_state(x)  # Print final GPU state


def set_seed(seed):
    """
    Set the random seed for reproducibility across multiple libraries.

    Args:
        seed (int): The seed value to use for random number generators.
    """
    np.random.seed(seed=seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python hash seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For current GPU
    torch.cuda.manual_seed_all(seed)  # For all GPUs (in multi-GPU setup)
    cudnn.deterministic = True  # Make cuDNN operations deterministic
    cudnn.benchmark = False  # Disable cuDNN auto-tuner for reproducibility


def nan_assert(x):
    """
    Assert that the input tensor does not contain any NaN values.

    Args:
        x (torch.Tensor): The tensor to check for NaN values.

    Raises:
        AssertionError: If any NaN values are found in the tensor.
    """
    assert torch.any(torch.isnan(x)) == False, "Tensor contains NaN values."