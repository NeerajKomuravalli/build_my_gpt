import os
import torch

def read_data(file_path) -> str:
    if not os.path.exists(file_path):
        raise FileExistsError(f"{file_path} is not a valid path")
    data = []
    with open(file_path, "r") as f:
        data = f.readlines()
    
    data_str = "\n".join(data)

    return data_str

def get_batch(block_size, batch_size, data_tensor):
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])

    return x, y