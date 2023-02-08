import torch
import numpy as np
from torch.utils.data import DataLoader
from imagedataset import ImageDataset
from tqdm import tqdm

def get_image_stats(batch_size: int = 100, num_channels: int = 3) -> tuple:
    """
    Identifies mean and standard deviation of each channel for the
    entire dataset for use in Normalize transform.

    Args:
        batch_size (int): Loading all of the data will give a memory error.
                          This determines the batch size.
        num_channels (int)
    Returns:
        Tuple of the form (average, sd_hat)
    """
    dataset = ImageDataset()
    assert not len(dataset) % batch_size # Data length (87 000) must be divisible by batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size)
    num_batches = int(len(dataset) / batch_size)

    mu_hats = torch.zeros((num_batches, num_channels))
    vr_hats = torch.zeros((num_batches, num_channels))
    for i, (images, _) in enumerate(tqdm(dataloader)):
        images = images.float()
        mu_hats[i] = images.mean(dim=(0, 2, 3)) # Channel is the first axis of each batch member
        vr_hats[i] = images.var(dim=(0, 2, 3))

    return (mu_hats.mean(dim=0), np.sqrt(vr_hats.mean(dim=0)))