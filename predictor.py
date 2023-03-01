import torch
import torch.nn as nn
import numpy as np


def predict(model: nn.Module, model_path: str, images: torch.Tensor, device = "cpu") -> torch.tensor:
    """
    Given the path to a trained PyTorch model, and one or more images to
    predict, returns predictions for all images.

    Args:
        model       (nn.Module): A blank (untrained) model of the kind you want to load
        model_path        (str): Path to saved model
        images (PyTorch tensor): Pytorch tensor of one or more images to predict
    Returns:
        Predictions for each image in raw format
    """
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()
    images = images.float().to(device)
    with torch.no_grad():
        return model(images)