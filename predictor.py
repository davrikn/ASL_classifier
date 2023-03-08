import torch
import torch.nn as nn
import numpy as np


def load_model(model: nn.Module, model_path) -> None:
    """
    Given a blank model of the desired type, and a path to the dictionary it
    is saved to, loads the model parameters into the model INPLACE and calls
    .eval().

    Args:
        model (nn.Module): A blank (untrained) model of the kind you want to load
        model_path  (str): Path to saved model
    Returns:
        Trained PyTorch model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(
        torch.load(
        model_path, map_location=device)
    )
    model.eval()


def predict(model: nn.Module, images: torch.Tensor) -> torch.tensor:
    """
    Given the path to a trained PyTorch model, and one or more images to
    predict, returns predictions for all images.

    Args:
        model       (nn.Module): Trained PyTorch model
        images (PyTorch tensor): Pytorch tensor of one or more images to predict
    Returns:
        Predictions for each image in raw format
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = images.float().to(device)
    with torch.no_grad():
        return model(images)