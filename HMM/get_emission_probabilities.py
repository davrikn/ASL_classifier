import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_emission_probabilities(model: nn.Module, testloader: DataLoader, normalizer: str = "softmax") -> None:
    """
    WARNING: THIS FUNCTION IS NOT TESTED

    Given a trained pytorch model and related testing data, estimates and returns
    an emission probability matrix, i.e. P( Y | X = x ) for all true classes X,
    and all predicted classes Y. This is done by summing up individual distributions
    for each x and normalizing.

    Args:
        model          (PyTorch module): Trained PyTorch model
        testloader (PyTorch Dataloader): Dataloader for testing data
        normalizer             (string): Function used to normalize distributions
    Returns:
        2D mxm numpy array of estimated emission probabilities, where
        m is the amount of classes. Rows indicate true class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # TODO: Figure out how to retrieve m from model or trainloader
    m = 29

    # Assigning normalizer
    if normalizer == "softmax": normalizer = nn.Softmax(dim=1)
    elif normalizer == "norm":  normalizer = lambda x: x/np.linalg.norm(x, ord=1)
    else: raise RuntimeError("Normalizer not recognized. Please use either \"softmax\" or \"norm\".")

    # Prediction for all test data
    pred = []
    labs = np.empty(0, dtype=object)

    with torch.no_grad():

        for (images, labels) in testloader:
            images = images.float().to(device)
            labels = labels.to(device)
            
            # Assuming shape m x n (n amount of datapoints)
            pred.append(normalizer(model(images)).numpy())
            labs = np.append(labs, labels)
        
    # Emission probabilities:
    emissions = np.ones((m, m))/m # Initialized as discrete uniform distribution.

    for x in range(m):
        if np.any(labs == x):
            emissions[x] = normalizer(
                pred[labs == x, :].mean(axis=0).flatten()
            )
        
    return emissions