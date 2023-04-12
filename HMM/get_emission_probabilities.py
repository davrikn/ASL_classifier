import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.special import softmax


def get_emission_probabilities(model: nn.Module, testloader: DataLoader) -> None:
    """
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
    N = len(testloader.dataset)
    batch_size = testloader.batch_size

    # Normalizing function
    def norm(data, axis=0):
        """
        This is an ugly function for ugly people
        """
        if not isinstance(data, np.ndarray):
            data = data.numpy()
        if axis == 1:
            data -= data.min(axis=1)[:, None]
            data_norm = np.linalg.norm(data, ord=1, axis=1)
            return data/data_norm[:, None]
        else:
            data -= data.min(axis=0)
            data_norm = np.linalg.norm(data, ord=1)
            return data/data_norm

    # norm = softmax

    def average_predictions(pred, labs):
        ems = np.ones((m, m))/m
        for x in range(m):
            if np.any(labs == x):
                ems[x] = norm(
                    pred[labs == x, :].mean(axis=0),
                    axis=0
                )
        
        return ems

    # Prediction:
    emissions = np.zeros((m, m))/m # Initialized as discrete uniform distribution.

    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images = images.float().to(device)
            labs = labels.to(device).numpy()
            pred = norm((model(images)).numpy(), axis=1)
            emissions = (i/(i+1))*emissions + (1/(i+1))*average_predictions(pred, labs)
            print(f"Emission probability calculation progress: {round(100*(i+1)*batch_size/N)} %")
            print(emissions.shape)
        
    return emissions