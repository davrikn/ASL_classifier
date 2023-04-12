from numpy import std
from matplotlib.pyplot import imshow, colorbar
from torch.utils.data import DataLoader, random_split
from image_datasets.imagedataset import ImageDataset
from HMM.get_emission_probabilities import get_emission_probabilities
from models.dropoutModel3 import DropoutModel
from predictor import load_model
from models.transforms import norm_transform
import pickle


def test_get_emission_probabilities(N: int = 87000) -> None:
    """
    Function to test get_emission_probabilities().

    Args:
        N: Amount of images to use. Defualt: All
    Returns:
        None
    """
    assert 0 < N <= 87000
    images, _ = random_split(ImageDataset(transform=norm_transform), (N, 87000 - N))
    loader = DataLoader(images, batch_size=N, shuffle=True)
    model = DropoutModel()
    load_model(model, model_path="./models/saved/model_v3_1.pth")
    emission_probabilities = get_emission_probabilities(model, testloader=loader)
    with open("emission.txt", 'wb') as f:
        pickle.dump(emission_probabilities, f)
    c = imshow(emission_probabilities)
    colorbar(c)