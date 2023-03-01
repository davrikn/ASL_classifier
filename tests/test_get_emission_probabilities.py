from torch.utils.data import DataLoader, random_split
from image_datasets.imagedataset import ImageDataset
from HMM.get_emission_probabilities import get_emission_probabilities
from models.dropoutModel import DropoutModel
from predictor import load_model


def test_get_emission_probabilities(N: int = 87000) -> None:
    """
    Function to test get_emission_probabilities().

    Args:
        N: Amount of images to use. Defualt: All
    Returns:
        None
    """
    assert 0 < N <= 87000
    images, _ = random_split(ImageDataset(), (N, 87000 - N))
    loader = DataLoader(images, batch_size=1000, shuffle=False)
    model = DropoutModel()
    load_model(model, model_path="./models/saved/model_dropout_v3.pth")
    emission_probabilities = get_emission_probabilities(model, testloader=loader)
    print(emission_probabilities)