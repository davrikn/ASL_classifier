import numpy as np
import cv2 as cv
import torch
import os.path as path
from os import listdir
import matplotlib.pyplot as plt
from rembg import remove
from torch.utils.data import DataLoader, random_split

from models.dropoutModel3 import DropoutModel
from models.transforms import BGRemover
from image_datasets.imagedataset import ImageDataset
from models.dropoutModel import DropoutModel
from predictor import predict, load_model

from tests.test_get_emission_probabilities import test_get_emission_probabilities, test_get_emission_probabilities_alt
from tests.test_TorchQueue import test_TorchQueue


# cv.imshow("Window", img)
# cv.waitKey(0) & 0xFF

def main():
    # test_TorchQueue()
    test_get_emission_probabilities()
    plt.show()

if __name__ == "__main__":
    main()