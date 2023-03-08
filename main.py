import numpy as np
import cv2 as cv
import torch
import os.path as path
from os import listdir
import matplotlib.pyplot as plt

from image_datasets.imagedataset import ImageDataset
from models.dropoutModel import DropoutModel
from predictor import predict, load_model

from tests.test_get_emission_probabilities import test_get_emission_probabilities

# cv.imshow("Window", img)
# cv.waitKey(0) & 0xFF

def main():
    test_get_emission_probabilities(N=1500)
    plt.show()

if __name__ == "__main__":
    main()