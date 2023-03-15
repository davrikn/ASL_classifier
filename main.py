import numpy as np
import cv2 as cv
import torch
import os.path as path
from os import listdir
import matplotlib.pyplot as plt
from rembg import remove

from models.transforms import BGRemover
from image_datasets.imagedataset import ImageDataset
from models.dropoutModel import DropoutModel
from predictor import predict, load_model

from tests.test_get_emission_probabilities import test_get_emission_probabilities

# cv.imshow("Window", img)
# cv.waitKey(0) & 0xFF

def main():
    model = DropoutModel()
    load_model(model, model_path="./models/saved/model_dropout_v1.pth")

if __name__ == "__main__":
    main()