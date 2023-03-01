import numpy as np
import cv2 as cv
import torch as tr
import os.path as path
from os import listdir
import matplotlib.pyplot as plt

from image_datasets.imagedataset import ImageDataset
from models.dropoutModel import DropoutModel
from predictor import predict

# cv.imshow("Window", img)
# cv.waitKey(0) & 0xFF

def main():
    images = ImageDataset()
    image = images[0][0]
    print(image.shape)

    pred = predict(DropoutModel(), "./models/saved/model_dropout_v3.pth", images=image)
    print(pred)


if __name__ == "__main__":
    main()