import numpy as np
import cv2 as cv
import torch
import os.path as path
from os import listdir
import matplotlib.pyplot as plt

from image_datasets.imagedataset import ImageDataset
from models.dropoutModel import DropoutModel
from predictor import predict, load_model

# cv.imshow("Window", img)
# cv.waitKey(0) & 0xFF

def main():
    load_model(model:=DropoutModel(), "./models/saved/model_dropout_v3.pth")

    images = ImageDataset()
    image = images[0][0]
    print(image)

    pred = predict(model, images=image)
    print(pred)

if __name__ == "__main__":
    main()