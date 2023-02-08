import numpy as np
import cv2 as cv
import torch as tr
import os.path as path
from os import listdir
from imagepathloader import ImagePathLoader
from imagedataset import ImageDataset
import matplotlib.pyplot as plt

# cv.imshow("Window", img)
# cv.waitKey(0) & 0xFF

if __name__ == "__main__":
    data = ImageDataset()
    print(data[0])