import numpy as np
import cv2 as cv
import torch as tr
import os.path as path
from os import listdir
import matplotlib.pyplot as plt

from imagepathloader import ImagePathLoader
from imagedataset import ImageDataset
from get_image_stats import get_image_stats

# cv.imshow("Window", img)
# cv.waitKey(0) & 0xFF

if __name__ == "__main__":
    print(get_image_stats())