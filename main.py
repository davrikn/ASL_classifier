import numpy as np
import cv2 as cv
import torch as tr
import os.path as path
from os import listdir
from imageloader import ImageLoader
import matplotlib.pyplot as plt

# cv.imshow("Window", img)
# cv.waitKey(0) & 0xFF

if __name__ == "__main__":
    paths = ImageLoader.read_paths_and_classify()
    print(np.array(paths).shape)