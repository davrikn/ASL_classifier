import numpy as np
import cv2 as cv
import torch as tr
import os.path as path
from os import listdir
from imageloader import ImageLoader

img = cv.imread("./data/asl_alphabet_train/asl_alphabet_train/A/A1.jpg", cv.IMREAD_GRAYSCALE)
# cv.imshow("Window", img)
# cv.waitKey(0) & 0xFF
# Test comment

img = np.array(img).flatten()
print(img)
files = 0


ImageLoader.read_paths_and_classify()

