import numpy as np
import cv2 as cv
import torch as tr
import os.path as path
from os import listdir

img = cv.imread("./data/asl_alphabet_train/asl_alphabet_train/A/A1.jpg", cv.IMREAD_GRAYSCALE)
# cv.imshow("Window", img)
# cv.waitKey(0) & 0xFF

img = np.array(img).flatten()
print(img)
files = 0


def read_paths_and_classify(base_path: str = './data/asl_alphabet_train/asl_alphabet_train'):
    """
    Create all 
    :param base_path:
    :return:
    """
    images = []
    class_id = 0
    for c in listdir(base_path):
        class_id += 1
        for file in listdir(base_path + "/" + c):
            images.append([base_path + "/" + c + "/" + file, class_id])
    return images


def load_image_from_path(path: str):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = np.array(img).flatten()
    return img


print(read_paths_and_classify())

