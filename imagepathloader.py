import numpy as np
import cv2 as cv
from os import listdir

class ImagePathLoader:

    def read_paths_and_classify(base_path: str = './data/asl_alphabet_train'):
        """
        Create path and label for all images in training-set
        :param base_path: Path to directory containing subfolders with images
        :return: (87000x2) array with paths in the first column and labels in the second
        """
        classes = dict()
        images = []
        class_id = 0
        for c in listdir(base_path):
            classes[c] = class_id
            for file in listdir(base_path + "/" + c):
                images.append([base_path + "/" + c + "/" + file, class_id])
            class_id += 1
        return images


    # def load_image_from_path(path: str):
    #     """
    #     Load an image from path in color and return it as an array
    #     :param path: Path to a specific image
    #     :return: (200x200x3) array with the image
    #     """
    #     img = cv.imread(path, cv.IMREAD_COLOR)
    #     img = np.array(img)
    #     return img