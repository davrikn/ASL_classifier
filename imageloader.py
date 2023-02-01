import numpy as np
import cv2 as cv
from os import listdir

#Test

class ImageLoader:

    def read_paths_and_classify(base_path: str = './data/asl_alphabet_train/asl_alphabet_train'):
        """
        Create path and label for all images in training-set
        :param base_path:
        :return:
        """
        classes = dict()
        images = []
        class_id = 0
        for c in listdir(base_path):
            classes[c] = class_id
            for file in listdir(base_path + "/" + c):
                images.append([base_path + "/" + c + "/" + file, class_id])
            class_id += 1
        print(classes)
        return images

    def load_image_from_path(self, path: str):
        """
        Load an image from path as grayscale and return it as an array
        :param path:
        :return:
        """
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img = np.array(img).flatten()
        return img