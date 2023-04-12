import json
import os.path
import random

import numpy
import numpy as np
import torch
from torchvision.transforms import transforms
from rembg import remove
from PIL import Image
from image_datasets.imagedataset import ImageDataset
from utility.imagePathDict import ImagePathDict
from os.path import isdir, isfile
from os import mkdir
from scipy.ndimage.interpolation import rotate

# Instance of Normalize transform with relevant parameters
norm_transform = transforms.Normalize(
    (132.3501, 127.2977, 131.0638),
    (55.5031, 62.3274, 64.1869)
)

# Background remover transform:
class BGRemover():

    def __call__(self, image):
        mask = remove(image[0].numpy(), only_mask=True) < 125
        image[:, mask] = 0 # Setting backroung pixels to 0
        return image

class BGReplacer():
    background: np.ndarray = torch.tensor(np.random.randint(0, 255, (3, 192, 192), dtype=numpy.uint8))

    @staticmethod
    def set_background(background: np.ndarray) -> None:
        size = background.size
        if size[0] != 3 or size[1] != 192 or size[2] != 192:
            raise Exception(f"Cannot use background of dimensions {size[0]}x{size[1]}, must be 192x192")
        BGReplacer.background = torch.tensor(background)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        mask = remove(image[0].numpy(), only_mask=True) < 125
        for i in range(192):
            for j in range(192):
                if mask[i][j]:
                    for k in range(3):
                        image[k][i][j] = BGReplacer.background[k][i][j]
        return image

if __name__ == "__main__":
    o = 0
    if isfile('./o.txt'):
        with open('./o.txt', 'r') as f:
            o = int(f.read())

    print(f"O: {o}")
    ds = ImageDataset(transform=BGReplacer(), base_path='../data/asl_alphabet_train')
    classes = ImagePathDict(True)
    class_idx = {val: o for val in list(classes.values())}
    for c in classes.values():
        classdir = f'../data/asl_alphabet_train_modified/{c}'
        if not isdir(classdir):
            mkdir(classdir)

    rand = random.Random()
    for i in range(o%3000, 3000):
        if i % 1 == 0:
            print(f"Image index at: {i}")

        for k in range(29):
            im = ds[k*3000 +i]
            imclass = classes[im[1]]
            im = im[0].numpy().transpose(1, 2, 0).astype(numpy.uint8)

            rot_angle = [0, 90, 180, 270][rand.randint(0, 3)]
            flip = bool(rand.getrandbits(1))

            if rot_angle != 0:
                rotate(im, rot_angle)

            im = Image.fromarray(im)
            im.save(f'../data/asl_alphabet_train_modified/{imclass}/{imclass}{class_idx[imclass]}.jpg')
            class_idx[imclass] += 1
        o += 1

    with open('./o.txt', 'w') as f:
        f.write(str(o))