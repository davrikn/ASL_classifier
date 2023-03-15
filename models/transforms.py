import numpy as np
import torch
from torchvision.transforms import transforms
from rembg import remove

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