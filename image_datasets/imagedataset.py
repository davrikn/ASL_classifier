from torch.utils.data import Dataset
from torchvision.io import read_image
from image_datasets.imagepathloader import ImagePathLoader


class ImageDataset(Dataset):

    """
    Loads ASL sign language alphabet data as a PyTorch dataset.
    """

    def __init__(self, transform=None, target_transform=None, base_path: str = './data/asl_alphabet_train') -> None:
        self.img_paths = ImagePathLoader.read_paths_and_classify(base_path)
        self.transform = transform                  # Transform of images
        self.target_transform = target_transform    # Transform of labels
    

    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self, ind: int) -> tuple:
        image_path, label = self.img_paths[ind]
        image = read_image(image_path)[:,4:-4, 4:-4].float() # This returns a tensor, ToTensor is therefore not necessary
        if self.transform: image = self.transform(image)
        if self.target_transform: label = self.target_transform(label)
        return image, label