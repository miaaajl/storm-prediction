""""This module includes functions and classes such as Custom Dataset
 that are used in developing our model."""

from pathlib import Path
import json
from torch.utils.data import Dataset
from PIL import Image


# Create a custom dataset class for the images
class ImageDataset(Dataset):
    """
    A dataset class that returns an image and its label as wind speed.

    Inherits from torch.utils.data.Dataset class.

    Args:
        data_directory (str): The directory path where the image data is stored.
        transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. Default: None.
    """

    def __init__(self, data_directory: str, transform=None):
        # data path
        self.data_directory = data_directory

        # sorted image paths
        if len(list(Path(self.data_directory).glob("*.jpg"))) == 0:
            self.image_paths = list(Path(self.data_directory).glob("*/*.jpg"))
        else:
            self.image_paths = list(Path(self.data_directory).glob("*.jpg"))
        self.image_paths = sorted(
            self.image_paths,
            key=lambda path: (
                path.stem.split("_")[0],  # First criterion
                int(path.stem.split("_")[1]),
            ),
        )  # Second criterion

        # initialise transform
        self.transform = transform

    def load_label(self, image_path):
        "Returns the label of the input image."
        label_path = (
            image_path.parent / f"{image_path.stem.split('_')[0]}_"
            f"{int(image_path.stem.split('_')[1]):03d}_label.json"
        )
        with open(label_path) as f:
            label = json.load(f)
        return int(label["wind_speed"])

    def load_image(self, idx) -> (Image.Image, int):
        "Returns the PIL image from index and its label."
        image_path = self.image_paths[idx]
        label = self.load_label(image_path)
        return Image.open(image_path), label

    def __len__(self):
        "Returns the total number of samples."
        return len(self.image_paths)

    def __getitem__(self, idx):
        "Returns one sample of data as a tensor."
        # load image
        image, label = self.load_image(idx)

        # apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label
