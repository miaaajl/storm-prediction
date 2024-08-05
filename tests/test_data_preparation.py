import pytest
from forecaster.data_preparation import ImageDataset
from torchvision import transforms
from PIL import Image
import os

# Change the working directory to here
os.chdir(os.path.dirname(__file__))

expected_num_samples = 6

data_directory = "test_data/bkh"


class TestImageDataset:
    @pytest.fixture
    def sample_dataset(self):
        return ImageDataset(data_directory, transform=None)

    def test_data_loading(self, sample_dataset):
        # Check if the number of samples matches your expectations
        assert len(sample_dataset) == expected_num_samples

        # Check if loading a sample works without errors
        sample, label = sample_dataset[0]

        # Add more specific assertions based on your dataset structure
        assert isinstance(sample, Image.Image)
        assert isinstance(label, int)

    def test_transformations(self):
        # Test data transformations
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        dataset = ImageDataset(data_directory, transform=transform)

        # Verify that transformations are applied correctly
        sample, label = dataset[0]

        # Add more specific assertions based on your transformations
        assert sample.shape == (1, 224, 224)  # Assuming RGB images

    def test_load_label(self):
        # Test the load_label method
        dataset = ImageDataset(data_directory, transform=None)

        # Assuming you have a known label file for the first image
        image_path = dataset.image_paths[0]
        label = dataset.load_label(image_path)

        # Add assertions based on your expectations
        assert isinstance(label, int)
