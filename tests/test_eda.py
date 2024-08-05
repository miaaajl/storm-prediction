from PIL import Image
import os
import pandas as pd

from forecaster.eda import get_image, top_storms


def test_get_image():
    # Change the working directory to here
    os.chdir(os.path.dirname(__file__))

    # Define test data
    storm_id = "bkh"
    id = 1
    data_directory = "test_data"

    image = get_image(storm_id, id, data_directory)

    assert isinstance(image, Image.Image)


def test_top_storms():
    # Define test data
    df = pd.DataFrame(
        {
            "storm_id": ["bkh", "abc", "def", "ghi", "jkl"],
            "id": [1, 2, 3, 4, 5],
            "wind_speed": [100, 200, 150, 180, 120],
        }
    )

    # Call the function
    top_storm_id, top_id = top_storms(df)

    # Check the output
    assert top_storm_id == ["abc", "ghi", "def", "jkl"]
    assert top_id == [2, 4, 3, 5]
