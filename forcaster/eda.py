"This module includes eda functions."

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def get_image(storm_id, id, data_directory):
    """
    Return PIL image from storm id and id.

    Args:
        storm_id (str): The storm id.
        id (int): The image id.
        data_directory (str): The directory path where the images are located.

    Returns:
        PIL.Image.Image: The image.
    """
    # put zeros in front of id if it is less than 4 digits
    if len(str(id)) < 3:
        image_path = (
            Path(data_directory)
            / storm_id
            / f"{storm_id}_{'0' * (3-len(str(id)))}{id}.jpg"
        )
    else:
        image_path = Path(data_directory) / storm_id / f"{storm_id}_{id}.jpg"
    # if files are located directly under the folder
    if not image_path.exists():
        image_path = Path(data_directory) / f"{storm_id}_{id}.jpg"
    return Image.open(image_path)


def plot_images(df, data_directory, ascending=True):
    """
    Plots images of storms based on the given dataframe and data directory.

    Parameters:
    - df (pandas.DataFrame): The dataframe containing storm data.
    - data_directory (str): The directory where the storm images are stored.
    - ascending (bool, optional): Whether to sort the storms in ascending order.
        Default is True.
    """
    storm_id, id = top_storms(df, ascending=ascending)
    fig, ax = plt.subplots(1, 4, figsize=(15, 15))
    for i in range(4):
        ax[i].imshow(get_image(storm_id[i], id[i], data_directory), cmap="gray")
        ax[i].set_title(
            f'{storm_id[i]}, wind speed: {df.loc[id[i], "wind_speed"]}'
        )
        ax[i].axis("off")


def top_storms(df, ascending=False):
    """
    Find the top 4 storms based on wind speed.

    Args:
        df (DataFrame): The input DataFrame containing storm data.
        ascending (bool, optional): Whether to sort the storms in ascending
                                   order of wind speed. Defaults to False.

    Returns:
        tuple: A tuple containing two lists - top_storm_id and top_id.
               top_storm_id (list): The storm IDs of the top 4 storms.
               top_id (list): The IDs of the top 4 storms.
    """
    df_sorted = df.sort_values(
        by=["wind_speed"], ascending=ascending, ignore_index=True
    )
    top_storm_id1, top_id1 = df_sorted[["storm_id", "id"]].iloc[0]
    df_sorted = df_sorted[df_sorted["storm_id"] != top_storm_id1].reset_index(
        drop=True
    )

    top_storm_id2, top_id2 = df_sorted[["storm_id", "id"]].iloc[0]
    df_sorted = df_sorted[df_sorted["storm_id"] != top_storm_id2].reset_index(
        drop=True
    )

    top_storm_id3, top_id3 = df_sorted[["storm_id", "id"]].iloc[0]
    df_sorted = df_sorted[df_sorted["storm_id"] != top_storm_id3].reset_index(
        drop=True
    )

    top_storm_id4, top_id4 = df_sorted[["storm_id", "id"]].iloc[0]
    df_sorted = df_sorted[df_sorted["storm_id"] != top_storm_id4].reset_index(
        drop=True
    )

    top_storm_id = [top_storm_id1, top_storm_id2, top_storm_id3, top_storm_id4]
    top_id = [top_id1, top_id2, top_id3, top_id4]

    return top_storm_id, top_id
