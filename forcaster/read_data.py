"This module includes functions to read data."

from pathlib import Path
import pandas as pd
import json
import warnings

warnings.filterwarnings("ignore")


def read_tabular_data(data_directory):
    """
    Read tabular features and labels from JSON files in the specified directory,
    and return a merged dataframe.

    Args:
        data_directory (str): The directory path where the JSON files are located.

    Returns:
        pandas.DataFrame: A dataframe containing the merged features and labels.
    """

    # Check if the directory exists
    if not Path(data_directory).exists():
        raise FileNotFoundError(f"Directory {data_directory} does not exist.")

    # Check if there is any JSON file in the directory
    if len(list(Path(data_directory).glob("*.json"))) == 0:
        features_paths = list(Path(data_directory).glob("*/*features.json"))
        labels_paths = list(Path(data_directory).glob("*/*label.json"))
    else:
        features_paths = list(Path(data_directory).glob("*features.json"))
        labels_paths = list(Path(data_directory).glob("*label.json"))

    # Read in features to a list of dictionaries
    features = []
    id = []
    storm_id = []
    for path in features_paths:
        # print(path)
        with open(path) as file:
            features.append(json.load(file))
            id.append(
                int(path.stem.split("_")[1])
            )  # Get the image id from the path and convert it to integer
            storm_id.append(path.stem.split("_")[0])
    # create a df with features and id as columns
    df_features = pd.DataFrame(features)
    df_features["id"] = id
    df_features["storm_id"] = storm_id

    # Read in labels to a list of dictionaries
    labels = []
    id = []
    storm_id = []
    for path in labels_paths:
        with open(path) as file:
            labels.append(json.load(file))
            id.append(
                int(path.stem.split("_")[1])
            )  # Get the image id from the path and convert it to integer
            storm_id.append(path.stem.split("_")[0])
    df_labels = pd.DataFrame(labels)
    df_labels["id"] = id
    df_labels["storm_id"] = storm_id

    # Merge features and labels into a single dataframe
    df = pd.merge(df_features, df_labels, on=["id", "storm_id"])

    # Convert numerical columns to integers
    df[["wind_speed", "relative_time"]] = df[
        ["wind_speed", "relative_time"]
    ].astype(int)

    # Sort the dataframe by the name and id
    df.sort_values(by=["storm_id", "id"], inplace=True)

    return df
