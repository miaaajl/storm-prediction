import pytest
import os

from forecaster.read_data import read_tabular_data


def test_read_tabular_data():
    # Change the working directory to here
    os.chdir(os.path.dirname(__file__))

    # Read reference storm data
    data_directory = "test_data/bkh"
    df = read_tabular_data(data_directory)

    # Check that the data frame has the expected shape
    expected_shape = (6, 5)
    assert df.shape == expected_shape

    # Check that the data frame is sorted by relative_time
    expected_times = [0, 1801, 3600, 5400, 7200, 10802]
    assert list(df.relative_time) == expected_times

    # Check if the directory does not exist
    data_directory = "data/gkf_invalid"
    with pytest.raises(FileNotFoundError):
        read_tabular_data(data_directory)
