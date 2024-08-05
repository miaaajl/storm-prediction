"""
This module includes all the functions and classes used in our
Feature based LSTM model for wind speed prediction.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from livelossplot import PlotLosses
import warnings

warnings.filterwarnings("ignore")


def interpolate_df(df, timestep=1800):
    """
    Expand the dataframe and interpolate so that there is a wind speed for every
    multiple of the timestep for each stoem in the dataframe(non-integer
    multiples get rounded to the nearest integer multiple).

    Args:
        df (pandas.DataFrame): a dataframe containing the sorm_id, relative_time,
        wind_speed and ocean

        timestep (int): the timestep to interpolate to

    Returns:
        pandas.DataFrame: A dataframe with the same columns as the input
        dataframe, but with one row for each multiple of the timestep for each
        storm in the input dataframe and with the wind_speed
        interpolated for each row.
    """

    unique_storms = df["storm_id"].unique()
    df.drop(columns=["id"], inplace=True)
    interpolated_dfs = []
    for storm in unique_storms:
        storm_df = df.loc[df["storm_id"] == storm]
        storm_df.loc[:, "relative_time"] = storm_df["relative_time"].apply(
            lambda x: round(x / timestep)
        )
        complete_timestamps = pd.DataFrame(
            {
                "relative_time": range(
                    int(storm_df["relative_time"].min()),
                    int(storm_df["relative_time"].max()) + 1,
                )
            }
        )
        merged_df = pd.merge(
            complete_timestamps, storm_df, on="relative_time", how="left"
        )
        merged_df["wind_speed"] = merged_df["wind_speed"].interpolate(
            method="linear"
        )
        merged_df["ocean"] = storm_df["ocean"].iloc[0]
        merged_df["storm_id"] = storm
        merged_df["relative_time"] *= timestep
        interpolated_dfs.append(merged_df)
    interpolated_df = pd.concat(interpolated_dfs)
    return interpolated_df


def create_dataset(dataset, lookback, shift=1):
    """
    Create a dataset for time series forecasting.

    Args:
        dataset (list or numpy.ndarray): The input time series data.
        lookback (int): The number of previous time steps to use as features.
        shift (int, optional): The number of time steps to shift the target.
            Defaults to 1.

    Returns:
        tuple: A tuple containing the input features (X) and the corresponding
            target values (y).
    """
    X, y = [], []
    for i in range(len(dataset) - lookback - shift + 1):
        feature = dataset[i : i + lookback]
        target = dataset[i + shift : i + lookback + shift]
        X.append(feature)
        y.append(target)
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))


class WindModel(nn.Module):
    """
    A PyTorch module for wind prediction.

    Args:
        hidden_size (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of recurrent layers in the LSTM.

    Attributes:
        lstm (nn.LSTM): The LSTM layer for sequence modeling.
        linear (nn.Linear): The linear layer for output prediction.
    """

    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


def df_to_loaders(
    df,
    lookback=20,
    shift=1,
    train_batch_size=32,
    test_batch_size=200,
    train_ratio=0.8,
):
    """
    Converts a DataFrame into PyTorch DataLoader objects for training and testing.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the time series data.
        lookback (int, optional): The number of previous time steps to include in the
            input sequence. Defaults to 20.
        shift (int, optional): The number of time steps to shift the target variable.
            Defaults to 1.
        train_batch_size (int, optional): The batch size for the training DataLoader.
            Defaults to 32.
        test_batch_size (int, optional): The batch size for the testing DataLoader.
            Defaults to 200.
        train_ratio (float, optional): The ratio of data to use for training.
            Defaults to 0.8.

    Returns:
        tuple: A tuple containing the training DataLoader and testing DataLoader.
    """
    df = interpolate_df(df)
    unique_storms = df["storm_id"].unique()
    timeseries_list = [
        df.loc[df["storm_id"] == storm_id, ["wind_speed"]].values.astype(
            "float32"
        )
        for storm_id in unique_storms
    ]

    X, y = [], []
    for timeseries in timeseries_list:
        X_, y_ = create_dataset(timeseries, lookback=lookback, shift=shift)
        X.append(X_)
        y.append(y_)
    X = torch.cat(X)
    y = torch.cat(y)

    indices = torch.randperm(len(X))
    X = X[indices]
    y = y[indices]

    X_train, y_train = (
        X[: int(len(X) * train_ratio)],
        y[: int(len(X) * train_ratio)],
    )
    X_test, y_test = (
        X[int(len(X) * train_ratio) :],
        y[int(len(X) * train_ratio) :],
    )

    train_loader = data.DataLoader(
        data.TensorDataset(X_train, y_train),
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = data.DataLoader(
        data.TensorDataset(X_test, y_test),
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def train(model, optimizer, criterion, data_loader, loss_window=0):
    """
    Trains the model using the given optimizer and criterion on the
        provided data_loader.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        data_loader (torch.utils.data.DataLoader): The data loader for training data.
        loss_window (int, optional): The number of time steps to consider for
            loss calculation. Defaults to 0.

    Returns:
        float: The average training loss.
    """
    model.train()
    train_loss = 0
    for batch_idx, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output[:, -loss_window:, :], y[:, -loss_window:, :])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)


def validate(model, criterion, data_loader, loss_window=0):
    """
    Validates the model using the given criterion and data loader.

    Args:
        model (torch.nn.Module): The model to be validated.
        criterion (torch.nn.Module): The criterion used for calculating the loss.
        data_loader (torch.utils.data.DataLoader): The data loader for loading
            the validation data.
        loss_window (int, optional): The number of time steps to consider for
            calculating the loss. Defaults to 0, i.e. considering all time steps.

    Returns:
        float: The average test loss over the validation data.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            output = model(X)
            loss = criterion(
                output[:, -loss_window:, :], y[:, -loss_window:, :]
            )
            test_loss += loss.item()
    return test_loss / len(data_loader)


def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    epochs,
    loss_window=0,
):
    """
    Trains a model using the specified optimizer and criterion
        for the specified number of epochs.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for
            the training dataset.
        test_loader (torch.utils.data.DataLoader): The data loader for
            the test dataset.
        epochs (int): The number of epochs to train the model.
        loss_window (int, optional): The size of the loss window for
            tracking the average loss.
            Defaults to 0, which means considering all time steps.

    Returns:
        dict: The state dictionary of the best model based on the validation loss.
    """
    best_loss = np.inf
    best_model = None

    liveloss = PlotLosses()
    for epoch in range(epochs):
        logs = {}
        train_loss = train(
            model, optimizer, criterion, train_loader, loss_window=loss_window
        )
        test_loss = validate(
            model, criterion, test_loader, loss_window=loss_window
        )
        logs["loss"] = np.log(train_loss)
        logs["val_loss"] = np.log(test_loss)
        liveloss.update(logs)
        liveloss.send()

        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model.state_dict()

    return best_model


def predict_future(model, list, future_steps):
    """
    Predicts the future values using the given model and input list.

    Args:
        model (torch.nn.Module): The trained model used for prediction.
        list (torch.Tensor): The input list of values.
        future_steps (int): The number of future steps to predict.

    Returns:
        list: A list of predicted future values.
    """
    future = []
    model.eval()
    with torch.no_grad():
        for i in range(future_steps):
            output = model(list)[:, -1, :].unsqueeze(1)
            future.append(float(output.flatten().__getitem__(0)))
            list = torch.cat((list[:, 1:, :], output), dim=1)
    return future
