import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
from numpy import ndarray


def split_dataset(df: pd.DataFrame, test_size: float, random_seed: float = 42)\
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function splits a dataset into train, test and validation split. Based on the test_size float
    argument first the df will be split into two, in this case train and test with the given ratio.
    Then, we split again the test set into 1:1 ratio, and return with train, test, val sets for training.
    Random seed is used to enforce reproducibility.
    Args:
        df (pd.DataFrame): DataFrame object.
        test_size (float): Test size ratio compared to the training set. Eg.: 0.15.
        random_seed (float): Random seed for reproducibility.

    Returns:
        Tuple of DataFrame objects.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_seed)
    test, validation = train_test_split(test, test_size=0.5, random_state=random_seed)
    return train, validation, test


def transform_numerical_features(*dataframes: Tuple[pd.DataFrame]) -> tuple[ndarray, ...]:
    """
    This function takes arbitrary number of dataframes and transforms normalizes its numerical features and then
    returns it as a numpy array. This function only works if every column is numerical.
    Args:
        *dataframes: Tuple of DataFrame objects.

    Returns:
        Tuple of ndarrays
    """
    return tuple(
        StandardScaler().fit_transform(df)
        for df in dataframes
    )


def transform_labels(df: pd.Series) -> ndarray:
    """
    This function maps the label values to 0 or 1 and returns the pd.Series object.
    In this func I assume that the df will be a Series object because there is usually
    only one label column.
    Args:
        df (pd.Series): Label series object.

    Returns:
        mapped Series object.
    """
    return df.map({"No": 0, "Yes": 1, }).to_numpy()
