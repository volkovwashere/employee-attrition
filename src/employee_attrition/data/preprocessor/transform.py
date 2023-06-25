import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
from numpy import ndarray

def split_dataset(df: pd.DataFrame, test_size: float, random_seed: float=42) -> Tuple[pd.DataFrame,]:
    train, test = train_test_split(df, test_size=test_size, random_state=random_seed)
    test, validation = train_test_split(test, test_size=0.5, random_state=random_seed)
    return train, validation, test

def transform_numerical_features(*dataframes: Tuple[pd.DataFrame]) -> Tuple[ndarray]: 
    return tuple(
        StandardScaler().fit_transform(df)
        for df in dataframes
    )

def transform_labels(df: pd.Series) -> ndarray:
    return df.map({"No": 0, "Yes": 1,}).to_numpy()