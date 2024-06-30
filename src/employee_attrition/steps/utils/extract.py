import pandas as pd
from pathlib import Path


def load_dataset_mock(db_path: str = "data/fake_db/employee-attrition.csv") -> pd.DataFrame:
    """
    This function is a mocking func for loading a dataset from "SQL", which in this case
    is from a locally stored csv file.
    Args:
        db_path (str): DB path, in this case locally stored csv. Defaults to the relative path
            compared to this func.

    Returns:
        read pd.DataFrame object
    """
    db_path = Path(__file__).parent.parent.parent / db_path
    return pd.read_csv(db_path)


def extract_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function extracts the feature columns from a given dataframe. Here I
    assumed that for training only the numerical feature columns will be used.
    This is due to making the data pipeline easier and faster, because there is
    no need for transforming the categorical columns later.
    Args:
        df (pd.DataFrame): DataFrame object.

    Returns:
        pd.DataFrame object only with numerical columns.
    """
    return df.select_dtypes(include="int")


def extract_label_column(df: pd.DataFrame, label_column_name: str = "Attrition") -> pd.DataFrame:
    """
    This function extracts the label column from a given dataframe and a given column_name, and
    returns a pd.Series or DataFrame object.
    Args:
        df (pd.Dataframe): DataFrame object.
        label_column_name (str): Column name where the labels are stored.

    Returns:
        pd.DataFrame or Series
    """
    return df[label_column_name]
