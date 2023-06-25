import pandas as pd
from pathlib import Path

def load_dataset_mock(db_path: str="fake_db/employee-attrition.csv") -> pd.DataFrame:
    """This functions is used as a mock database loader, returns the csv as
    a pandas dataframe.

    Parameters:
        db_path(str): CSV file path in the fake db. (Default)
    
    Returns:
        pd.DataFrame
    """
    # for now we just assume that the fake db will always be on the same level
    db_path = Path(__file__).parent.parent / db_path
    return pd.read_csv(db_path)

def extract_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="int")

def extract_label_column(df: pd.DataFrame, label_column_name: str = "Attrition") -> pd.DataFrame:
    return df[label_column_name]
