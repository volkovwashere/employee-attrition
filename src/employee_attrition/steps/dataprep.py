from employee_attrition.steps.utils.extract import load_dataset_mock, extract_feature_columns, extract_label_column
from employee_attrition.steps.utils.transform import transform_labels, split_dataset


def preprocess_data_step(dataset_path: str = None) -> tuple:
    # load data
    df = load_dataset_mock()
    df_features = extract_feature_columns(df)
    df_labels = transform_labels(extract_label_column(df))

    # split data
    x = split_dataset(df_features, 0.3)
    y = split_dataset(df_labels, 0.3)
    return x, y
