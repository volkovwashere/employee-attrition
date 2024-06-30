import requests
import pandas as pd
from employee_attrition.steps.utils.extract import extract_feature_columns
from employee_attrition.steps.utils.transform import split_dataset
from sklearn.preprocessing import StandardScaler
import json

# Note this is not really a test, more like a demo suite


df = pd.read_csv("../src/employee_attrition/data/fake_db/employee-attrition.csv")
df_features = extract_feature_columns(df)
train_x, _, test_x = split_dataset(df_features, 0.3)

inference_inputs = test_x.head(100)  # lets assume that we got this from somewhere else


def inference_preprocessor(training_data: pd.DataFrame, inference_data: pd.DataFrame) -> pd.DataFrame:
    standardized_data = StandardScaler().fit_transform(pd.concat([training_data, inference_data]))
    standardized_data = standardized_data[training_data.shape[0] :]
    standardized_df = pd.DataFrame(data=standardized_data, columns=training_data.columns)
    assert standardized_df.shape == inference_data.shape, "shape mismatch between std df and inference df"
    return standardized_df


inference_inputs = inference_preprocessor(train_x, inference_inputs)

headers = {
    "Content-Type": "application/json",
}
r = requests.post(
    url="http://127.0.0.1:5002/invocations",
    data=json.dumps({"dataframe_records": inference_inputs.to_dict("records")}),
    headers=headers,
)

print(list(map(lambda x: {0: "no", 1: "yes"}.get(x), r.json()["predictions"])))
