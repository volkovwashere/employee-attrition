from catboost import CatBoostClassifier
from employee_attrition.data.preprocessor.extract import extract_feature_columns, extract_label_column, load_dataset_mock
from employee_attrition.data.preprocessor.transform import transform_labels, transform_numerical_features, split_dataset
from sklearn.metrics import classification_report

def _save_evaluation_results(results: dict) -> None:
    import json
    with open("model_artifacts/results.json", "w") as f:
        json.dump(results, f)

    
def eval_stage(run_id: str) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    assert run_id, "Run id must be provided."
    mlflow.set_tracking_uri(uri:="http://127.0.0.1:5000")
    client = MlflowClient(uri)
    latest_model = client.search_model_versions("name='catboost_attrition_model'")[0]

    with mlflow.start_run(run_id=run_id, nested=True) as run:
        print(f"Running evaluation stage with id: {run.info.run_id}")
        # load data
        df = load_dataset_mock()
        df_features = extract_feature_columns(df)
        df_labels = transform_labels(extract_label_column(df))

        # split data
        _, _, test_x = split_dataset(df_features, 0.3)
        _, _, test_y = split_dataset(df_labels, 0.3)

        # apply standarscaler after splitting the data
        test_x = transform_numerical_features(test_x)[0]

        # load trained model
        catboost_model = mlflow.sklearn.load_model(
            model_uri=f"models:/catboost_attrition_model/{latest_model.version}"
        )
        
        # predict the test dataset
        predictions = catboost_model.predict(test_x)
        target_names = ["no", "yes"]
        stage = "test"
        clf_report = classification_report(
            test_y,
            predictions,
            target_names=target_names,
            output_dict=True,
        )
        
        # format report NOTE this could be a general util function ...
        results_per_label = tuple(
            clf_report[label] for label in target_names
        )
        updated_results_per_label = []
        for result, label in zip(results_per_label, target_names):
            updated_results_per_label.append({
                f"{stage}/class_{label}/{k}": round(v, 4) for k, v in result.items()
            })
        formated_report = updated_results_per_label[0] | updated_results_per_label[1]
        mlflow.log_metrics(formated_report)
        mlflow.set_tags(latest_model.__dict__)

if __name__ == "__main__":
    eval_stage()
