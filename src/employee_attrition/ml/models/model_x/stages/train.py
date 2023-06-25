from catboost import CatBoostClassifier
from employee_attrition.data.preprocessor.extract import extract_feature_columns, extract_label_column, load_dataset_mock
from employee_attrition.data.preprocessor.transform import transform_labels, transform_numerical_features, split_dataset
from sklearn.metrics import classification_report
from numpy import ndarray
from typing import List, Optional

def _create_classification_report(
        target:ndarray,
        preds:ndarray,
        labels: List[str],
        stage: str,
    ) -> dict:
    results = classification_report(
        target,
        preds,
        target_names=labels,
        output_dict=True,
    )
    # format clf report
    results_per_label = tuple(
        results[label] for label in labels
    )
    updated_results_per_label = []
    for result, label in zip(results_per_label, labels):
        updated_results_per_label.append({
            f"{stage}/class_{label}/{k}": round(v, 4) for k, v in result.items()
        })

    formated_results = updated_results_per_label[0] | updated_results_per_label[1] | {f"{stage}/accuracy": results["accuracy"]}
    return formated_results
    


def train_stage(run_id: Optional[str]) -> None:
    import mlflow
    from mlflow.models.signature import infer_signature

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run(run_id=run_id, nested=True) as run:
        print(f"Starting training run on mlflow with run id: {run.info.run_id}")
        # load data
        df = load_dataset_mock()
        df_features = extract_feature_columns(df)
        df_labels = transform_labels(extract_label_column(df))

        # split data
        train_x, val_x, _ = split_dataset(df_features, 0.3)
        train_y, val_y, _ = split_dataset(df_labels, 0.3)

        # apply standarscaler after splitting the data
        train_x, val_x = transform_numerical_features(train_x, val_x)
        signature = infer_signature(train_x, train_y)

        # configure params (for now only hardcoding it instead of creating a conf file)
        model_params = {
            "iterations": 100,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 5,
        }

        # start training
        model = CatBoostClassifier(**model_params)
        model.fit(
            train_x,
            train_y,
            eval_set=(val_x, val_y),
            plot_file="attrition_train_results",
        )

        # make clf
        train_results = _create_classification_report(
            target=train_y,
            preds=model.predict(train_x),
            labels=["no", "yes"],
            stage="train",
        )
        val_results = _create_classification_report(
            target=val_y,
            preds=model.predict(val_x),
            labels=["no", "yes"],
            stage="validation",
        )
        # log to mlflow
        mlflow.log_params(model_params)
        mlflow.log_metrics(train_results)
        mlflow.log_metrics(val_results)
        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     artifact_path="employee_attrition",
        #     signature=signature,
        #     registered_model_name="catboost_attrition_model",
        # )

if __name__ == "__main__":
    train_stage()