from employee_attrition.steps.utils.transform import transform_numerical_features
from sklearn.metrics import classification_report


def test_step(*args, run_id: str) -> None:
    """
    Runs the eval stage of the current ml pipeline.
    Args:
        run_id (str): Defaults to None. Current mlflow experiment run id.

    Returns:
        None
    """
    import mlflow

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run(run_id=run_id, nested=True) as run:
        print(f"Running evaluation stage with id: {run.info.run_id}")  # should be logger.info later ...
        test_x, test_y, catboost_model = args

        # apply standarscaler after splitting the data
        test_x = transform_numerical_features(test_x)[0]

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
        results_per_label = tuple(clf_report[label] for label in target_names)
        updated_results_per_label = []
        for result, label in zip(results_per_label, target_names):
            updated_results_per_label.append({f"{stage}/class_{label}/{k}": round(v, 4) for k, v in result.items()})
        formatted_report = updated_results_per_label[0] | updated_results_per_label[1]
        mlflow.log_metrics(formatted_report)
