from catboost import CatBoostClassifier
from employee_attrition.steps.utils.transform import transform_numerical_features
from src.employee_attrition.steps.utils.metrics import create_classification_report


def train_step(*args, run_id: str) -> CatBoostClassifier:
    """
    This function contains the main logic for the train stage in the ml pipeline. It
    conducts training for a catboost classifier model, and then registers the model at
    the end of the job based on some logic / strategy pre-defined.
    Args:
        run_id (str): Current run id if exists.

    Returns:
        None
    """
    import mlflow

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run(run_id=run_id, nested=True) as run:
        print(f"Starting training run on mlflow with run id: {run.info.run_id}")
        train_x, train_y, val_x, val_y = args

        # apply standarscaler after splitting the data
        train_x, val_x = transform_numerical_features(train_x, val_x)

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
        train_results = create_classification_report(
            target=train_y,
            preds=model.predict(train_x),
            labels=["no", "yes"],
            stage="train",
        )
        val_results = create_classification_report(
            target=val_y,
            preds=model.predict(val_x),
            labels=["no", "yes"],
            stage="validation",
        )
        # log to mlflow
        mlflow.log_params(model_params)
        mlflow.log_metrics(train_results)
        mlflow.log_metrics(val_results)
        return model
    