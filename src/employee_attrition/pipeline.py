import mlflow
from employee_attrition.steps.test import test_step
from employee_attrition.steps.train import train_step
from employee_attrition.steps.deploy import deploy_model_step
from employee_attrition.steps.dataprep import preprocess_data_step


def run_pipeline() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run() as run:
        print("Starting data prep step ...")
        x, y = preprocess_data_step()
        train_x, val_x, test_x = x
        tran_y, val_y, test_y = y

        print("Starting train step ...")
        model = train_step(
            train_x,
            tran_y,
            val_x,
            val_y,
            run_id=run.info.run_id,
        )

        print("Starting test step ...")
        test_step(
            test_x,
            test_y,
            model,
            run_id=run.info.run_id,
        )

        print("Starting deploy step ...")
        deploy_model_step(
            train_x,
            tran_y,
            model,
            run_id=run.info.run_id,
        )


run_pipeline()
