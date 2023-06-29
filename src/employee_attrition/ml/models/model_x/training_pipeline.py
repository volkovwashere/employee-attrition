from employee_attrition.ml.models.model_x.stages.train import train_stage
from employee_attrition.ml.models.model_x.stages.evaluate import eval_stage
from employee_attrition.ml.models.model_x.constants import MLFLOW_TRACKING_URI
import mlflow


def main() -> None:
    """
    This func bundles the ml pipeline stages together and executes the stage jobs sequentially.
    Returns:
        None
    """
    # Note the tracking uri could be some constant parameter that is shared everywhere it's needed.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run() as run:
        print("Started running employee attrition training pipeline ...")

        print("Starting training stage ...")
        train_stage(run_id=run.info.run_id)

        print("Starting eval stage ...")
        eval_stage(run_id=run.info.run_id) 
    # Note that logger.info can be used here later.
    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
