import os
import mlflow
from mlflow.models.signature import infer_signature

def _is_prod_ready():
    return True

def deploy_model_step(*args, run_id: str) -> None:
    """"""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    train_x, train_y, model = args
    with mlflow.start_run(run_id=run_id, nested=True) as _:
        if _is_prod_ready():
            signature = infer_signature(train_x, train_y)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="employee_attrition",
                signature=signature,
                registered_model_name="catboost_attrition_model",
            )