from employee_attrition.ml.models.model_x.stages.train import train_stage
from employee_attrition.ml.models.model_x.stages.evaluate import eval_stage
import mlflow

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run() as run:
        print("Started running employee attrition training pipeline ...")

        print("Starting training stage ...")
        train_stage(run_id=run.info.run_id)

        print("Starting eval stage ...")
        eval_stage(run_id=run.info.run_id) 
    
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()