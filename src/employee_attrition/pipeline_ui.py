import os
import streamlit as st
from employee_attrition.pipeline import run_pipeline
from mlflow.tracking import MlflowClient
import requests

st.title("Pipeline admin UI")

execute_pipeline = st.button("Run pipeline")

if execute_pipeline:
    try:
        run_pipeline()
        st.info("Successful run!")
    except Exception as e:
        print(f"{e}")
        st.error(f"Run was unsuccessful due to: {e}")


@st.cache_data(show_spinner=False, ttl=5)
def _get_available_models():
    client = MlflowClient(os.getenv("MLFLOW_TRACKING_URI"))
    try:
        registered_models = client.search_registered_models()
        return registered_models[0].latest_versions
    except IndexError:
        return []


if not _get_available_models():
    st.info("Could not find model registered, run the pipeline first ...")
else:
    model_block = st.container()
    model_block.title("Available models and versions")
    model_block.write(_get_available_models())
    model_block.write(f"Find the experiments and models here: http://127.0.0.1:5000 or use localhost:5000")
    ping = model_block.button("ping", help="Press to get health status of the inference endpoint.")
    if ping:
        try:
            res = requests.get(f"{os.getenv('MLFOW_INFERENCE_ENDPIPOINT_URI')}/ping")
            model_block.write("Inference endpoint is currently functioning at: http://127.0.0.1:5002/invocations")
        except Exception:
            model_block.write("Inference endpoint is currently not functioning at: http://127.0.0.1:5002/invocations")



