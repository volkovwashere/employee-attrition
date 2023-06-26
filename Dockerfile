FROM python:3.10-alpine

WORKDIR /employee-attrition/

COPY src /employee-attrition/src/
COPY mlartifacts /employee-attrition/mlartifacts
COPY requirements.txt /employee-attrition/requirements.txt
RUN pip install -r /employee-attrition/requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/employee-attrition/src"
ENV MLFLOW_TRACKING_URI "http://localhost:5000"

CMD mlfow models serve -m "models:/catboost_attrition_model/latest" --port 5002 --no-conda
