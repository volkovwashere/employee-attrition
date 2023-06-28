FROM python:3.10-slim

WORKDIR /employee-attrition/

COPY src /employee-attrition/src/
COPY mlartifacts /employee-attrition/mlartifacts
COPY mlruns /employee-attrition/mlruns
COPY requirements.txt /employee-attrition/requirements.txt

RUN pip install -r /employee-attrition/requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/employee-attrition/src"

CMD mlflow models serve -m "models:/catboost_attrition_model/latest" -h 0.0.0.0 -p 5002 --no-conda
