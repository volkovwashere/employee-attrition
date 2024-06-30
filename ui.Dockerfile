FROM python:3.10-slim

WORKDIR /employee-attrition/

COPY src /employee-attrition/src/
# COPY mlartifacts /employee-attrition/mlartifacts

RUN pip install mlflow
ENV PYTHONPATH "${PYTHONPATH}:/employee-attrition/src"

CMD mlflow server --backend-store-uri sqlite:///my.db --port 5000 --host 0.0.0.0

