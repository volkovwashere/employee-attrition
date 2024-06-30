FROM python:3.10-slim

WORKDIR /employee-attrition/

COPY src /employee-attrition/src/
COPY requirements.txt /employee-attrition/requirements.txt

RUN pip install -r requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/employee-attrition/src"

CMD streamlit run src/employee_attrition/pipeline_ui.py --server.address 0.0.0.0 --server.port 8501