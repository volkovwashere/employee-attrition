# Employee-attrition case study
<!-- ABOUT THE PROJECT -->
## About The Project
Attrition is a problem that impacts all businesses and it leads to significant costs for a business, including the cost of business disruption, hiring new staff and training new staff. Therefore, businesses, in particular their HR departments have great interest in understanding the drivers of, and minimizing staff attrition. The use of classification models to predict if an employee is likely to quit could greatly increase HR's ability to intervene on time and remedy the situation to prevent attrition. This projects aims to create a model inference pipeline deployed on docker that can be accessed and distributed easliy on to different systems.

## Built with
The application was built with:
- Python
- Catboost
- MlFlow
- Docker

<!-- Getting started -->
## Getting started
### Prequisites
For the prequisites you need python 3.10 installed on your machine with anaconda preferred as environment management.
<br />https://www.anaconda.com/products/individual

### Installation guide
#### From scratch only using the api (harder)
~~~
git clone https://github.com/volkovwashere/employee-attrition.git
~~~
~~~
export PYTHONPATH=$PYTHONPATH:/home/<your_username>/employee-attrition/src
~~~
~~~
cd employee-attrition
~~~
~~~
pip install -r requirements.txt
~~~
~~~
python <path_to_script>/training_pipeline.py
~~~
Note that you also need to run the mlflow server which will be exposed on http://localhost:5000 by default:
~~~
mlflow ui
~~~
If you want to serve the models locally also run:
~~~
export MLFLOW_TRACKING_URI=http://localhost:5000
~~~
~~~
mlflow models serve -m "models:/catboost_attrition_model/latest" -p 5002 --no-conda
~~~

__RECOMMENDED use docker__
Simply run
~~~
docker-compose up
~~~
And you should be able to use mlflow ui at localhost:5000 and do inference at localhost:5002/invocations. An example rest api request is set up under src/employee_attrition/test_endpoint.py . Further documentations of the input/output structure the nference pipeline expects can bee found at the mfllow ui
