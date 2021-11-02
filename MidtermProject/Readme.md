# PROJECT DESCRIPTION
This project performed as part of the _ML Zoomcamp Course_, Midterm Project. This course is conducted by [Alexey Grigorev](https://bit.ly/3BxeAoB)


## Health Insurance Cross-Sell Prediction
Cross-Sell Prediction help _Health Insurance Companies_ predicting which customers may interest in Vehicle Insurance contract extension.

### Dataset Reference:
This Model was built using [kaggle Dataset](https://bit.ly/3bEwA5D).

### Task Detail
Auto insurance is a contract between customer and the insurance company that protects policyholder against financial loss in the event of an accident or theft. In exchange for paying a premium, the insurance company agrees to pay customer's losses as outlined in the policy.

In this project, the client is a _Health Insurance company_ which provides health insurance. They need a model to predict if their last-year customers want to have their vehicle contract with this company or not.

Prediction model helps the company to expand their business and plan their communication strategies to have more policyholders. So, they collect some information about the customers and their previous vehicle insurance and premium to build the prediction model.

### Data Description
Following is the features used for the prediction model. The _Response_ shows the target value.


| **Features** | **Definitions** |
|---|---|
|Id|Unique ID for the customer|
|Gender|Gender of the customer|
|Age|Age of the customer|
|Driving_License|1 : Customer already has DL, 0 : Customer does not have DL|
|Region_Code|Unique code for the region of the customer|
|Previously_Insured|1 : Customer already has Vehicle Insurance, 0 : Customer doesn't have Vehicle Insurance|
|Vehicle_Age|Age of the Vehicle|
|Vehicle_Damage|1 : Customer got his/her vehicle damaged in the past. 0 : Customer didn't get his/her vehicle damaged in the past.|
|Annual_Premium|The amount customer needs to pay as premium in the year|
|PolicySalesChannel|Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.|
|Vintage|Number of Days, Customer has been associated with the company|
|Response|1 : Customer is interested, 0 : Customer is not interested|

### Features Characteristics
Features characteristics are provided in [DataAnalysis_Insurance](https://bit.ly/3ExkHei)

**1. Features Type**
|**Categorical**|**Numerical**|
|---|---|
|gender|age|
|previously_insured|region_code|
|driving_license|annual_premium|
|vehicle_damage|policy_sales_channel|
|vehicle_age|vintage|

**2.Correlated Features**

_AGE_ had the highest correaltion with _RESPONSE_

**3.High-Risk Features**

_VEHICLE_AGE: More than 2 years_, _VEHICLE_DAMAGE:yes_, _PREVIOUSLY_INSURED:no_ had the highest risks, respectively.

**4.Mutual Informaton**

_PREVIOUSLY_INSURED_ had the highest mutual information.


### Evaluation Metrics
_AUC_ROC_Curve_ and _RMSE_ were used as evaluation Metrics.

### Prediction Model
By evaluating different models, _XGBoost_ achieved the best result.
|**Model**|**RMSE**|**AUC**|
|---|---|---|
|Logistic_Regression|0.3273|0.5944|
|Ridge_Regression|0.3056|0.8186|
|Decision Tree|0.2998|0.8376|
|Random_Forest|0.2988|0.8421|
|***XGBoost***|***0.2982***|***0.8445***|



# FILE DESCRIPTION

Folder [MidtermProject](https://github.com/LeilaRanjbar82/ML-zoomcamp-course-homework/tree/main/MidtermProject) includes following files:

|**File Name**|**Description**|
|---|---|
|insurance_prediction.csv|Dataset|
|DataAnalysis_insurance.ipynb|Exploratory Data Analysis & Feature important Analysis|
|notebook.ipynb|Data preparation and cleaning & Model selection|
|train.py|Training the final model|
|model.bin|Saved model by pickle|
|predict.py|Loading the model &Serving it via a web serice (with Flask)|
|predict-test.py|Testing the model|
|Pipfile & Pipfile.lock|Python virtual environment, Pipenv file|
|Dockerfile|Environment management, Docker, for running file|

# RUNNING INSTRUCTION
1. Copy scripts (train, predict and predict-test), pipenv file and Dockerfile to a folder
2. Run Windows Terminal Linux (WSL2) in the that folder
3. Install `pipenv`
   ```
   pip install pipenv
   ```
4. Install essential packages
   ```
   pipenv install numpy pandas scikit-learn==1.0 flask xgboost
   ```
5. Install Docker
 - SingUp for a DockerID in [Docker](https://hub.docker.com/)
 - Download & Intall [Docker Desktop](https://docs.docker.com/desktop/windows/install/)
 
6. In WSL2 run the following command to create the image `zoomcamp-midproj'
   ```
   docker build -t zoomcamp-midproj .
   ```
7. Run Docker to loading model
   ```
   docker run -it --rm -p 8889:8889 zoomcamp-midproj
   ```
   (Maybe need to change the port is scripts and Dockerfile due to the system)

8. In another WSL tab run the test 
   ```
   python predict-test.py
   ```
9. The Result would be
   ```
   Customer is NOT interested in Vehicle Insurance provided by the company
   ```
  
