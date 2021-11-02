# PROJECT DESCRIPTION
This project performed as part of the ML Zoomcamp course, Midterm Project. This course is conducted by [Alexey Grigorev](https://bit.ly/3BxeAoB)


## Health Insurance Cross-Sell Prediction
Cross-Sell Prediction help _Health Insurance Companies_ predicting which customers may interest in Vehicle Insurance contract extension.

### Dataset Reference:
This Model was built using [kaggle Dataset](https://bit.ly/3bEwA5D).

### Task Detail
Auto insurance is a contract between customer and the insurance company that protects policyholder against financial loss in the event of an accident or theft. In exchange for paying a premium, the insurance company agrees to pay customer's losses as outlined in the policy.

In this project, the client is a _Health Insurance company_ which provides health insurance for its customers. Now, they needs a model to predict if their last-year customers want to have their vehicle contract with this company or not.

Prediction model helps the company to expand their business and plan their communication strategies to have more policyholders. So, they collect some information about the customers and their previous vehicle insurance contract and the payments to build the prediction model. The following is the informations.

### Data Description
Following is the features used for the prediction model. The _Response_ shows the target value.


| Features | Definitions |
|---|---|
|Id|Unique ID for the customer|
|Gender|Gender of the customer|
|Age|Age of the customer|
|Driving_License|0 : Customer does not have DL, 1 : Customer already has DL|
|Region_Code|Unique code for the region of the customer|
|Previously_Insured|1 : Customer already has Vehicle Insurance, 0 : Customer doesn't have Vehicle Insurance|
|Vehicle_Age|Age of the Vehicle|
|Vehicle_Damage|1 : Customer got his/her vehicle damaged in the past. 0 : Customer didn't get his/her vehicle damaged in the past.|
|Annual_Premium|The amount customer needs to pay as premium in the year|
|PolicySalesChannel|Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.|
|Vintage|Number of Days, Customer has been associated with the company|
|Response|1 : Customer is interested, 0 : Customer is not interested|

#### Features' Characteristics
Features' characteristics are provided in [DataAnalysis_Insurance](https://bit.ly/3ExkHei)

**1. Features Type**
|Categorical|Numerical|
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


# RUNNING INSTRUCTION
## File Description

Folder [MidtermProject](https://github.com/LeilaRanjbar82/ML-zoomcamp-course-homework/tree/main/MidtermProject) includes following files:

|File Name|Description|
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




