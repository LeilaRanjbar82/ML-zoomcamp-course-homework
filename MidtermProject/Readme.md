# Health Insurance Cross-Sell Prediction
Cross-Sell Prediction help _Health Insurance Companies_ predicting which customers may interest in Vehicle Insurance contract extension.

## Dataset Refrence:
This Model was built using [kaggle Dataset](https://bit.ly/3bEwA5D).

## Task Detail
Auto insurance is a contract between customer and the insurance company that protects policyholder against financial loss in the event of an accident or theft. In exchange for paying a premium, the insurance company agrees to pay customer's losses as outlined in the policy.

In this project, the client is a _Health Insurance company_ which provides health insurance for its customers. Now, they needs a model to predict if their last-year customers want to have their vehicle contract with this company or not.

Prediction model helps the company to expand their business and plan their communication strategies to have more policyholders. So, they collect some information about the customers and their previous vehicle insurance contract and the payments to build the prediction model. The following is the informations.

## Data Description

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
