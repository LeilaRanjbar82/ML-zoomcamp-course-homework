#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

# parameters


output_file = 'model.bin'


# data preparation

df = pd.read_csv('insurance_prediction.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

df['vehicle_age'] = df['vehicle_age'].str.replace('> 2 Years', 'more than 2 Years')
df['vehicle_age'] = df['vehicle_age'].str.replace('< 1 Year', 'less than 1 Year')
df['previously_insured'] = df['previously_insured'].map({0:'no', 1:'yes'})
df['driving_license'] = df['driving_license'].map({0:'no', 1:'yes'})

categorical_list = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_list:
    df[c] = df[c].str.lower().str.replace(' ', '_')
    
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=25)
y_full_train = df_full_train.response.values
y_test = df_test.response.values

categorical = ['gender',
               'driving_license',
               'previously_insured',
               'vehicle_age',
               'vehicle_damage']

numerical = ['age',
             'region_code',
             'annual_premium',
             'policy_sales_channel',
             'vintage']



# training


def train(df_train, y_train):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    dtrain = xgb.DMatrix(X_train, label=y_train,
                             feature_names=dv.get_feature_names_out())

    xgb_params = {
        'eta': 0.1,
        'max_depth': 6,
        'min_child_weight': 30,

        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params, dtrain, num_boost_round=100)
    
    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    dtest = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())
    y_pred = model.predict(dtest)

    return y_pred



#training the final model
print('training the final model')

dv, model = train (df_full_train, y_full_train)
y_pred = predict (df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))

print(f'auc={auc}')
print(f'rmse={rmse}')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved in {output_file}')


