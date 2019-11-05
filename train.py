# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 23:50:40 2019

@author: bhaskar
"""

#  Importing libraries
import numpy as np
import pandas as pd
import json
import os
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

# Loading the dataset
logging.basicConfig(level=logging.INFO)
try:
    df = pd.read_csv('ml_eng_ay_data.csv.gz',parse_dates=['date']) # Importing the dataset
except OSError as e:
    logging.error("File not found")
    
# Data Preprocessing    
df = df.query('rent_total<4000 and rent_total>0')
std_dev = df['rent_total'].std()  # Calculate standard deviation
mean = df['rent_total'].mean()   # Calculate mean of the variable
cutoff_val = mean + (3*std_dev)    # Finding the cutoff value
df = df[df['rent_total'] < cutoff_val] # Considering only values till 3 standard deviations

# Replacing string values with numbers
df.loc[df['has_elevator'] == 't', 'has_elevator'] = 1
df.loc[df['has_elevator'] == 'f', 'has_elevator'] = 0
df.loc[df['has_garden'] == 't', 'has_garden'] = 1
df.loc[df['has_garden'] == 'f', 'has_garden'] = 0
df.loc[df['has_balcony'] == 't', 'has_balcony'] = 1
df.loc[df['has_balcony'] == 'f', 'has_balcony'] = 0
df.loc[df['has_kitchen'] == 't', 'has_kitchen'] = 1
df.loc[df['has_kitchen'] == 'f', 'has_kitchen'] = 0
df.loc[df['has_guesttoilet'] == 't', 'has_guesttoilet'] = 1
df.loc[df['has_guesttoilet'] == 'f', 'has_guesttoilet'] = 0

# Imputing the missing values
df['flat_thermal_characteristic'].fillna(df['flat_thermal_characteristic'].median(), inplace=True) # Filling null values with median

# Dropping some unnecessary values
df.drop('geo_city',axis=1,inplace=True)
df.loc[df['flat_type'] == 'appartment', 'flat_type'] = 'apartment'

# Creating new time related features
df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month
df['day'] = pd.DatetimeIndex(df['date']).day
df['weekday'] = pd.DatetimeIndex(df['date']).weekday

df['is_weekend'] = df.apply(lambda row:1 if row['weekday'] >4 else 0,axis = 1)  # Checking whether the day is weekend or not.

#defining the function to bin the data
def day_split(row):
    if row['day'] < 10:
        return '1-10'
    elif row['day'] >= 10 and row['day'] < 20:
        return '10-20'
    else:
        return '21-31'

#binning the day features
df['day_r'] = df.apply(day_split, axis = 1)

#one hot encoding
one_hot_columns = ['flat_type','flat_interior_quality','flat_condition','flat_age','geo_city_part','day_r']   # Dropping the features which are note required for development of model and the target label
df_final = pd.get_dummies(df,columns=one_hot_columns,drop_first=True)

#splitting the features and label
X = df_final.drop(["date","rent_total","rent_base","month","year","day","weekday"],axis=1)
y = df_final["rent_total"]

#splitting the data into training and testing data
X_train,y_train = X[0:12000],y[0:12000]
X_test,y_test = X[12000:],y[12000:]

#standardizing the features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting the Random forest model . I have used the parameters from grid search best estimator.
model_rf = RandomForestRegressor(n_estimators = 500,max_depth = 25,min_samples_leaf = 1,min_samples_split=5)

#fitting the data
model_rf.fit(X_train,y_train)

test_mse = np.sqrt(mean_squared_error(y_test, model_rf.predict(X_test)))
metadata = {
    "rmse value": test_mse
}

##############################################################################
# Serialize model and metadata
print("Serializing model to: {}".format(MODEL_PATH))
dump(model_rf, MODEL_PATH)

try:
    print("Serializing metadata to: {}".format(METADATA_PATH))
    with open(METADATA_PATH, 'w') as outfile:  
        json.dump(metadata, outfile)
except OSError as e:
    print('METADATA_PATH not found')
    















