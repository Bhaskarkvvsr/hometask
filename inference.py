# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 23:50:40 2019

@author: bhaskar
"""

import os
from joblib import load
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler


MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

def get_data():
    """
    Return data for inference.
    """
    print("Loading data...")
    # Loading the dataset
    logging.basicConfig(level=logging.INFO)
    try:
        df = pd.read_csv('ml_eng_ay_data.csv.gz',parse_dates=['date']) # Importing the dataset
    except OSError as e:
        logging.error("File not found")
    df = pd.read_csv('ml_eng_ay_data.csv.gz',parse_dates=['date'])
    df = df.query('rent_total<4000 and rent_total>0')
    
    std_dev = df['rent_total'].std()  # Calculate standard deviation
    mean = df['rent_total'].mean()   # Calculate mean of the variable
    cutoff_val = mean + (3*std_dev)    # Finding the cutoff value
    df = df[df['rent_total'] < cutoff_val] # Considering only values till 3 standard deviations
    
    # Applying the same transformations which we applied on the training set
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
    
    df['flat_thermal_characteristic'].fillna(df['flat_thermal_characteristic'].median(), inplace=True) # Filling null values with median
    
    df.drop('geo_city',axis=1,inplace=True) # Dropping geo_city feature
    df.loc[df['flat_type'] == 'appartment', 'flat_type'] = 'apartment'
    
    df['year'] = pd.DatetimeIndex(df['date']).year  # Extracting time features
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day
    df['weekday'] = pd.DatetimeIndex(df['date']).weekday
    
    df['is_weekend'] = df.apply(lambda row:1 if row['weekday'] >4 else 0,axis = 1)  # Checking whether the day is weekend or not.
    
    def day_split(row):
        # Defining a function that can bin the days features
        if row['day'] < 10:
            return '1-10'
        elif row['day'] >= 10 and row['day'] < 20:
            return '10-20'
        else:
            return '21-31'

    df['day_r'] = df.apply(day_split, axis = 1)

    one_hot_columns = ['flat_type','flat_interior_quality','flat_condition','flat_age','geo_city_part','day_r']   # Dropping the features which are note required for development of model and the target label
    df_final = pd.get_dummies(df,columns=one_hot_columns,drop_first=True) # One hot encoding the data.
    
    test_data = df_final[12000:] # Defining last 3000 rows as test set.
    X = test_data.drop(["date","rent_total","rent_base","month","year","day","weekday"],axis=1)
    y = test_data["rent_total"]
    
    sc = StandardScaler() #Scaling the features
    X = sc.fit_transform(X)
    return X,y

print("Running inference...")

X, y = get_data()


# Load model
print("Loading model from: {}".format(MODEL_PATH))
model_rf = load(MODEL_PATH)

# Run inference
print("Scoring observations...")
y_pred = model_rf.predict(X)
print(y_pred)
