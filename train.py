print('Importing libraries....')
import numpy as np
import pandas as pd
import datetime as dt
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, InputLayer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.impute import KNNImputer, SimpleImputer

print('Libraries imported.....')

FEATURE_COLUMNS = ["Time","Source","Destination","direction","trans_id","unit_id","func_code","rest"]

os.chdir('Raw-Data')
df = pd.read_csv('normal_modbus.csv', skiprows=[1])
df.columns = FEATURE_COLUMNS

def timeprocess(df):
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df = df.sort_values('Time').reset_index(drop=True)

    # Initialise values
    time_differences = []
    processed_indices = set()
    df['response_time'] = np.nan
    
    # Makes sure we're not going over the same rows with adjacency checks
    for i in range(len(df)):
        if i in processed_indices:
            continue

        current_row = df.iloc[i]
        current_trans_id = current_row['trans_id']
        current_direction = current_row['direction']

        # Checks 3 spots lower and 3 spots higher which is from what I can see, the maximum length apart a response can be 
        search_range = range(max(0, i-4), min(len(df), i + 4))

        # Same check as before,, insuring we're not going over the same rows/not comparing itself with itself
        for j in search_range:
            if j == i or j in processed_indices:
                continue

            other_row = df.iloc[j]
            
            # Found matching transaction with opposite direction
            if (other_row['trans_id'] == current_trans_id and 
                other_row['direction'] != current_direction):
                
                # Determine query and response times
                if current_direction == 'Query':
                    query_indx, response_indx = i, j
                    query_time = current_row['Time']
                    response_time = other_row['Time']
                else:
                    query_indx, response_indx = j, i
                    query_time = other_row['Time']
                    response_time = current_row['Time']
                
                # Calculate time difference (response - query)
                time_diff = response_time - query_time
                time_differences.append(time_diff)
                
                # Adds the respective response time to each index of the response_time column
                df.loc[query_indx, 'response_time'] = time_diff
                df.loc[response_indx, 'response_time'] = time_diff

                # Mark both rows as processed
                processed_indices.add(i)
                processed_indices.add(j)
                break
    
    return time_differences, df

def processing(df):
    df['source_encoded'] = LabelEncoder().fit_transform(df['Source'])
    df['dest_encoded'] = LabelEncoder().fit_transform(df['Destination'])
    df['direction_binary'] = (df['direction'] == 'Query').astype(int)

    #Difference between each message, beginning at 0 for the NAN value in first index
    df['time_dif'] = df['Time'].dif().fillna(0)

    scaling_features = ['Time', 'response_time', 'time_dif']

    scaler = MinMaxScaler()
    for feat in scaling_features:
        df[feat] = scaler.fit_transform(df[feat].fillna(0))

    X, y = [], []

    X_train, X_test, y_train, y_test = train_test_split()

    return scaler, df

def iplist(df):
    uniquesrc = df['Source'].unique()
    uniquedest = df['Destination'].unique()
    uniqueappend = np.append(uniquesrc, uniquedest)
    uniquelist = np.unique(uniqueappend)
    return uniquesrc, uniquedest, uniquelist

def modelfeatures(scaled_df, modeltype):
    if modeltype == 'lstm':
        features = [
            'Time',
            'response_time',
            'time_dif',
            'direction_binary',
            'func_code'
        ]
        
    elif modeltype == 'isolation forest':
        features = {
            'response_time',
            'time_dif',
            'func_code',
            'unit_id',
            'direction_binary',
            'source_encoded',
            'dest_encoded',
            'Time'
        }
    return features

def modelbuild(df, features, modeltype):
    if modeltype == 'isolation forest':
        model = IsolationForest(n_estimators=100, features=1.0, contamination=0.1).fit(features)
    
    elif modeltype == 'lstm':
        model = Sequential()
        
        

def main():
    uniquesrc, uniquedest, uniquelist = iplist(df)
    time_differences, resp_df = timeprocess(df)
    mean_time_diff = np.mean([t for t in time_differences if t > 0])
    scaler, scaled_df = processing(resp_df)
    lstm_features = modelfeatures(scaled_df, 'lstm')
    isof_features = modelfeatures(scaled_df, 'isolation forest') 
    lstm_model = modelbuild(df, lstm_features, 'lstm')
    isof_model = modelbuild(None, isof_features, 'isolation forest')
    print(f"Mean Response Time: {mean_time_diff:.6f} seconds")
    print(f"Total pairs found: {len(time_differences)}")
    print(f"Allowed sources are: {uniquelist}")

main()