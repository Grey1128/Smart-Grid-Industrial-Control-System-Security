import matplotlib.pyplot as plt
import pickle
import os
import glob
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from keras.models import load_model

def loadmodel():
    os.chdir('models')

    isof_file = glob.glob(f'isolation_*')
    isof_file = max(isof_file)
    lstm_file = glob.glob(f'lstm_*')
    lstm_file = max(lstm_file)

    isof_model = pickle.load(open(isof_file, 'rb'))
    lstm_model = pickle.load(open(lstm_file, 'rb'))

    os.chdir("..")

    return isof_model, lstm_model

def loaddata():
    FEATURE_COLUMNS = ["Time","Source","Destination","direction","trans_id","unit_id","func_code","rest"]

    os.chdir('Raw-Data')
    df = pd.read_csv('normal_modbus.csv', skiprows=[1])
    os.chdir('..')
    df.columns = FEATURE_COLUMNS
    
    return df

def modeldatasets(df, model_name):

    if model_name == 'lstm':
        features = [
            'Time',
            'response_time',
            'time_dif',
            'direction_binary',
            'func_code'
        ]

        data = df[features].values
        sequence = 30
        X, y = [], []
        for i in range(len(data) - sequence):
            X.append(data[i:(i + sequence)])
            y.append(data[i + sequence])
        
        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test 


    else: 
        features = [
            'response_time',
            'time_dif',
            'func_code',
            'unit_id',
            'direction_binary',
            'source_encoded',
            'dest_encoded',
            'Time'
        ]

        X = df[features]

        return X

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
    df['response_time'].fillna(df['response_time'].median(), inplace=True)

    #Difference between each message, beginning at 0 for the NAN value in first index
    df['time_dif'] = df['Time'].diff().fillna(0)

    scaling_features = ['Time', 'response_time', 'time_dif']

    scaler = MinMaxScaler()
    for feat in scaling_features:
        df[feat] = scaler.fit_transform(df[[feat]])

    return scaler, df

def main():
    isof_model, lstm_model = loadmodel()
    predf = loaddata()
    time_differences, resp_df = timeprocess(predf)
    scaler, scaled_df = processing(resp_df)
    isof_X = modeldatasets(scaled_df, 'isolation forest')
    lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test = modeldatasets(scaled_df, 'lstm')
    

main()

 