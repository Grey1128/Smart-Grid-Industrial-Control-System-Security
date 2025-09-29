print('Importing libraries....')
import numpy as np
import pandas as pd
import datetime as dt
import os

import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, InputLayer
from keras.optimizers import Adam
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.impute import KNNImputer, SimpleImputer

print('Libraries imported.....')

def timeprocessTCP(df):
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df = df.sort_values('Time').reset_index(drop=True)

    #initialise
    time_differences = []
    processed_indeces = set()
    df['reponse_time'] = np.nan

    for i in range(len(df)):
        if i in processed_indeces:
            continue

        current_row = df.iloc[i]
        current_flag = current_row['Flags']
        current_seq = current_row['Seq']
        current_ack = current_row['Ack']
        current_source = current_row['Source']

        search_range = range(max(0, i - 2), min(len(df), i + 2))

        for j in search_range:
            if j == i or j in processed_indeces:
                continue

            other_row = df.iloc[j]

            if (other_row['Flags'] != current_flag and (other_row['Seq'] == current_ack or int(other_row['Ack']) == int(current_seq) + 1) and other_row['Source'] != current_source):
                if current_flag != 'ACK':
                    query_indx, response_indx = i, j
                    query_time = current_row['Time']
                    response_time = other_row['Time']
                else:
                    query_indx, response_indx = j, i
                    query_time = other_row['Time']
                    response_time = current_row['Time']

                time_diff = response_time - query_time
                time_differences.append(time_diff)

                df.loc[query_indx, 'response_time'] = time_diff
                df.loc[response_indx, 'response_time'] = time_diff

                processed_indeces.add(i)
                processed_indeces.add(j)
                break

    return time_differences, df

def modelfeaturesTCP(scaled_df, modeltype):
    if modeltype == 'lstm':
        features = [
            'Time',
            'response_time',
            'time_dif',
            'Flag_encode',
            'Seq'
            'Ack'
            'Win'
            'Len'
            'MSS'
        ]
        
    elif modeltype == 'isolation forest':
        features = [
            'response_time',
            'time_dif',
            'Seq'
            'Ack'
            'Win'
            'Len'
            'MSS'
            'source_encoded',
            'dest_encoded',
            'SrcPort_encode'
            'DstPort_encode'
            'Time'
        ]
    return features





# def main():
#     uniquesrc, uniquedest, uniquelist = iplist(df)
#     time_differences, resp_df = timeprocess(df)
#     mean_time_diff = np.mean([t for t in time_differences if t > 0])
#     print(f"Mean Response Time: {mean_time_diff:.6f} seconds")
#     print(f"Total pairs found: {len(time_differences)}")
#     print(f"Allowed sources are: {uniquelist}")
#     scaler, scaled_df = processing(resp_df)
#     isof_features = modelfeatures(scaled_df, 'isolation forest') 
#     lstm_features = modelfeatures(scaled_df, 'lstm')
#     isof_model = modelbuild(scaled_df, isof_features, 'isolation forest')
#     lstm_model = modelbuild(scaled_df, lstm_features, 'lstm')
#     modelsave(lstm_model, 'lstm')
#     modelsave(isof_model, 'isolation_forest')
    

# main()