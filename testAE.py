import matplotlib.pyplot as plt
import pickle
import os
import glob
import pandas as pd
import numpy as np

from trainMODBUS import *
from trainTCP import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from keras.models import load_model

def loadmodel():
    os.chdir('models')

    tcp_file = glob.glob(f'LSTM_Autoencoder_TCP_*')
    mb_file = glob.glob(f'LSTM_Autoencoder_TCP_*')
    tcp_file = max(tcp_file)
    mb_file = max(mb_file)

    mb_model = pickle.load(open(mb_file, 'rb'))
    tcp_model = pickle.load(open(tcp_file, 'rb'))

    os.chdir("..")

    return mb_model, tcp_model

def loaddata():
    os.chdir('Raw-Data')
    df = pd.read_csv('MB_n_TCP.csv')
    print(df.columns)
    os.chdir('..')
    df = df.sort_values('Time').reset_index(drop=True)

    #Drops unnecesary columns and all ARP rows 
    df = df.drop('No.', axis=1)
    df = df.drop('Other Protocol Details', axis=1)
    droparp_df = df[df['Protocol'] != 'ARP']

    #Seperate dfs into MODBUS/TCP and TCP traffic
    MB_df = droparp_df[droparp_df['Protocol'] == 'Modbus/TCP']
    tcp_cols = ['SrcPort', 'DstPort', 'Flags', 'Seq', 'Ack', 'Win', 'Len', 'MSS']
    mb_cols = ['Direction', 'TransID', 'UnitID', 'FuncCode', 'FuncDesc']
    MB_df = MB_df.drop(tcp_cols, axis=1)
    TCP_df = droparp_df[droparp_df['Protocol'] == 'TCP']
    TCP_df = TCP_df.drop(mb_cols, axis=1)
    TCP_df['MSS'].fillna(0, inplace=True)
    TCP_df['Ack'] = TCP_df['Ack'].fillna(0)
    TCP_df = TCP_df.dropna()

    return MB_df, TCP_df, droparp_df

def processing(df):
    df['source_encoded'] = LabelEncoder().fit_transform(df['Source'])
    df['dest_encoded'] = LabelEncoder().fit_transform(df['Destination'])
    df['protocol_encoded'] = LabelEncoder().fit_transform(df['Protocol'])
    #This adds additional data to the model to say whether or not there was a response. 
    df['has_response_time'] = df['response_time'].notna().astype(int)
    df['response_time'].fillna(0, inplace=True)

    # Difference between each message, beginning at 0 for the NAN value in first index
    df['time_dif'] = df['Time'].diff().fillna(0)

    scaling_features = df.select_dtypes(include=[np.number]).columns.tolist()

    scaler = MinMaxScaler()
    df[scaling_features] = scaler.fit_transform(df[scaling_features])

    return scaler, df

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

#modbus() and tcp() can later be added to trainMODBUS and trainTCP to reuse them across trainLSTM and testAE
def modbus(MB_df, mb_model):
    time_differences, resp_df = timeprocess(MB_df)
    mean_time_diff = np.mean([t for t in time_differences if t > 0])
    print(f"Mean Response Time: {mean_time_diff:.6f} seconds")
    print(f"Total pairs found: {len(time_differences)}")
    resp_df['direction_binary'] = (resp_df['Direction'] == 'Query').astype(int)
    scaler, scaled_df = processing(resp_df)
    MB_Feat_Cols = ['Time', 'source_encoded', 'dest_encoded', 'protocol_encoded', 'Length', 'direction_binary', 'TransID', 'UnitID', 'FuncCode', 'response_time', 'time_dif', 'has_response_time']
    X_sd = scaled_df[MB_Feat_Cols].values
    anomaly_scores, anomalies = detect(X_sd, mb_model, mb_model.threshold)
    resp_df['anomaly_score'], resp_df['is_anomaly'] = anomaly_scores, anomalies
    print(f'TCP anomalies detected: {np.sum(anomalies)} out of {len(resp_df)} entries')
    print(f'Threshold is: {mb_model.threshold:.2f}%')
    anomaly_indexes = np.where(anomalies)[0] #0 used to only use the first value from the row/column tuple 
    if len(anomaly_indexes) > 0:
        print(f"Anomaly row indexes: {anomaly_indexes.tolist()}")

def tcp(TCP_df, tcp_model):
    time_differences, resp_df = timeprocessTCP(TCP_df)
    mean_time_diff = np.mean([t for t in time_differences if t > 0])
    print(f"Mean Response Time: {mean_time_diff:.6f} seconds")
    print(f"TOtal pairs found: {len(time_differences)}")
    resp_df['Flag_encode'] = LabelEncoder().fit_transform(resp_df['Flags'])
    resp_df['SrcPort_encode'] = LabelEncoder().fit_transform(resp_df['SrcPort'])
    resp_df['DstPort_encode'] = LabelEncoder().fit_transform(resp_df['DstPort'])
    scaler, scaled_df = processing(resp_df)
    TCP_Feat_Cols = ['Time', 'source_encoded', 'dest_encoded', 'protocol_encoded', 'Length', 'SrcPort_encode', 'DstPort_encode', 'Flag_encode', 'Seq', 'Ack', 'Win', 'Len', 'MSS', 'response_time', 'time_dif', 'has_response_time']
    X_sd = scaled_df[TCP_Feat_Cols].values
    anomaly_scores, anomalies = detect(X_sd, tcp_model, tcp_model.threshold)
    resp_df['anomaly_score'], resp_df['is_anomaly'] = anomaly_scores, anomalies
    print(f'TCP anomalies detected: {np.sum(anomalies)} out of {len(resp_df)} entries')
    print(f'Threshold is: {tcp_model.threshold:.2f}%')
    anomaly_indexes = np.where(anomalies)[0]
    if len(anomaly_indexes) > 0:
        print(f"Anomaly row indexes: {anomaly_indexes.tolist()}")

def detect(scaled_data, model, threshold):
    reconstruct = model.predict(scaled_data)
    anom_scores = np.array([mean_squared_error(scaled_data[i], reconstruct[i]) for i in range(len(scaled_data))])
    anomalies = anom_scores > threshold
    return anom_scores, anomalies, threshold

def main():
    MB_df, TCP_df, droparp_df = loaddata()
    mb_model, tcp_model = loadmodel()
    modbus(MB_df, mb_model)
    tcp(TCP_df, tcp_model)
    

main()

 