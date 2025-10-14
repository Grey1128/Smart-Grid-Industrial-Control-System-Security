import matplotlib.pyplot as plt
import pickle
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import date
from trainMODBUS import *
from trainTCP import *
from threshold import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from keras.models import load_model


def loadmodel():
    os.chdir('models')

    tcp_file = glob.glob(f'LSTM_Autoencoder_TCP_*')
    mb_file = glob.glob(f'LSTM_Autoencoder_MODBUS_*')
    mb_scaler_file = glob.glob(f'Scaler_MODBUS_*')
    tcp_scaler_file = glob.glob(f'Scaler_TCP_*')
    encoders_file = glob.glob(f'Encoders_TCP_*')
    
    tcp_file = max(tcp_file)
    mb_file = max(mb_file)
    mb_scaler_file = max(mb_scaler_file)
    tcp_scaler_file = max(tcp_scaler_file)
    encoders_file = max(encoders_file) 

    mb_model = pickle.load(open(mb_file, 'rb'))
    tcp_model = pickle.load(open(tcp_file, 'rb'))
    mb_scaler = pickle.load(open(mb_scaler_file, 'rb'))
    tcp_scaler = pickle.load(open(tcp_scaler_file, 'rb'))
    encoders = pickle.load(open(encoders_file, 'rb'))

    os.chdir("..")

    return mb_model, tcp_model, mb_scaler, tcp_scaler, encoders

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
    MB_df = MB_df.iloc[2:]
    TCP_df = droparp_df[droparp_df['Protocol'] == 'TCP']
    TCP_df = TCP_df.drop(mb_cols, axis=1)
    TCP_df['MSS'].fillna(0, inplace=True)
    TCP_df['Ack'] = TCP_df['Ack'].fillna(0)
    TCP_df = TCP_df.dropna()

    return MB_df, TCP_df, droparp_df

def processing(df, scaler):
    # df['source_encoded'] = LabelEncoder().fit_transform(df['Source'])
    # df['dest_encoded'] = LabelEncoder().fit_transform(df['Destination'])
    df['protocol_encoded'] = LabelEncoder().fit_transform(df['Protocol'])
    #This adds additional data to the model to say whether or not there was a response. 
    df['has_response_time'] = df['response_time'].notna().astype(int)
    df['response_time'].fillna(0, inplace=True)

    # Difference between each message, beginning at 0 for the NAN value in first index
    df['time_dif'] = df['Time'].diff().fillna(0)

    scaling_features = df.select_dtypes(include=[np.number]).columns.tolist()

    df[scaling_features] = scaler.transform(df[scaling_features])

    return df

def sequencing(X_data):
    seq_length = 35
    X = []
    for i in range(len(X_data) - seq_length + 1):
        X.append(X_data[i:i + seq_length])
    X = np.array(X)
    return X

#modbus() and tcp() can later be added to trainMODBUS and trainTCP to reuse them across trainLSTM and testAE
def modbus(MB_df, mb_model, scaler):
    time_differences, resp_df = timeprocess(MB_df)
    mean_time_diff = np.mean([t for t in time_differences if t > 0])
    print(f"Mean Response Time: {mean_time_diff:.6f} seconds")
    print(f"Total pairs found: {len(time_differences)}")
    
    resp_df['direction_binary'] = (resp_df['Direction'] == 'Query').astype(int)
    scaled_df = processing(resp_df, scaler)
  
    MB_Feat_Cols = ['protocol_encoded', 'Length', 'direction_binary', 'TransID', 'UnitID', 'FuncCode', 'response_time', 'time_dif', 'has_response_time']
    X_data = scaled_df[MB_Feat_Cols].values
    X_sd = sequencing(X_data)
    anomaly_scores, anomalies = detect(X_sd, mb_model, 'modbus')
    
    scaled_df_sequenced = scaled_df.iloc[34:].reset_index(drop=True)
    scaled_df_sequenced['anomaly_score'] = anomaly_scores
    scaled_df_sequenced['is_anomaly'] = anomalies
    print(f'MODBUS anomalies detected: {np.sum(anomalies)} out of {len(scaled_df_sequenced)} entries')
    print(f'Threshold is: {MODBUS_THRESHOLD}%')
    anomaly_indexes = np.where(anomalies)[0] #0 used to only use the first value from the row/column tuple 
    if len(anomaly_indexes) > 0:
        print(f"Anomaly row indexes: {anomaly_indexes.tolist()}")
    return scaled_df_sequenced

def tcp(TCP_df, tcp_model, scaler, encoders):
    time_differences, resp_df = timeprocessTCP(TCP_df)
    mean_time_diff = np.mean([t for t in time_differences if t > 0])
    print(f"Mean Response Time: {mean_time_diff:.6f} seconds")
    print(f"TOtal pairs found: {len(time_differences)}")

    resp_df['Flag_encode'] = encoders['flag'].fit_transform(resp_df['Flags'])
    resp_df['SrcPort_encode'] = encoders['srcport'].fit_transform(resp_df['SrcPort'])
    resp_df['DstPort_encode'] = encoders['dstport'].fit_transform(resp_df['DstPort'])
    scaled_df = processing(resp_df, scaler)

    TCP_Feat_Cols = ['protocol_encoded', 'Length', 'SrcPort_encode', 'DstPort_encode', 'Flag_encode', 'Seq', 'Ack', 'Win', 'Len', 'MSS', 'response_time', 'time_dif', 'has_response_time']
    X_data = scaled_df[TCP_Feat_Cols].values
    X_sd = sequencing(X_data)
    anomaly_scores, anomalies = detect(X_sd, tcp_model, 'tcp')

    #34 as sequencing removes 34 indexes from the data (64,333 versus 64,299)
    scaled_df_sequenced = scaled_df.iloc[34:].reset_index(drop=True)
    scaled_df_sequenced['anomaly_score'] = anomaly_scores
    scaled_df_sequenced['is_anomaly'] = anomalies
    print(f'TCP anomalies detected: {np.sum(anomalies)} out of {len(scaled_df_sequenced)} entries')
    print(f'Threshold is: {TCP_THRESHOLD}%')
    anomaly_indexes = np.where(anomalies)[0]
    if len(anomaly_indexes) > 0:
        print(f"Anomaly row indexes: {anomaly_indexes.tolist()}")

    return scaled_df_sequenced

def detect(scaled_data, model, string):
    reconstruct = model.predict(scaled_data)
    anom_scores = np.array([mean_squared_error(scaled_data[i], reconstruct[i]) for i in range(len(scaled_data))])
    if string == 'modbus':
        anomalies = anom_scores > MODBUS_THRESHOLD
    elif string == 'tcp':
        anomalies = anom_scores > TCP_THRESHOLD
    return anom_scores, anomalies

def plotanom(mb_anom, tcp_anom):
    mb_values = mb_anom['anomaly_score'].values
    tcp_values = tcp_anom['anomaly_score'].values
    mb_x = mb_anom['Time'].values
    tcp_x = tcp_anom['Time'].values

    plt.subplot(2,1,1, figsize=(12,6))
    plt.plot(mb_x, mb_values, color='blue')
    plt.title('MODBUS Anomaly Scores')
    plt.ylabel('Anomaly Scores')
    plt.xlabel('Time')

    plt.subplot(2,1,2, figsize=(12,6))
    plt.plot(tcp_x, tcp_values, color='green')
    plt.title('TCP/MODBUS Anomaly Scores')
    plt.ylabel('Anomaly Scores')
    plt.xlabel('Time')

    plt.show()

def createlogs(mb_anom, tcp_anom):
    os.chdir('anomalylogs')
    today = date.today().strftime('%d%m%Y')
    mb_anom.to_csv(f'mb_log_{today}.csv', index=False)
    tcp_anom.to_csv(f'tcp_log_{today}.csv', index=False)
    os.chdir('..')

def main():
    MB_df, TCP_df, droparp_df = loaddata()
    mb_model, tcp_model, mb_scaler, tcp_scaler, encoders = loadmodel()
    mb_anom = modbus(MB_df, mb_model, mb_scaler)
    tcp_anom = tcp(TCP_df, tcp_model, tcp_scaler, encoders)
    createlogs(mb_anom, tcp_anom)
    plotanom(mb_anom, tcp_anom)
    

main()

 