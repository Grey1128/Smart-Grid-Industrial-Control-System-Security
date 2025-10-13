import numpy as np
import pandas as pd
import os
import pickle

from datetime import date
from trainMODBUS import *
from trainTCP import *
from keras.models import Model
from keras.layers import Dense, LSTM, Input, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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
    # df['source_encoded'] = LabelEncoder().fit_transform(df['Source'])
    # df['dest_encoded'] = LabelEncoder().fit_transform(df['Destination'])
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

def modelbuild(df, features, sequence):
    data = df[features].values
    X = []
    for i in range(len(data) - sequence):
        X.append(data[i:(i + sequence)])
    X = np.array(X)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    inputs = Input(shape=(timesteps, n_features))
    encoded = LSTM(128, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(64, activation='relu')(encoded)
    repeated = RepeatVector(timesteps)(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(repeated)
    decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(n_features))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.summary()
    #wrapping early_stop in [] turns it into a list which callbacks requires - note to remember 
    model.fit(X_train, X_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1, callbacks=[early_stop])

    #Reconstruction error on training data
    X_train_pred = model.predict(X_train)
    train_mse = np.mean(np.square(X_train_pred - X_train), axis=(1,2))

    #threshold is made to be 99% to reduce chance of false positives
    threshold = np.percentile(train_mse, 99)
    model.threshold = threshold
    print(f"Reconstruction error threshold set to {threshold:.6f}")

    return model, threshold
    
def modelsave(model, string, scaler, encoders):
    date = dt.datetime.now()
    datestring = date.strftime("%Y-%m-%d %H-%M-%S")
    filename = f"LSTM_Autoencoder_{string}_{datestring}.pk1"

    if not os.path.isdir("models"):
        os.mkdir("models")
        print("models directory created\n")
    os.chdir("models")
    pickle.dump(model, open(filename, 'wb'))
    
    scaler_filename = f"Scaler_{string}_{datestring}"
    pickle.dump(scaler, open(scaler_filename, 'wb'))

    if encoders != 'None':
        encoders_filename = f"Encoders_{string}_{datestring}"
        pickle.dump(encoders, open(encoders_filename, 'wb'))

    os.chdir("..")

    print(f"{string} model, scaler and encoders saved")

def modbus(MB_df):
    time_differences, resp_df = timeprocess(MB_df)
    mean_time_diff = np.mean([t for t in time_differences if t > 0])
    print(f"Mean Response Time: {mean_time_diff:.6f} seconds")
    print(f"Total pairs found: {len(time_differences)}")
    resp_df['direction_binary'] = (resp_df['Direction'] == 'Query').astype(int)
    scaler, scaled_df = processing(resp_df)
    createlogs(scaled_df, 'mb')
    #MB_Feat_Cols = ['Time', 'source_encoded', 'dest_encoded', 'protocol_encoded', 'Length', 'direction_binary', 'TransID', 'UnitID', 'FuncCode', 'response_time', 'time_dif', 'has_response_time']
    MB_Feat_Cols = ['protocol_encoded', 'Length', 'direction_binary', 'TransID', 'UnitID', 'FuncCode', 'response_time', 'time_dif', 'has_response_time']
    lstm_model, threshold = modelbuild(scaled_df, MB_Feat_Cols, 35)
    modelsave(lstm_model, 'MODBUS', scaler, 'None')
    return threshold

def tcp(TCP_df):
    uniquesrc, uniquedest, uniquelist = iplist(TCP_df)
    time_differences, resp_df = timeprocessTCP(TCP_df)
    mean_time_diff = np.mean([t for t in time_differences if t > 0])
    print(f"Mean Response Time: {mean_time_diff:.6f} seconds")
    print(f"TOtal pairs found: {len(time_differences)}")

    flagencoder = LabelEncoder()
    srcportencoder = LabelEncoder()
    dstportencoder = LabelEncoder()

    resp_df['Flag_encode'] = flagencoder.fit_transform(resp_df['Flags'])
    resp_df['SrcPort_encode'] = srcportencoder.fit_transform(resp_df['SrcPort'])
    resp_df['DstPort_encode'] = dstportencoder.fit_transform(resp_df['DstPort'])

    encoders = {
        'flag': flagencoder,
        'srcport': srcportencoder,
        'dstport' : dstportencoder
    }

    scaler, scaled_df = processing(resp_df)
    createlogs(scaled_df, 'tcp')
    #TCP_Feat_Cols = ['Time', 'source_encoded', 'dest_encoded', 'protocol_encoded', 'Length', 'SrcPort_encode', 'DstPort_encode', 'Flag_encode', 'Seq', 'Ack', 'Win', 'Len', 'MSS', 'response_time', 'time_dif', 'has_response_time']
    TCP_Feat_Cols = ['protocol_encoded', 'Length', 'SrcPort_encode', 'DstPort_encode', 'Flag_encode', 'Seq', 'Ack', 'Win', 'Len', 'MSS', 'response_time', 'time_dif', 'has_response_time']
    lstm_model, threshold = modelbuild(scaled_df, TCP_Feat_Cols, 35)
    modelsave(lstm_model, 'TCP', scaler, encoders)
    return threshold

def createlogs(df, string):
    os.chdir('proccessed_data')
    today = date.today().strftime('%d%m%Y')
    df.to_csv(f'{string}_scaled_{today}.csv', index=False)
    os.chdir('..')

def iplist(df):
    uniquesrc = df['Source'].unique()
    uniquedest = df['Destination'].unique()
    uniqueappend = np.append(uniquesrc, uniquedest)
    uniquelist = np.unique(uniqueappend)
    print(f"Allowed sources are: {uniquelist}")
    return uniquesrc, uniquedest, uniquelist

def threshsave(tcp_threshold, mb_threshold):
    if os.path.exists('threshold.py'):
        os.remove('threshold.py')
    with open('threshold.py', 'w') as f:
        f.write(f'TCP_THRESHOLD = {tcp_threshold}\nMODBUS_THRESHOLD = {mb_threshold}')
    print('Threshold saved')

def main():
    MB_df, TCP_df, droparp_df = loaddata()
    uniquesrc, uniquedest, uniquelist = iplist(droparp_df)
    mb_threshold = modbus(MB_df)
    tcp_threshold = tcp(TCP_df)
    threshsave(tcp_threshold, mb_threshold)

main()
