import numpy as np
import pandas as pd
import os
import pickle


from trainMODBUS import *
from trainTCP import *
from keras.models import Model
from keras.layers import Dense, LSTM, Input, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

MB_Feat_Cols = ['Time', 'Source', 'Destination', 'Protocol', 'Length', 'Direction', 'TransID', 'UnitID', 'FuncCode']
TCP_Feat_Cols = ['Time', 'Source', 'Destination', 'Protocol', 'Length', 'SrcPort', 'DstPort', 'Flags', 'Seq', 'Ack', 'Win', 'Len', 'MSS']

def loaddata():
    os.chdir('Raw-Data')
    df = pd.read_csv('MB_n_TCP.csv')
    print(df.columns)
    df = df.sort_values('Time').reset_index(drop=True)

    #Drops unnecesary columns and all ARP rows 
    df = df.drop('No.', axis=1)
    df = df.drop('Other Protocol Details', axis=1)
    droparp_df = df[df['Protocol'] != 'ARP']

    #Seperate dfs into MODBUS/TCP and TCP traffic
    MB_df = droparp_df[droparp_df['Protocol'] == 'Modbus/TCP']
    TCP_df = droparp_df[droparp_df['Protocol'] == 'TCP']
    TCP_df['Ack'] = TCP_df['Ack'].fillna(0)

    return MB_df, TCP_df, droparp_df

def processing(df):
    df['source_encoded'] = LabelEncoder().fit_transform(df['Source'])
    df['dest_encoded'] = LabelEncoder().fit_transform(df['Destination'])
    df['response_time'].fillna(df['response_time'].median(), inplace=True)

    # Difference between each message, beginning at 0 for the NAN value in first index
    df['time_dif'] = df['Time'].diff().fillna(0)

    scaling_features = ['Time', 'response_time', 'time_dif']

    scaler = MinMaxScaler()
    for feat in scaling_features:
        df[feat] = scaler.fit_transform(df[[feat]])

    return scaler, df

def modelbuild(df, features, modeltype='lstm_autoencoder', sequence=30):
    data = df[features].values
    X = []
    for i in range(len(data) - sequence):
        X.append(data[i:(i + sequence)])
    X = np.array(X)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    inputs = Input(shape=(timesteps, n_features))
    encoded = LSTM(64, activation="relu")(inputs)
    repeated = RepeatVector(timesteps)(encoded)
    decoded = LSTM(64, activation="relu", return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(n_features))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.summary()
    model.fit(X_train, X_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1)

    #Reconstruction error on training data
    X_train_pred = model.predict(X_train)
    train_mse = np.mean(np.square(X_train_pred - X_train), axis=(1,2))

    threshold = np.percentile(train_mse, 95)
    print(f"Reconstruction error threshold set to {threshold:.6f}")

    return model, threshold
    
def modelsave(model, string):
    date = dt.datetime.now()
    datestring = date.strftime("%Y-%m-%d %H-%M-%S")
    filename = f"{string}_Modbus_{datestring}.pk1"

    if not os.path.isdir("models"):
        os.mkdir("models")
        print("models directory created\n")
    os.chdir("models")
    pickle.dump(model, open(filename, 'wb'))
    os.chdir("..")

def modbus(MB_df):
    time_differences, resp_df = timeprocess(MB_df)
    mean_time_diff = np.mean([t for t in time_differences if t > 0])
    print(f"Mean Response Time: {mean_time_diff:.6f} seconds")
    print(f"Total pairs found: {len(time_differences)}")
    resp_df['direction_binary'] = (resp_df['Direction'] == 'Query').astype(int)
    scaler, scaled_df = processing(resp_df)
    isof_features = modelfeaturesMB(scaled_df, 'isolation forest') 
    lstm_features = modelfeaturesMB(scaled_df, 'lstm')
    lstm_model = modelbuild(scaled_df, MB_Feat_Cols, 'lstm')
    modelsave(lstm_model, 'lstm')

def tcp(TCP_df):
    uniquesrc, uniquedest, uniquelist = iplist(TCP_df)
    time_differences, resp_df = timeprocessTCP(TCP_df)
    resp_df['Flag_encode'] = LabelEncoder().fit_transform(resp_df['Flags'])
    resp_df['SrcPort_encode'] = LabelEncoder().fit_transform(resp_df['SrcPort'])
    resp_df['DstPort_encode'] = LabelEncoder().fit_transform(resp_df['DstPort'])
    scaler, scaled_df = processing(resp_df)
    isof_features = modelfeaturesTCP(scaled_df, 'isolation forest')
    lstm_features = modelfeaturesTCP(scaled_df, 'lstm')
    lstm_model = modelbuild(scaled_df, TCP_Feat_Cols, 'lstm')
    modelsave(lstm_model, 'lstm')

def iplist(df):
    uniquesrc = df['Source'].unique()
    uniquedest = df['Destination'].unique()
    uniqueappend = np.append(uniquesrc, uniquedest)
    uniquelist = np.unique(uniqueappend)
    print(f"Allowed sources are: {uniquelist}")
    return uniquesrc, uniquedest, uniquelist

def main():
    MB_df, TCP_df, droparp_df = loaddata()
    uniquesrc, uniquedest, uniquelist = iplist(droparp_df)
    modbus(MB_df)
    tcp(TCP_df)

main()
