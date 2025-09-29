import numpy as np
import pandas as pd
import os

from trainMODBUS import *
from trainTCP import *

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
    TCP_df = TCP_df['Ack'].fillna(0)

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

def modelbuild(df, features, modeltype):
    if modeltype == 'isolation forest':
        X = df[features]
        model = IsolationForest(n_estimators=100, contamination=0.1).fit(X)
        predictions = model.predict(X)
        normal_anomalies = np.sum(predictions == -1)

        print(f"Features used: {features}")
        # print(f"Flagged {predictions}/{len(X)} samples as potential anomalies in clean data") #this doesnt work right now 

        return model
    
    elif modeltype == 'lstm':
        data = df[features].values
        sequence = 30
        X, y = [], []
        for i in range(len(data) - sequence):
            X.append(data[i:(i + sequence)])
            y.append(data[i + sequence])
        
        X, y = np.array(X), np.array(y)

        print(f"X contains NaN: {np.isnan(X).any()}\nY contains NaN: {np.isnan(y).any()}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # input shape is: sequence length, number of features (30, 5)
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True,  input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=25, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=25, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(X_train.shape[2], activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
        model.summary
        # [:, -1, :] = all sequences, last timestep, all features 
        model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=1)
        predictions = model.predict(X_train)
        train_mse = np.mean(np.square((predictions) - X_train[:, -1, :]), axis=1)
        
        print(f"Features used: {features}")
        print(f"MSE for reconstruction: {np.percentile(train_mse, 95):.6f}")

        return model   
    
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
    isof_model = modelbuild(scaled_df, isof_features, 'isolation forest')
    lstm_model = modelbuild(scaled_df, lstm_features, 'lstm')
    modelsave(lstm_model, 'lstm')
    modelsave(isof_model, 'isolation_forest')

def tcp(TCP_df):
    uniquesrc, uniquedest, uniquelist = iplist(TCP_df)
    time_differences, resp_df = timeprocessTCP(TCP_df)
    resp_df['Flag_encode'] = LabelEncoder().fit_transform(resp_df['Flags'])
    resp_df['SrcPort_encode'] = LabelEncoder().fit_transform(resp_df['SrcPort'])
    resp_df['DstPort_encode'] = LabelEncoder().fit_transform(resp_df['DstPort'])
    scaler, scaled_df = processing(resp_df)
    isof_features = modelfeaturesTCP(scaled_df, 'isolation forest')
    lstm_features = modelfeaturesTCP(scaled_df, 'lstm')
    isof_model = modelbuild(scaled_df, isof_features, 'isolation forest')
    lstm_model = modelbuild(scaled_df, lstm_features, 'lstm')
    modelsave(lstm_model, 'lstm')
    modelsave(isof_model, 'isolation_forest')

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


