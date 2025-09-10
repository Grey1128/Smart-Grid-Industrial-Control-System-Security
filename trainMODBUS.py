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

FEATURE_COLUMNS = ["Time","Source","Destination","direction","trans_id","unit_id","func_code","rest"]

os.chdir('Raw-Data')
df = pd.read_csv('normal_modbus.csv', skiprows=[1])
os.chdir('..')
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
    df['response_time'].fillna(df['response_time'].median(), inplace=True)

    #Difference between each message, beginning at 0 for the NAN value in first index
    df['time_dif'] = df['Time'].diff().fillna(0)

    scaling_features = ['Time', 'response_time', 'time_dif']

    scaler = MinMaxScaler()
    for feat in scaling_features:
        df[feat] = scaler.fit_transform(df[[feat]])

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
        X = df[features]
        model.fit(X)
        predictions = model.predict(X)
        normal_anomalies = np.sum(predictions == -1)

        print(f"Features used: {features}")
        print(f"Flagged {predictions}/{len(X)} samples as potential anomalies in clean data")

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

        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        
        #input shape is: sequence length, number of features (30, 5)
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
        #[:, -1, :] = all sequences, last timestep, all features 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test), verbose=1)
        predictions = model.predict(X_train)
        train_mse = np.mean(np.square((predictions) - X_train[:, -1, :]), axis=1)
        
        print(f"Features used: {features}")
        print(f"MSE for reconstruction: {np.percentile(train_mse, 95):.6f}")

        return model   
    
def modelsave(model, string):
    date = dt.datetime.now()
    datestring = date.strftime("%Y-%m-%d %H:%M:%S")
    filename = f"{string}_{datestring}.pk1"

    if not os.path.isdir("models"):
        os.mkdir("models")
        print("models directory created\n")
    os.chdir("models")
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    uniquesrc, uniquedest, uniquelist = iplist(df)
    time_differences, resp_df = timeprocess(df)
    mean_time_diff = np.mean([t for t in time_differences if t > 0])
    print(f"Mean Response Time: {mean_time_diff:.6f} seconds")
    print(f"Total pairs found: {len(time_differences)}")
    print(f"Allowed sources are: {uniquelist}")
    scaler, scaled_df = processing(resp_df)
    isof_features = modelfeatures(scaled_df, 'isolation forest') 
    lstm_features = modelfeatures(scaled_df, 'lstm')
    isof_model = modelbuild(None, isof_features, 'isolation forest')
    lstm_model = modelbuild(scaled_df, lstm_features, 'lstm')
    modelsave(lstm_model, 'lstm')
    modelsave(isof_model, 'isolation_forest')
    

main()