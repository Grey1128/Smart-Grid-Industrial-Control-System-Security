print('Importing libraries....')
import numpy as np
import pandas as pd

print('Libraries imported.....')

FEATURE_COLUMNS = ['Time', 'Source', 'Destination', 'Protocol', 'Length', 'Direction', 'TransID', 'UnitID', 'FuncCode']

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
        current_trans_id = current_row['TransID']
        current_direction = current_row['Direction']

        # Checks 3 spots lower and 3 spots higher which is from what I can see, the maximum length apart a response can be 
        search_range = range(max(0, i-4), min(len(df), i + 4))

        # Same check as before,, insuring we're not going over the same rows/not comparing itself with itself
        for j in search_range:
            if j == i or j in processed_indices:
                continue

            other_row = df.iloc[j]
            
            # Found matching transaction with opposite direction
            if (other_row['TransID'] == current_trans_id and 
                other_row['Direction'] != current_direction):
                
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

def modelfeaturesMB(scaled_df, modeltype):
    if modeltype == 'lstm':
        features = [
            'Time',
            'response_time',
            'time_dif',
            'direction_binary',
            'FuncCode'
        ]
        
    elif modeltype == 'isolation forest':
        features = [
            'response_time',
            'time_dif',
            'FuncCode',
            'UnitID',
            'direction_binary',
            'source_encoded',
            'dest_encoded',
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