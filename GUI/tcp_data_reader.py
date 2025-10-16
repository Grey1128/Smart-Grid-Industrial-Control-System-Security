import pandas as pd
import random
import itertools

class TCPDataReader:
    
    def __init__(self, filepath):
        
        print(f"Attempting to load data from: {filepath}")
        try:
            self.df = pd.read_csv(filepath)
            print("Successfully loaded CSV file.")
            
            if 'is_anomaly' not in self.df.columns:
                print("Warning: 'is_anomaly' column not found. Simulating it for now.")
                #make anomalies a bit rarer to be more realistic.
                self.df['is_anomaly'] = [random.choices([0, 1], weights=[0.85, 0.15], k=1)[0] for _ in range(len(self.df))]
                print("Added simulated 'is_anomaly' column.")

            # Add a simulated confidence score for realism 
            self.df['confidence_score'] = [random.randint(60, 95) if anomaly == 1 else 0 for anomaly in self.df['is_anomaly']]
            
            # Create an iterator that will loop indefinitely 
            self.data_iterator = itertools.cycle(self.df.iterrows())
            print("Data reader initialized and ready.")

        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found. Please ensure it's in the same directory.")
            self.df = pd.DataFrame() # Create an empty dataframe to avoid errors
            self.data_iterator = None

    def get_next_packet(self):
        
        if self.data_iterator:
            _, row = next(self.data_iterator)
            return row
        return None
