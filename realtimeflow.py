#!/usr/bin/env python3
"""
Windows-Optimized Real-time Data Simulator
Reads from Both_dataset.csv and streams data in real-time
Compatible with Python 3.13.3 on Windows 11
"""

import socket
import json
import time
import random
import threading
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import os
import sys
from pathlib import Path

class CSVDataSimulator:
    def __init__(self, csv_file="send_a_fake_command_modbus.csv"): #change the csv file to the file name you wanted, or just leave it blank
                                                                    #The progam will list out all available csv
        self.csv_file = csv_file
        self.data = None
        self.running = False
        self.current_index = 0
        self.stats = {
            'rows_sent': 0,
            'start_time': None,
            'total_rows': 0
        }
        
        # Load and validate CSV
        self.load_csv_data()
    
    def load_csv_data(self):
        """Load and prepare CSV data"""
        try:
            if not os.path.exists(self.csv_file):
                print(f"ERROR: CSV file '{self.csv_file}' not found!")
                print(f"Current directory: {os.getcwd()}")
                print("Available files:", [f for f in os.listdir('.') if f.endswith('.csv')])
                return False
            
            print(f"Loading data from {self.csv_file}...")
            self.data = pd.read_csv(self.csv_file)
            self.stats['total_rows'] = len(self.data)
            
            print(f"Successfully loaded {len(self.data)} rows")
            print(f"Columns: {list(self.data.columns)}")
            print(f"Data shape: {self.data.shape}")
            
            # Show sample of data
            print("\nSample data (first 3 rows):")
            print(self.data.head(3).to_string())
            
            return True
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def get_next_row(self):
        """Get next row from dataset (cycles through data)"""
        if self.data is None or len(self.data) == 0:
            return None
        
        # Get current row
        row = self.data.iloc[self.current_index]
        
        # Move to next row (cycle back to start if at end)
        self.current_index = (self.current_index + 1) % len(self.data)
        
        # Convert to dictionary with meaningful structure
        row_dict = {
            'simulation_timestamp': datetime.now().isoformat(),
            'row_index': self.current_index,
            'energy_data': {
                'interval_datetime': str(row['INTERVAL_DATETIME']),
                'device_id': str(row['DUID']),
                'mwh_reading': float(row['MWH_READING']) if pd.notna(row['MWH_READING']) else 0.0,
                'last_changed': str(row['LASTCHANGED']),
                'mwh_readings_anomaly': float(row['MWH_READINGS_ANOMALY']) if pd.notna(row['MWH_READINGS_ANOMALY']) else 0.0,
                'is_anomaly': bool(row['is_Anomaly']) if pd.notna(row['is_Anomaly']) else False
            },
            'metadata': {
                'source_file': self.csv_file,
                'total_rows': self.stats['total_rows']
            }
        }
        
        return row_dict
    
    def send_udp_data(self, host='127.0.0.1', port=9999, interval=1.0):
        """Send CSV data via UDP packets"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"Starting UDP transmission to {host}:{port}")
            
            while self.running:
                row_data = self.get_next_row()
                if row_data is None:
                    print("No data available, stopping UDP transmission")
                    break
                
                try:
                    # Convert data to JSON and send
                    message = json.dumps(row_data, default=str)  # default=str handles numpy types
                    sock.sendto(message.encode('utf-8'), (host, port))
                    
                    self.stats['rows_sent'] += 1
                    print(f"UDP: Sent row {self.stats['rows_sent']} ({len(message)} bytes)")
                    
                except Exception as e:
                    print(f"Error sending UDP data: {e}")
                
                time.sleep(interval)
            
            sock.close()
            print("UDP transmission stopped")
            
        except Exception as e:
            print(f"UDP setup error: {e}")
    
    def send_tcp_data(self, host='127.0.0.1', port=9998, interval=1.0):
        """Send CSV data via TCP connection"""
        print(f"Starting TCP transmission to {host}:{port}")
        
        while self.running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)  # 10 second timeout
                sock.connect((host, port))
                
                while self.running:
                    row_data = self.get_next_row()
                    if row_data is None:
                        print("No data available, stopping TCP transmission")
                        break
                    
                    try:
                        message = json.dumps(row_data, default=str) + '\n'
                        sock.send(message.encode('utf-8'))
                        
                        self.stats['rows_sent'] += 1
                        print(f"TCP: Sent row {self.stats['rows_sent']} ({len(message)} bytes)")
                        
                    except Exception as e:
                        print(f"Error sending TCP data: {e}")
                        break
                    
                    time.sleep(interval)
                
                sock.close()
                
            except ConnectionRefusedError:
                print("TCP connection refused, retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                print(f"TCP connection error: {e}")
                time.sleep(5)
        
        print("TCP transmission stopped")
    
    def write_streaming_csv(self, output_file='streaming_output.csv', interval=1.0):
        """Write data to a new CSV file for file-based monitoring"""
        try:
            # Prepare output file
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = None
                
                print(f"Writing streaming data to {output_file}")
                
                while self.running:
                    row_data = self.get_next_row()
                    if row_data is None:
                        print("No data available, stopping file write")
                        break
                    
                    # Initialize writer with first row's keys
                    if writer is None:
                        fieldnames = list(row_data.keys())
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                    
                    try:
                        writer.writerow(row_data)
                        csvfile.flush()  # Ensure immediate write
                        
                        self.stats['rows_sent'] += 1
                        print(f"File: Wrote row {self.stats['rows_sent']}")
                        
                    except Exception as e:
                        print(f"Error writing to file: {e}")
                    
                    time.sleep(interval)
            
            print(f"File writing stopped. Output saved to {output_file}")
            
        except Exception as e:
            print(f"File write error: {e}")
    
    def start_simulation(self, methods=['udp', 'tcp', 'file'], interval=1.0, duration=None):
        """Start simulation with specified methods"""
        if self.data is None:
            print("Cannot start simulation: No data loaded")
            return False
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        print(f"\n=== Starting CSV Data Simulation ===")
        print(f"Dataset: {self.csv_file}")
        print(f"Total rows: {self.stats['total_rows']}")
        print(f"Transmission interval: {interval} seconds")
        print(f"Methods: {methods}")
        
        # Start transmission methods in separate threads
        threads = []
        
        if 'udp' in methods:
            udp_thread = threading.Thread(
                target=self.send_udp_data, 
                args=('127.0.0.1', 9999, interval),
                daemon=True
            )
            threads.append(udp_thread)
        
        if 'tcp' in methods:
            tcp_thread = threading.Thread(
                target=self.send_tcp_data,
                args=('127.0.0.1', 9998, interval),
                daemon=True
            )
            threads.append(tcp_thread)
        
        if 'file' in methods:
            file_thread = threading.Thread(
                target=self.write_streaming_csv,
                args=('streaming_output.csv', interval),
                daemon=True
            )
            threads.append(file_thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        try:
            if duration:
                print(f"Running for {duration} seconds...")
                time.sleep(duration)
            else:
                print("Press Enter to stop simulation...")
                input()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        finally:
            self.stop_simulation()
        
        return True
    
    def stop_simulation(self):
        """Stop simulation and print statistics"""
        print("\n=== Stopping Simulation ===")
        self.running = False
        time.sleep(2)  # Give threads time to cleanup
        
        if self.stats['start_time']:
            duration = datetime.now() - self.stats['start_time']
            print(f"Duration: {duration}")
            print(f"Rows sent: {self.stats['rows_sent']}")
            if duration.total_seconds() > 0:
                rate = self.stats['rows_sent'] / duration.total_seconds()
                print(f"Average rate: {rate:.2f} rows/second")
        
        print("Simulation stopped")

def main():
    print("=== CSV-Based Real-time Data Simulator ===")
    print(f"Python Version: {sys.version}")
    print(f"Current Directory: {os.getcwd()}")
    
    # Check for CSV file
    csv_file = "Both_dataset.csv"
    if not os.path.exists(csv_file):
        print(f"\nCSV file '{csv_file}' not found in current directory.")
        
        # Look for other CSV files
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            print("Available CSV files:")
            for i, file in enumerate(csv_files, 1):
                print(f"  {i}. {file}")
            
            choice = input(f"\nEnter file number (1-{len(csv_files)}) or filename: ").strip()
            
            try:
                if choice.isdigit():
                    csv_file = csv_files[int(choice) - 1]
                else:
                    csv_file = choice
            except (IndexError, ValueError):
                print("Invalid choice, using default filename")
        else:
            print("No CSV files found in current directory!")
            return
    
    # Initialize simulator
    simulator = CSVDataSimulator(csv_file)
    
    if simulator.data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Configuration
    print(f"\nConfiguration options:")
    print("1. UDP only (port 9999)")
    print("2. TCP only (port 9998)")  
    print("3. File output only")
    print("4. All methods (UDP + TCP + File)")
    
    choice = input("Select option (1-4, default: 4): ").strip() or "4"
    
    method_map = {
        '1': ['udp'],
        '2': ['tcp'], 
        '3': ['file'],
        '4': ['udp', 'tcp', 'file']
    }
    
    methods = method_map.get(choice, ['udp', 'tcp', 'file'])
    
    # Get interval
    try:
        interval = float(input("Enter transmission interval in seconds (default: 1.0): ") or "1.0")
    except ValueError:
        interval = 1.0
    
    # Get duration
    duration_input = input("Enter duration in seconds (default: continuous): ").strip()
    duration = None
    if duration_input:
        try:
            duration = float(duration_input)
        except ValueError:
            duration = None
    
    # Start simulation
    simulator.start_simulation(methods=methods, interval=interval, duration=duration)

if __name__ == "__main__":
    main()