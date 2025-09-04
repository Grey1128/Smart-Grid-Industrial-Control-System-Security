#!/usr/bin/env python3
"""
Windows Data Processor
Processes data without requiring tshark installation
Works with Python 3.13.3 on Windows 11
"""

import socket
import json
import threading
import queue
import time
import sys
import os
from datetime import datetime
import pandas as pd

class WindowsDataProcessor:
    def __init__(self):
        self.running = False
        self.data_queue = queue.Queue()
        self.device_stats = {}  # Track statistics per device
        self.stats = {
            'packets_received': 0,
            'packets_processed': 0,
            'start_time': None,
            'errors': 0,
            'anomalies_detected': 0
        }
        self.server_threads = []
    
    def udp_server(self, host='127.0.0.1', port=9999):
        """UDP server to receive data"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1.0)  # 1 second timeout
            sock.bind((host, port))
            
            print(f"UDP server listening on {host}:{port}")
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(65536)  # 64KB buffer
                    
                    # Decode and queue data
                    message = data.decode('utf-8')
                    self.data_queue.put({
                        'type': 'UDP',
                        'source': addr,
                        'data': message,
                        'timestamp': datetime.now().isoformat(),
                        'size': len(data)
                    })
                    
                    self.stats['packets_received'] += 1
                    
                except socket.timeout:
                    continue  # Check if still running
                except Exception as e:
                    if self.running:  # Only log errors if we're still supposed to be running
                        print(f"UDP server error: {e}")
                        self.stats['errors'] += 1
            
            sock.close()
            print("UDP server stopped")
            
        except Exception as e:
            print(f"UDP server setup error: {e}")
    
    def tcp_server(self, host='127.0.0.1', port=9998):
        """TCP server to receive data"""
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.settimeout(1.0)
            server_sock.bind((host, port))
            server_sock.listen(5)
            
            print(f"TCP server listening on {host}:{port}")
            
            while self.running:
                try:
                    client_sock, addr = server_sock.accept()
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_tcp_client,
                        args=(client_sock, addr),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"TCP server error: {e}")
                        self.stats['errors'] += 1
            
            server_sock.close()
            print("TCP server stopped")
            
        except Exception as e:
            print(f"TCP server setup error: {e}")
    
    def handle_tcp_client(self, client_sock, addr):
        """Handle individual TCP client connection"""
        try:
            print(f"TCP client connected from {addr}")
            client_sock.settimeout(5.0)
            
            buffer = ""
            while self.running:
                try:
                    data = client_sock.recv(4096)
                    if not data:
                        break
                    
                    buffer += data.decode('utf-8')
                    
                    # Process complete lines (JSON objects ending with \n)
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            self.data_queue.put({
                                'type': 'TCP',
                                'source': addr,
                                'data': line.strip(),
                                'timestamp': datetime.now().isoformat(),
                                'size': len(line)
                            })
                            
                            self.stats['packets_received'] += 1
                
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"TCP client error: {e}")
                        self.stats['errors'] += 1
                    break
            
            client_sock.close()
            print(f"TCP client {addr} disconnected")
            
        except Exception as e:
            print(f"TCP client handler error: {e}")
    
    def file_monitor(self, filename='streaming_output.csv'):
        """Monitor CSV file for new entries"""
        print(f"Monitoring file: {filename}")
        
        last_position = 0
        last_size = 0
        
        while self.running:
            try:
                if os.path.exists(filename):
                    current_size = os.path.getsize(filename)
                    
                    # Check if file has grown
                    if current_size > last_size:
                        with open(filename, 'r', encoding='utf-8') as file:
                            file.seek(last_position)
                            new_lines = file.readlines()
                            last_position = file.tell()
                            
                            for line in new_lines:
                                if line.strip() and not line.startswith('simulation_timestamp'):
                                    self.data_queue.put({
                                        'type': 'FILE',
                                        'source': filename,
                                        'data': line.strip(),
                                        'timestamp': datetime.now().isoformat(),
                                        'size': len(line)
                                    })
                                    
                                    self.stats['packets_received'] += 1
                        
                        last_size = current_size
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                if self.running:
                    print(f"File monitor error: {e}")
                    self.stats['errors'] += 1
                time.sleep(1)
        
        print("File monitoring stopped")
    
    def process_data(self, data_entry):
        """Process individual data entry"""
        try:
            # Extract information
            data_type = data_entry['type']
            source = data_entry['source']
            raw_data = data_entry['data']
            timestamp = data_entry['timestamp']
            size = data_entry['size']
            
            # Try to parse JSON data
            try:
                parsed_data = json.loads(raw_data)
                data_dict = parsed_data
            except json.JSONDecodeError:
                # Handle non-JSON data (e.g., CSV lines)
                data_dict = {'raw_content': raw_data}
            
            # YOUR PROCESSING LOGIC HERE
            print("working")  # As requested
            
            # Print processing info
            print(f"[{data_type}] Processed data from {source} ({size} bytes)")
            
            # Optional: Print first few fields of data for debugging
            if isinstance(data_dict, dict):
                sample_fields = list(data_dict.keys())[:3]  # First 3 fields
                sample_info = {k: data_dict[k] for k in sample_fields if k in data_dict}
                print(f"  Sample fields: {sample_info}")
            
            # Add your custom processing here:
            # - Database storage
            # - Data analysis
            # - Alert generation
            # - Machine learning inference
            # - etc.
            
            self.stats['packets_processed'] += 1
            
            return True
            
        except Exception as e:
            print(f"Error processing data: {e}")
            self.stats['errors'] += 1
            return False
    
    def start_processing(self, methods=['udp', 'tcp', 'file']):
        """Start the complete processing pipeline"""
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        print(f"\n=== Starting Windows Data Processor ===")
        print(f"Python Version: {sys.version}")
        print(f"Processing methods: {methods}")
        print(f"Press Ctrl+C to stop\n")
        
        # Start server threads
        if 'udp' in methods:
            udp_thread = threading.Thread(target=self.udp_server, daemon=True)
            udp_thread.start()
            self.server_threads.append(udp_thread)
        
        if 'tcp' in methods:
            tcp_thread = threading.Thread(target=self.tcp_server, daemon=True)
            tcp_thread.start()
            self.server_threads.append(tcp_thread)
        
        if 'file' in methods:
            file_thread = threading.Thread(target=self.file_monitor, daemon=True)
            file_thread.start()
            self.server_threads.append(file_thread)
        
        # Main processing loop
        try:
            print("=== Processing started ===")
            while self.running:
                try:
                    # Get data from queue with timeout
                    data_entry = self.data_queue.get(timeout=1.0)
                    self.process_data(data_entry)
                    
                except queue.Empty:
                    continue  # No data available, keep checking
                except KeyboardInterrupt:
                    break
                    
        except Exception as e:
            print(f"Error in main processing loop: {e}")
        finally:
            self.stop_processing()
    
    def stop_processing(self):
        """Stop all processing activities"""
        print("\n=== Stopping Data Processor ===")
        self.running = False
        
        # Wait for threads to finish
        time.sleep(2)
        
        # Print statistics
        if self.stats['start_time']:
            duration = datetime.now() - self.stats['start_time']
            print(f"Processing duration: {duration}")
            print(f"Packets received: {self.stats['packets_received']}")
            print(f"Packets processed: {self.stats['packets_processed']}")
            print(f"Errors: {self.stats['errors']}")
            
            if duration.total_seconds() > 0:
                rate = self.stats['packets_processed'] / duration.total_seconds()
                print(f"Average processing rate: {rate:.2f} packets/second")
        
        print("Data processor stopped")

class CSVAnalyzer:
    """Additional utility for analyzing the CSV dataset"""
    def __init__(self, csv_file="Both_dataset.csv"):
        self.csv_file = csv_file
        self.data = None
    
    def analyze_dataset(self):
        """Analyze the CSV dataset structure"""
        try:
            if not os.path.exists(self.csv_file):
                print(f"CSV file '{self.csv_file}' not found!")
                return
            
            print(f"\n=== Analyzing {self.csv_file} ===")
            
            # Load data
            self.data = pd.read_csv(self.csv_file)
            
            # Basic info
            print(f"Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            
            # Data types
            print(f"\nData types:")
            print(self.data.dtypes.to_string())
            
            # Missing values
            print(f"\nMissing values:")
            missing = self.data.isnull().sum()
            print(missing[missing > 0].to_string())
            
            # Sample data
            print(f"\nFirst 5 rows:")
            print(self.data.head().to_string())
            
            # Statistics for numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"\nNumeric column statistics:")
                print(self.data[numeric_cols].describe().to_string())
            
            return True
            
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            return False

def main():
    print("=== Windows Data Processor ===")
    print("Choose processing mode:")
    print("1. Network processing (UDP + TCP servers)")
    print("2. File monitoring only")  
    print("3. All methods (Network + File)")
    print("4. Analyze CSV dataset first")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '4':
        # Analyze dataset first
        analyzer = CSVAnalyzer()
        if analyzer.analyze_dataset():
            input("\nPress Enter to continue to processing...")
        else:
            return
    
    # Set up processor
    processor = WindowsDataProcessor()
    
    if choice == '1':
        methods = ['udp', 'tcp']
    elif choice == '2':
        methods = ['file']
    else:
        methods = ['udp', 'tcp', 'file']
    
    print(f"\nStarting processor with methods: {methods}")
    
    # Start processing
    processor.start_processing(methods)

if __name__ == "__main__":
    main()