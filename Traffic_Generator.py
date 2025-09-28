
import socket
import json
import time
import random
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import os
import sys
import struct

class NetworkTrafficGenerator:
    def __init__(self, csv_file="MB_n_TCP.csv"):
        self.csv_file = csv_file
        self.data = None
        self.running = False
        self.current_index = 0
        self.transaction_id = 0
        
        self.stats = {
            'tcp_packets_sent': 0,
            'modbus_packets_sent': 0,
            'total_packets_sent': 0,
            'start_time': None,
            'total_rows': 0,
            'protocol_stats': {},
            'connection_attempts': 0,
            'failed_connections': 0
        }
        
        self.network_config = {
            'tcp_ports': [502, 4362, 4363, 4840, 10000, 20000],
            'modbus_ports': [502, 503],
            'host': '127.0.0.1',
            'external_hosts': ['192.168.1.99', '192.168.1.100', '192.168.1.101', '192.168.1.102']
        }
        
        self.timing_patterns = {
            'tcp': {
                'min_interval': 0.1,
                'max_interval': 5.0,
                'burst_probability': 0.3,
                'burst_size': (3, 10)
            },
            'modbus': {
                'min_interval': 0.5,
                'max_interval': 3.0,
                'burst_probability': 0.1,
                'burst_size': (2, 5)
            }
        }
        
        self.load_csv_data()
    
    def load_csv_data(self):
        try:
            if not os.path.exists(self.csv_file):
                csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
                if csv_files:
                    self.csv_file = csv_files[0]
                    print(f"Using CSV file: {self.csv_file}")
                else:
                    print("No CSV files found!")
                    return False
            
            self.data = pd.read_csv(self.csv_file)
            self.stats['total_rows'] = len(self.data)
            
            if 'Protocol' in self.data.columns:
                protocol_counts = self.data['Protocol'].value_counts()
                for protocol, count in protocol_counts.items():
                    self.stats['protocol_stats'][protocol] = count
            
            print(f"Loaded {len(self.data)} traffic records from {self.csv_file}")
            return True
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def get_random_packet(self, protocol_filter=None):
        if self.data is None or len(self.data) == 0:
            return None
        
        if protocol_filter and 'Protocol' in self.data.columns:
            filtered_data = self.data[self.data['Protocol'].str.contains(protocol_filter, na=False)]
            if len(filtered_data) == 0:
                return None
            row = filtered_data.sample(n=1).iloc[0]
        else:
            row = self.data.sample(n=1).iloc[0]
        
        return self.convert_row_to_packet(row)
    
    def convert_row_to_packet(self, row):
        packet = {
            'timestamp': datetime.now().isoformat(),
            'packet_id': f"pkt_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            'original_data': {}
        }
        
        column_mapping = {
            'Time': 'original_time',
            'Source': 'source_ip',
            'Destination': 'destination_ip', 
            'Protocol': 'protocol',
            'Length': 'packet_length',
            'SrcPort': 'source_port',
            'DstPort': 'destination_port',
            'Flags': 'tcp_flags',
            'Seq': 'sequence_number',
            'Ack': 'acknowledgment_number',
            'Win': 'window_size',
            'Len': 'payload_length',
            'MSS': 'max_segment_size',
            'Direction': 'direction',
            'TransID': 'transaction_id',
            'UnitID': 'unit_id',
            'FuncCode': 'function_code',
            'FuncDesc': 'function_description',
            'Other': 'additional_info'
        }
        
        for csv_col, packet_key in column_mapping.items():
            if csv_col in row.index and pd.notna(row[csv_col]):
                packet['original_data'][packet_key] = row[csv_col]
        
        protocol = str(row.get('Protocol', 'Unknown'))
        
        if 'Modbus' in protocol:
            packet.update(self.process_modbus_packet(row))
        elif 'TCP' in protocol:
            packet.update(self.process_tcp_packet(row))
        
        packet['network_metadata'] = {
            'generated_at': time.time(),
            'source_csv_row': int(row.name) if hasattr(row, 'name') else 0,
            'traffic_pattern': self.determine_traffic_pattern(packet)
        }
        
        return packet
    
    def process_modbus_packet(self, row):
        modbus_data = {
            'protocol_type': 'modbus',
            'modbus_details': {}
        }
        
        if pd.notna(row.get('UnitID')):
            modbus_data['modbus_details']['unit_id'] = int(float(row['UnitID']))
        
        if pd.notna(row.get('FuncCode')):
            modbus_data['modbus_details']['function_code'] = int(float(row['FuncCode']))
        
        if pd.notna(row.get('FuncDesc')):
            modbus_data['modbus_details']['function_description'] = str(row['FuncDesc'])
            modbus_data['command'] = str(row['FuncDesc'])
        
        if pd.notna(row.get('TransID')):
            modbus_data['modbus_details']['transaction_id'] = int(float(row['TransID']))
        
        # func_code = modbus_data['modbus_details'].get('function_code', 3)
        # if func_code in [3, 4]:
        #     register_count = random.randint(1, 10)
        #     modbus_data['modbus_details']['registers'] = {
        #         f'reg_{i}': random.randint(0, 65535) for i in range(register_count)
        #     }
        
        return modbus_data
    
    def process_tcp_packet(self, row):
        tcp_data = {
            'protocol_type': 'tcp',
            'tcp_details': {}
        }

        flags = str(row.get('Flags', '')) if pd.notna(row.get('Flags')) else ''
        tcp_data['command'] = f"TCP {flags}" if flags else "TCP packet"
        
        if pd.notna(row.get('Flags')):
            tcp_data['tcp_details']['flags'] = str(row['Flags'])
        
        if pd.notna(row.get('Seq')):
            tcp_data['tcp_details']['sequence_number'] = int(float(row['Seq']))
        
        if pd.notna(row.get('Ack')):
            tcp_data['tcp_details']['acknowledgment_number'] = int(float(row['Ack']))
        
        if pd.notna(row.get('Win')):
            tcp_data['tcp_details']['window_size'] = int(float(row['Win']))
        
        flags = tcp_data['tcp_details'].get('flags', '')
        if 'SYN' in flags and 'ACK' not in flags:
            tcp_data['tcp_details']['connection_state'] = 'syn_sent'
        elif 'SYN' in flags and 'ACK' in flags:
            tcp_data['tcp_details']['connection_state'] = 'syn_ack'
        elif 'FIN' in flags:
            tcp_data['tcp_details']['connection_state'] = 'fin_sent'
        elif 'ACK' in flags:
            tcp_data['tcp_details']['connection_state'] = 'established'
        else:
            tcp_data['tcp_details']['connection_state'] = 'unknown'
        
        return tcp_data
    
    def determine_traffic_pattern(self, packet):
        protocol = packet.get('protocol_type', 'unknown')
        dest_port = packet['original_data'].get('destination_port', 0)
        
        patterns = []
        
        hour = datetime.now().hour
        if 8 <= hour <= 17:
            patterns.append('business_hours')
        elif 18 <= hour <= 22:
            patterns.append('evening_maintenance')
        else:
            patterns.append('overnight_monitoring')
        
        if dest_port in [502, 503]:
            patterns.append('modbus_industrial')
        elif dest_port in [80, 443, 8080]:
            patterns.append('web_traffic')
        elif dest_port in [20000, 44818]:
            patterns.append('custom_industrial')
        
        if protocol == 'modbus':
            patterns.append('scada_polling')
        elif protocol == 'tcp':
            patterns.append('network_communication')
        
        return patterns
    
    def get_random_timing(self, protocol_type):
        pattern = self.timing_patterns.get(protocol_type, self.timing_patterns['tcp'])
        
        base_interval = random.uniform(pattern['min_interval'], pattern['max_interval'])
        
        hour = datetime.now().hour
        if 2 <= hour <= 6:
            base_interval *= random.uniform(1.5, 3.0)
        elif 8 <= hour <= 17:
            base_interval *= random.uniform(0.3, 0.8)
        elif 18 <= hour <= 22:
            base_interval *= random.uniform(0.8, 1.2)
        
        jitter = random.uniform(-0.2, 0.2) * base_interval
        final_interval = max(0.05, base_interval + jitter)
        
        return final_interval
    
    def should_generate_burst(self, protocol_type):
        pattern = self.timing_patterns.get(protocol_type, self.timing_patterns['tcp'])
        return random.random() < pattern['burst_probability']
    
    def get_burst_size(self, protocol_type):
        pattern = self.timing_patterns.get(protocol_type, self.timing_patterns['tcp'])
        return random.randint(*pattern['burst_size'])
    
    def send_tcp_traffic(self, target_ports=[4362, 4363, 10000], duration_hours=24):
        print(f"Starting TCP traffic generation...")
        
        end_time = time.time() + (duration_hours * 3600)
        
        try:
            while self.running and time.time() < end_time:
                packet = self.get_random_packet('TCP')
                if packet is None:
                    packet = self.get_random_packet()
                
                if packet is None:
                    time.sleep(1)
                    continue
                
                target_port = random.choice(target_ports)
                target_host = random.choice(self.network_config['external_hosts'])
                
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    
                    traffic_packet = {
                        'traffic_type': 'tcp_simulation',
                        'target': f"{target_host}:{target_port}",
                        'timestamp': datetime.now().isoformat(),
                        'packet_data': packet
                    }
                    
                    message = json.dumps(traffic_packet, default=str)
                    sock.sendto(message.encode('utf-8'), ('127.0.0.1', 9991))
                    sock.close()
                    
                    self.stats['tcp_packets_sent'] += 1
                    
                    if self.stats['tcp_packets_sent'] % 100 == 0:
                        print(f"TCP: Sent {self.stats['tcp_packets_sent']} packets")
                
                except Exception as e:
                    print(f"TCP error: {e}")
                
                if self.should_generate_burst('tcp'):
                    burst_size = self.get_burst_size('tcp')
                    
                    for _ in range(burst_size - 1):
                        if not self.running:
                            break
                        
                        burst_packet = self.get_random_packet('TCP')
                        if burst_packet:
                            try:
                                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                                burst_traffic = {
                                    'traffic_type': 'tcp_burst',
                                    'target': f"{target_host}:{target_port}",
                                    'packet_data': burst_packet
                                }
                                message = json.dumps(burst_traffic, default=str)
                                sock.sendto(message.encode('utf-8'), ('127.0.0.1', 9991))
                                sock.close()
                                
                                self.stats['tcp_packets_sent'] += 1
                            except:
                                pass
                            
                            time.sleep(random.uniform(0.05, 0.2))
                
                sleep_time = self.get_random_timing('tcp')
                time.sleep(sleep_time)
        
        except Exception as e:
            print(f"TCP traffic generation error: {e}")
    
    def send_modbus_traffic(self, target_ports=[502, 503], duration_hours=24):
        print(f"Starting Modbus traffic generation...")
        
        end_time = time.time() + (duration_hours * 3600)
        
        try:
            while self.running and time.time() < end_time:
                packet = self.get_random_packet('Modbus')
                if packet is None:
                    packet = self.generate_synthetic_modbus_packet()
                
                target_port = random.choice(target_ports)
                target_host = random.choice(self.network_config['external_hosts'])
                
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    
                    modbus_packet = {
                        'traffic_type': 'modbus_simulation',
                        'target': f"{target_host}:{target_port}",
                        'timestamp': datetime.now().isoformat(),
                        'packet_data': packet
                    }
                    
                    message = json.dumps(modbus_packet, default=str)
                    sock.sendto(message.encode('utf-8'), ('127.0.0.1', 9992))
                    sock.close()
                    
                    self.stats['modbus_packets_sent'] += 1
                    
                    if self.stats['modbus_packets_sent'] % 50 == 0:
                        print(f"Modbus: Sent {self.stats['modbus_packets_sent']} packets")
                
                except Exception as e:
                    print(f"Modbus send error: {e}")
                
                sleep_time = self.get_random_timing('modbus')
                time.sleep(sleep_time)
        
        except Exception as e:
            print(f"Modbus traffic generation error: {e}")
    
    def generate_synthetic_modbus_packet(self):
        return {
            'protocol_type': 'modbus',
            'synthetic': True,
            'timestamp': datetime.now().isoformat(),
            'original_data': {
                'source_ip': random.choice(self.network_config['external_hosts']),
                'destination_ip': random.choice(self.network_config['external_hosts']),
                'protocol': 'Modbus/TCP',
                'source_port': random.randint(1024, 65535),
                'destination_port': random.choice([502, 503])
            },
            'modbus_details': {
                'transaction_id': random.randint(1, 65535),
                'unit_id': random.randint(1, 10),
                'function_code': random.choice([1, 2, 3, 4, 5, 6, 15, 16]),
                'function_description': random.choice([
                    'Read Coils', 'Read Discrete Inputs', 'Read Holding Registers',
                    'Read Input Registers', 'Write Single Coil', 'Write Single Register'
                ])
            }
        }
    
    def monitor_traffic_stats(self):
        while self.running:
            time.sleep(60)
            
            if self.running:
                duration = datetime.now() - self.stats['start_time']
                total_packets = self.stats['tcp_packets_sent'] + self.stats['modbus_packets_sent']
                
                if duration.total_seconds() > 0:
                    pps = total_packets / duration.total_seconds()
                    print(f"Traffic Stats: {total_packets} total packets, {pps:.2f} pps")
    
    def start_traffic_generation(self, duration_hours=24, protocols=['tcp', 'modbus']):
        if self.data is None:
            print("Cannot start traffic generation: No data loaded")
            return False
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        print(f"Starting traffic generation for {duration_hours} hours with protocols: {protocols}")
        
        threads = []
        
        if 'tcp' in protocols:
            tcp_thread = threading.Thread(
                target=self.send_tcp_traffic,
                args=([4362, 4363, 10000], duration_hours),
                daemon=True
            )
            threads.append(tcp_thread)
            tcp_thread.start()
        
        if 'modbus' in protocols:
            modbus_thread = threading.Thread(
                target=self.send_modbus_traffic,
                args=([502, 503], duration_hours),
                daemon=True
            )
            threads.append(modbus_thread)
            modbus_thread.start()
        
        monitor_thread = threading.Thread(
            target=self.monitor_traffic_stats,
            daemon=True
        )
        threads.append(monitor_thread)
        monitor_thread.start()
        
        try:
            time.sleep(duration_hours * 3600)
        except KeyboardInterrupt:
            print("\nStopping traffic generation...")
        finally:
            self.stop_traffic_generation()
        
        return True
    
    def stop_traffic_generation(self):
        self.running = False
        time.sleep(2)
        
        if self.stats['start_time']:
            duration = datetime.now() - self.stats['start_time']
            total_packets = self.stats['tcp_packets_sent'] + self.stats['modbus_packets_sent']
            
            print(f"Traffic generation completed:")
            print(f"Duration: {duration}")
            print(f"TCP packets: {self.stats['tcp_packets_sent']}")
            print(f"Modbus packets: {self.stats['modbus_packets_sent']}")
            print(f"Total packets: {total_packets}")

if __name__ == "__main__":
    generator = NetworkTrafficGenerator()
    generator.start_traffic_generation(duration_hours=1.0, protocols=['tcp', 'modbus'])