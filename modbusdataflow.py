import struct
import socket
import time
import random
import threading
from datetime import datetime

class ModbusProtocolSimulator:
    def __init__(self):
        self.transaction_id = 0
        self.running = False
        
        # Modbus function codes
        self.FUNC_READ_COILS = 0x01
        self.FUNC_READ_DISCRETE = 0x02
        self.FUNC_READ_HOLDING = 0x03
        self.FUNC_READ_INPUT = 0x04
        self.FUNC_WRITE_SINGLE_COIL = 0x05
        self.FUNC_WRITE_SINGLE_REGISTER = 0x06
    
    def create_modbus_tcp_packet(self, unit_id, function_code, start_addr, quantity):
        """Create a proper Modbus TCP packet"""
        self.transaction_id += 1
        
        # Modbus TCP ADU (Application Data Unit) structure:
        # Transaction ID (2 bytes) + Protocol ID (2 bytes) + Length (2 bytes) + Unit ID (1 byte) + PDU
        
        # Protocol ID is always 0 for Modbus
        protocol_id = 0x0000
        
        # Create PDU (Protocol Data Unit)
        pdu = struct.pack('>BBH', function_code, start_addr, quantity)
        
        # Length = Unit ID (1 byte) + PDU length
        length = 1 + len(pdu)
        
        # Create full TCP packet
        tcp_header = struct.pack('>HHHB', 
                                self.transaction_id, 
                                protocol_id, 
                                length, 
                                unit_id)
        
        return tcp_header + pdu
    
    def create_modbus_response(self, unit_id, function_code, data_values):
        """Create a Modbus response packet"""
        self.transaction_id += 1
        
        # For read operations, response includes byte count + data
        if function_code in [0x03, 0x04]:  # Read holding/input registers
            byte_count = len(data_values) * 2  # 2 bytes per register
            pdu = struct.pack('>BB', function_code, byte_count)
            for value in data_values:
                pdu += struct.pack('>H', value)
        
        length = 1 + len(pdu)
        tcp_header = struct.pack('>HHHB', 
                                self.transaction_id, 
                                0x0000, 
                                length, 
                                unit_id)
        
        return tcp_header + pdu
    
    def simulate_modbus_master(self, host='127.0.0.1', port=502, rtu_addresses=[1, 2, 3], interval=2.0):
        """Simulate a Modbus master polling multiple RTUs"""
        print(f"Starting Modbus Master simulation - polling RTUs {rtu_addresses} every {interval}s")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            
            while self.running:
                for rtu_id in rtu_addresses:
                    # Simulate reading holding registers
                    query_packet = self.create_modbus_tcp_packet(
                        unit_id=rtu_id,
                        function_code=self.FUNC_READ_HOLDING,
                        start_addr=0,  # Starting register address
                        quantity=10    # Number of registers to read
                    )
                    
                    try:
                        sock.send(query_packet)
                        timestamp = datetime.now().isoformat()
                        print(f"[{timestamp}] Master -> RTU {rtu_id}: Read Holding Registers (addr:0, qty:10)")
                        
                        # Wait for response (in real implementation)
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"Error sending to RTU {rtu_id}: {e}")
                
                time.sleep(interval)
        
        except Exception as e:
            print(f"Master connection error: {e}")
        finally:
            sock.close()
    
    def simulate_modbus_slave(self, host='127.0.0.1', port=502, unit_id=1):
        """Simulate a Modbus RTU/slave responding to requests"""
        print(f"Starting Modbus Slave simulation - Unit ID {unit_id}")
        
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((host, port))
            server_sock.listen(5)
            
            while self.running:
                try:
                    client_sock, addr = server_sock.accept()
                    print(f"Connection from {addr}")
                    
                    while self.running:
                        data = client_sock.recv(1024)
                        if not data:
                            break
                        
                        # Parse Modbus TCP packet (simplified)
                        if len(data) >= 8:  # Minimum TCP packet size
                            trans_id, proto_id, length, unit_id_recv = struct.unpack('>HHHB', data[:7])
                            
                            if unit_id_recv == unit_id:
                                # Generate simulated register values
                                simulated_values = [random.randint(100, 1000) for _ in range(10)]
                                
                                response = self.create_modbus_response(
                                    unit_id=unit_id,
                                    function_code=0x03,  # Read holding registers response
                                    data_values=simulated_values
                                )
                                
                                client_sock.send(response)
                                timestamp = datetime.now().isoformat()
                                print(f"[{timestamp}] RTU {unit_id} -> Master: Response with {len(simulated_values)} values")
                
                except Exception as e:
                    print(f"Slave error: {e}")
                finally:
                    try:
                        client_sock.close()
                    except:
                        pass
        
        except Exception as e:
            print(f"Slave setup error: {e}")
        finally:
            server_sock.close()
    
    def start_simulation(self, mode='master', **kwargs):
        """Start the Modbus simulation"""
        self.running = True
        
        if mode == 'master':
            self.simulate_modbus_master(**kwargs)
        elif mode == 'slave':
            self.simulate_modbus_slave(**kwargs)
        elif mode == 'both':
            # Run both master and slave in separate threads
            slave_thread = threading.Thread(
                target=self.simulate_modbus_slave,
                kwargs={'port': 502, 'unit_id': 1},
                daemon=True
            )
            slave_thread.start()
            
            time.sleep(1)  # Give slave time to start
            
            master_thread = threading.Thread(
                target=self.simulate_modbus_master,
                kwargs={'port': 502, 'rtu_addresses': [1], 'interval': 2.0},
                daemon=True
            )
            master_thread.start()
            
            try:
                input("Press Enter to stop simulation...")
            except KeyboardInterrupt:
                pass
            finally:
                self.stop_simulation()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        print("Modbus simulation stopped")

# Example usage:
def main():
    simulator = ModbusProtocolSimulator()
    
    print("Modbus Protocol Simulator")
    print("1. Master only")
    print("2. Slave only") 
    print("3. Both (Master + Slave)")
    
    choice = input("Select mode (1-3): ").strip()
    
    if choice == '1':
        rtu_list = [1, 2, 3]  # RTU addresses to poll
        simulator.start_simulation(mode='master', rtu_addresses=rtu_list, interval=2.0)
    elif choice == '2':
        unit_id = int(input("Enter RTU Unit ID (default: 1): ") or "1")
        simulator.start_simulation(mode='slave', unit_id=unit_id)
    else:
        simulator.start_simulation(mode='both')

if __name__ == "__main__":
    main()