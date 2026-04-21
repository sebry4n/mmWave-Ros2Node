import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

import serial
import time
import struct

# --- CONFIGURATION ---
CLI_PORT = '/dev/ttyUSB0'  # Update for Linux (usually ttyUSB0 or ttyACM0)
DATA_PORT = '/dev/ttyUSB1' 
# For Windows, use 'COM3' and 'COM4'

cfg = [
    "sensorStop", "flushCfg", "dfeDataOutputMode 1", "channelCfg 15 3 0",
    "adcCfg 2 1", "adcbufCfg -1 0 0 1 1",
    "profileCfg 0 77 7 7 58.0 0 0 67.978 1 256 5020 0 0 36",
    "chirpCfg 0 0 0 0 0 0 0 1", "chirpCfg 1 1 0 0 0 0 0 2",
    "frameCfg 0 1 32 0 100 1 0", "lowPower 0 1", "guiMonitor 1 1 0 0",
    "cfarCfg 1 4 12 4 2 8 2 350 30 2 0 5 20", "dbscanCfg 4 4 13 20 3 256",
    "sensorStart"
]

class MMwaveODSNode(Node):
    def __init__(self):
        super().__init__('mmwave_ods_node')
        self.publisher_ = self.create_publisher(PointCloud2, 'radar/points', 10)
        
        # Initialize Serial
        self.send_config()
        self.ser_data = serial.Serial(DATA_PORT, 921600, timeout=0.1)
        
        self.magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        self.buffer = b''
        
        # Timer to poll serial data (100Hz)
        self.timer = self.create_timer(0.01, self.read_and_parse)
        self.get_logger().info("mmWave ODS ROS 2 Node Started")

    def send_config(self):
        try:
            with serial.Serial(CLI_PORT, 115200, timeout=1) as ser:
                for line in cfg:
                    ser.write((line + '\n').encode())
                    time.sleep(0.1)
                    echo = ser.read_all().decode(errors='ignore')
                    if "Error" in echo:
                        self.get_logger().error(f"Config Error: {echo.strip()}")
            self.get_logger().info("Configuration sent successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to send config: {e}")

    def parse_detected_objects(self, data, num_objects, q_format_value):
        objects = []
        xyz_q_format = 2**q_format_value
        for i in range(num_objects):
            offset = i * 12
            try:
                # speed, x, y, z (signed short), peak (unsigned short)
                speed, x, y, z, peak = struct.unpack('<hhhhH', data[offset:offset+10])
                # Convert to meters and append as [x, y, z, intensity/peak]
                objects.append([
                    float(x / xyz_q_format),
                    float(y / xyz_q_format),
                    float(z / xyz_q_format),
                    float(peak)
                ])
            except: continue
        return objects

    def read_and_parse(self):
        if self.ser_data.in_waiting > 0:
            self.buffer += self.ser_data.read(self.ser_data.in_waiting)
            
            if self.magic_word in self.buffer:
                idx = self.buffer.index(self.magic_word)
                if len(self.buffer[idx:]) < 40: return 
                
                header = self.buffer[idx:idx+40]
                total_len = struct.unpack('<I', header[12:16])[0]
                num_tlvs = struct.unpack('<I', header[32:36])[0]
                
                if len(self.buffer[idx:]) < total_len: return
                
                payload = self.buffer[idx+40 : idx+total_len]
                tlv_ptr = 0
                all_points = []

                for _ in range(num_tlvs):
                    t_type, t_len = struct.unpack('<II', payload[tlv_ptr : tlv_ptr+8])
                    t_data = payload[tlv_ptr+8 : tlv_ptr+8+t_len]
                    
                    if t_type == 1: # Detected Objects
                        num_obj = struct.unpack('<H', t_data[0:2])[0]
                        q_val = struct.unpack('<H', t_data[2:4])[0]
                        all_points = self.parse_detected_objects(t_data[4:], num_obj, q_val)
                    
                    tlv_ptr += (8 + t_len)

                if all_points:
                    self.publish_pc2(all_points)
                
                self.buffer = self.buffer[idx+total_len:]

    def publish_pc2(self, points):
        # points is a list of [x, y, z, peak]
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'radar_link'
        
        # Define fields: x, y, z, and intensity (peak)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        pc2_msg = pc2.create_cloud(header, fields, points)
        self.publisher_.publish(pc2_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MMwaveODSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.ser_data.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
