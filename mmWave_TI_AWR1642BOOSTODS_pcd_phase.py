import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, Header
import sensor_msgs_py.point_cloud2 as pc2

import serial
import time
import struct
import math

# --- CONFIGURATION ---
# In Surabaya/Linux, usually /dev/ttyUSB0 and /dev/ttyUSB1
CLI_PORT = '/dev/ttyUSB0'  
DATA_PORT = '/dev/ttyUSB1' 

class AWR1642ODSNode(Node):
    def __init__(self):
        super().__init__('awr1642_ods_node')
        
        # Publishers
        self.pc_pub = self.create_publisher(PointCloud2, 'radar/points', 10)
        self.phase_pub = self.create_publisher(Float32MultiArray, 'radar/phase_data', 10)
        
        # Serial Setup
        self.send_config()
        try:
            self.ser_data = serial.Serial(DATA_PORT, 921600, timeout=0.1)
        except Exception as e:
            self.get_logger().error(f"Could not open Data Port: {e}")
            return

        self.magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        self.buffer = b''
        
        # Timer to poll serial data at 100Hz
        self.timer = self.create_timer(0.01, self.main_loop)
        self.get_logger().info("AWR1642 ODS ROS 2 Node Started")

    def send_config(self):
        # Your specific ODS configuration
        cfg = [
            "sensorStop", "flushCfg", "dfeDataOutputMode 1", "channelCfg 15 3 0",
            "adcCfg 2 1", "adcbufCfg -1 0 0 1 1",
            "profileCfg 0 77 7 7 58.0 0 0 67.978 1 256 5020 0 0 36",
            "chirpCfg 0 0 0 0 0 0 0 1", "chirpCfg 1 1 0 0 0 0 0 2",
            "frameCfg 0 1 32 0 100 1 0", "lowPower 0 1", 
            "guiMonitor 1 1 0 1", # Last 1 enables Static Azimuth Heatmap (TLV 4)
            "cfarCfg 1 4 12 4 2 8 2 350 30 2 0 5 20", "dbscanCfg 4 4 13 20 3 256",
            "sensorStart"
        ]
        try:
            with serial.Serial(CLI_PORT, 115200, timeout=1) as ser:
                for line in cfg:
                    ser.write((line + '\n').encode())
                    time.sleep(0.1)
            self.get_logger().info("Configuration uploaded to AWR1642.")
        except Exception as e:
            self.get_logger().error(f"Config Upload Failed: {e}")

    def parse_detected_objects(self, data, num_objects, q_format_value):
        objects = []
        xyz_q_format = 2**q_format_value
        for i in range(num_objects):
            offset = i * 12
            try:
                # Format: speed(h), x(h), y(h), z(h), peak(H)
                speed, x, y, z, peak = struct.unpack('<hhhhH', data[offset:offset+10])
                objects.append([
                    float(x / xyz_q_format),
                    float(y / xyz_q_format),
                    float(z / xyz_q_format),
                    float(peak)
                ])
            except: continue
        return objects

    def main_loop(self):
        if self.ser_data.in_waiting > 0:
            self.buffer += self.ser_data.read(self.ser_data.in_waiting)
            
            if self.magic_word in self.buffer:
                idx = self.buffer.index(self.magic_word)
                if len(self.buffer[idx:]) < 40: return # Header incomplete
                
                header = self.buffer[idx:idx+40]
                total_len = struct.unpack('<I', header[12:16])[0]
                num_tlvs = struct.unpack('<I', header[32:36])[0]
                
                if len(self.buffer[idx:]) < total_len: return # Payload incomplete
                
                payload = self.buffer[idx+40 : idx+total_len]
                tlv_ptr = 0

                for _ in range(num_tlvs):
                    try:
                        t_type, t_len = struct.unpack('<II', payload[tlv_ptr : tlv_ptr+8])
                        t_data = payload[tlv_ptr+8 : tlv_ptr+8+t_len]
                        
                        if t_type == 1: # Detected Objects
                            num_obj = struct.unpack('<H', t_data[0:2])[0]
                            q_val = struct.unpack('<H', t_data[2:4])[0]
                            points = self.parse_detected_objects(t_data[4:], num_obj, q_val)
                            if points:
                                self.publish_pc2(points)

                        elif t_type == 4: # Static Azimuth Heatmap
                            self.extract_and_publish_phase(t_data)
                        
                        tlv_ptr += (8 + t_len)
                    except Exception: break

                # Clear processed data from buffer
                self.buffer = self.buffer[idx+total_len:]

    def extract_and_publish_phase(self, t_data):
        # AWR1642 ODS uses 4 bytes per complex sample (Imag:2, Real:2)
        num_samples = len(t_data) // 4
        phases = []
        for i in range(num_samples):
            try:
                imag, real = struct.unpack('<hh', t_data[i*4 : i*4+4])
                # Calculate phase shift in radians
                phase = math.atan2(imag, real)
                phases.append(float(phase))
            except: continue
        
        if phases:
            msg = Float32MultiArray()
            msg.data = phases
            self.phase_pub.publish(msg)

    def publish_pc2(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'radar_link'
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pc2_msg = pc2.create_cloud(header, fields, points)
        self.pc_pub.publish(pc2_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AWR1642ODSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'ser_data'):
            node.ser_data.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
