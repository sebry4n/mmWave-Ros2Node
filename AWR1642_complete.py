import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, Float32, Header
import sensor_msgs_py.point_cloud2 as pc2

import serial
import time
import struct
import math
import numpy as np
from scipy.signal import butter, lfilter

class AWR1642VitalSignsNode(Node):
    def __init__(self):
        super().__init__('awr1642_vital_signs_node')
        self.declare_parameter('cli_port', '/dev/ttyACM1')
        self.declare_parameter('data_port', '/dev/ttyACM2')
        self.cli_port = self.get_parameter('cli_port').get_parameter_value().string_value
        self.data_port = self.get_parameter('data_port').get_parameter_value().string_value
        
        # Publishers
        self.pc_pub = self.create_publisher(PointCloud2, 'mmwave_ti/points', 10)
        self.phase_pub = self.create_publisher(Float32MultiArray, 'mmwave_ti/phase_data', 10)
        self.hr_pub = self.create_publisher(Float32, 'mmwave_ti/heart_rate', 10)
        self.rr_pub = self.create_publisher(Float32, 'mmwave_ti/breath_rate', 10)
        self.last_stable_rr = 15.0
        
        # Configuration
        self.send_config()
        self.ser_data = serial.Serial(self.data_port, 921600, timeout=0.1)
        self.magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        self.buffer = b''
        
        # Signal Processing Variables
        self.fs = 10.0
        self.phase_history = []
        self.window_size = 120 # 12 seconds
        self.last_stable_bpm = 75.0
        
        self.timer = self.create_timer(0.01, self.main_loop)
        self.get_logger().info("Vital Signs Node Initialized")

    def send_config(self):
        cfg = [
            "sensorStop", "flushCfg", "dfeDataOutputMode 1", "channelCfg 15 3 0",
            "adcCfg 2 1", "adcbufCfg -1 0 0 1 1",
            "profileCfg 0 77 7 7 58.0 0 0 67.978 1 256 5020 0 0 36",
            "chirpCfg 0 0 0 0 0 0 0 1", "chirpCfg 1 1 0 0 0 0 0 2",
            "frameCfg 0 1 32 0 100 1 0", "lowPower 0 1", "guiMonitor 1 1 0 1", 
            "cfarCfg 1 4 12 4 2 8 2 350 30 2 0 5 20", "dbscanCfg 4 4 13 20 3 256",
            "sensorStart"
        ]
        with serial.Serial(self.cli_port, 115200, timeout=1) as ser:
            for line in cfg:
                ser.write((line + '\n').encode())
                time.sleep(0.1)

    def process_heart_rate(self, current_phase):
        if not self.phase_history:
            self.phase_history.append(current_phase)
            return
        
        # Unwrapping
        diff = current_phase - self.phase_history[-1]
        unwrapped = self.phase_history[-1] + (diff + math.pi) % (2 * math.pi) - math.pi
        self.phase_history.append(unwrapped)
        if len(self.phase_history) > self.window_size: self.phase_history.pop(0)
        
        if len(self.phase_history) >= 60:
            signal = np.array(self.phase_history) - np.mean(self.phase_history)
            # Bandpass Filter (0.8 - 2.5Hz)
            b, a = butter(4, [0.8/(self.fs/2), 2.5/(self.fs/2)], btype='band')
            filtered = lfilter(b, a, signal) * np.hanning(len(signal))
            
            # FFT
            fft_res = np.abs(np.fft.rfft(filtered, n=512))
            freqs = np.fft.rfftfreq(512, 1/self.fs)
            hr_bins = np.where((freqs >= 0.8) & (freqs <= 2.5))[0]
            
            if len(hr_bins) > 0:
                raw_bpm = freqs[hr_bins[np.argmax(fft_res[hr_bins])]] * 60.0
                self.last_stable_bpm = (0.15 * raw_bpm) + (0.85 * self.last_stable_bpm)
                self.hr_pub.publish(Float32(data=float(self.last_stable_bpm)))

    def process_respiration_rate(self, current_phase):
        if len(self.phase_history) < 60: return
        
        signal = np.array(self.phase_history) - np.mean(self.phase_history)
        
        # Bandpass Filter (0.1 - 0.5Hz)
        b, a = butter(4, [0.1/(self.fs/2), 0.5/(self.fs/2)], btype='band')
        filtered = lfilter(b, a, signal) * np.hanning(len(signal))
        
        # FFT
        fft_res = np.abs(np.fft.rfft(filtered, n=512))
        freqs = np.fft.rfftfreq(512, 1/self.fs)
        rr_bins = np.where((freqs >= 0.1) & (freqs <= 0.5))[0]
        
        if len(rr_bins) > 0:
            raw_rr = freqs[rr_bins[np.argmax(fft_res[rr_bins])]] * 60.0
            #smoothing
            self.last_stable_rr = (0.15 * raw_rr) + (0.85 * self.last_stable_rr)
            self.rr_pub.publish(Float32(data=float(self.last_stable_rr)))

    def main_loop(self):
        if self.ser_data.in_waiting == 0: return
        self.buffer += self.ser_data.read(self.ser_data.in_waiting)
        if self.magic_word not in self.buffer: return
        
        idx = self.buffer.index(self.magic_word)
        if len(self.buffer[idx:]) < 40: return
        
        total_len = struct.unpack('<I', self.buffer[idx+12:idx+16])[0]
        if len(self.buffer[idx:]) < total_len: return
        
        payload = self.buffer[idx+40 : idx+total_len]
        tlv_ptr = 0
        num_tlvs = struct.unpack('<I', self.buffer[idx+32:idx+36])[0]

        for _ in range(num_tlvs):
            t_type, t_len = struct.unpack('<II', payload[tlv_ptr : tlv_ptr+8])
            t_data = payload[tlv_ptr+8 : tlv_ptr+8+t_len]
            
            if t_type == 1: # Point Cloud
                num_points = t_len // 16
                points = [struct.unpack('<4f', t_data[i*16 : i*16+16])[:3] for i in range(num_points)]
                header = Header(frame_id="radar", stamp=self.get_clock().now().to_msg())
                self.pc_pub.publish(pc2.create_cloud_xyz32(header, points))
            
            elif t_type == 4: # Phase Data
                raw = np.frombuffer(t_data, dtype=np.int16)
                phases = np.arctan2(raw[0::2], raw[1::2])
                self.phase_pub.publish(Float32MultiArray(data=phases.tolist()))

                # self.process_heart_rate(phases[np.argmax(np.sqrt(raw[1::2]**2 + raw[0::2]**2))])
                best_phase = phases[np.argmax(np.sqrt(raw[1::2]**2 + raw[0::2]**2))]
                self.process_heart_rate(best_phase)
                self.process_respiration_rate(best_phase)

            tlv_ptr += (8 + t_len)
        self.buffer = self.buffer[idx+total_len:]

def main(args=None):
    rclpy.init(args=args)
    node = AWR1642VitalSignsNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
