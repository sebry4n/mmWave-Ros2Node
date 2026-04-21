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
from scipy.signal import butter, lfilter, find_peaks

class StableAWR1642Node(Node):
    def __init__(self):
        super().__init__('stable_awr1642_node')
        
        # ROS 2 Publishers
        self.pc_pub = self.create_publisher(PointCloud2, 'radar/points', 10)
        self.phase_pub = self.create_publisher(Float32MultiArray, 'radar/phase_data', 10)
        self.hr_pub = self.create_publisher(Float32, 'radar/heart_rate', 10)
        
        # Configuration & Serial
        self.send_config()
        self.ser_data = serial.Serial('/dev/ttyUSB1', 921600, timeout=0.1)
        self.magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        self.buffer = b''
        
        # Signal Processing Constants
        self.fs = 10.0  # Sampling Rate (10Hz based on 100ms frameCfg)
        self.window_size = 128  # Use a power of 2 for faster FFT
        self.phase_history = []
        self.last_bpm = 0.0
        
        self.timer = self.create_timer(0.01, self.main_loop)
        self.get_logger().info("Stable Vital Signs Node Initialized.")

    def send_config(self):
        cfg = [
            "sensorStop", "flushCfg", "dfeDataOutputMode 1", "channelCfg 15 3 0",
            "adcCfg 2 1", "adcbufCfg -1 0 0 1 1",
            "profileCfg 0 77 7 7 58.0 0 0 67.978 1 256 5020 0 0 36",
            "chirpCfg 0 0 0 0 0 0 0 1", "chirpCfg 1 1 0 0 0 0 0 2",
            "frameCfg 0 1 32 0 100 1 0", "lowPower 0 1", 
            "guiMonitor 1 1 0 1", 
            "cfarCfg 1 4 12 4 2 8 2 350 30 2 0 5 20", "dbscanCfg 4 4 13 20 3 256",
            "sensorStart"
        ]
        with serial.Serial('/dev/ttyUSB0', 115200, timeout=1) as ser:
            for line in cfg:
                ser.write((line + '\n').encode()); time.sleep(0.1)

    def butter_bandpass(self, lowcut, highcut, order=4):
        nyq = 0.5 * self.fs
        return butter(order, [lowcut/nyq, highcut/nyq], btype='band')

    def estimate_heart_rate(self, raw_phase):
        # 1. Phase Unwrapping (Stability Fix)
        if not self.phase_history:
            self.phase_history.append(raw_phase)
            return
        
        prev_phase = self.phase_history[-1]
        unwrapped = prev_phase + math.remainder(raw_phase - prev_phase, 2 * math.pi)
        self.phase_history.append(unwrapped)

        if len(self.phase_history) > self.window_size:
            self.phase_history.pop(0)

        # 2. Process only when window is full
        if len(self.phase_history) == self.window_size:
            signal = np.array(self.phase_history)
            
            # Detrend (remove slow drift)
            signal = signal - np.mean(signal)
            
            # Bandpass Filter (0.8Hz to 2.5Hz for Heart Rate)
            b, a = self.butter_bandpass(0.8, 2.5)
            filtered = lfilter(b, a, signal)
            
            # 3. Apply Hanning Window (Stability Fix)
            windowed = filtered * np.hanning(len(filtered))
            
            # 4. FFT
            fft_data = np.abs(np.fft.rfft(windowed))
            freqs = np.fft.rfftfreq(len(windowed), 1/self.fs)
            
            # Find peak within the human heart rate range
            max_idx = np.argmax(fft_res)
            raw_bpm = freqs[max_idx] * 60.0
            
            # 5. Temporal Smoothing (Exponential filter)
            # This prevents the BPM from jumping 20 points in one frame
            alpha = 0.2 
            stable_bpm = (alpha * raw_bpm) + ((1 - alpha) * self.last_bpm)
            self.last_bpm = stable_bpm
            
            msg = Float32()
            msg.data = float(stable_bpm)
            self.hr_pub.publish(msg)

    def main_loop(self):
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

                for _ in range(num_tlvs):
                    t_type, t_len = struct.unpack('<II', payload[tlv_ptr : tlv_ptr+8])
                    t_data = payload[tlv_ptr+8 : tlv_ptr+8+t_len]
                    
                    if t_type == 4: # Heatmap
                        num_samples = len(t_data) // 4
                        phases = []; magnitudes = []
                        for i in range(num_samples):
                            imag, real = struct.unpack('<hh', t_data[i*4 : i*4+4])
                            phases.append(math.atan2(imag, real))
                            magnitudes.append(math.sqrt(real**2 + imag**2))
                        
                        if magnitudes:
                            # Use the strongest bin (the person's chest)
                            best_bin = np.argmax(magnitudes)
                            self.estimate_heart_rate(phases[best_bin])
                    
                    tlv_ptr += (8 + t_len)
                self.buffer = self.buffer[idx+total_len:]

def main(args=None):
    rclpy.init(args=args)
    node = StableAWR1642Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.ser_data.close(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
