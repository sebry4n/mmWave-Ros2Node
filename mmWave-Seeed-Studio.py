import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import serial

class MMWavePublisher(Node):
    def __init__(self):
        super().__init__('mmwave_publisher')
        self.publisher_ = self.create_publisher(Float32, 'sensor/heart_rate', 10)
        
        # Configure your serial port (Change '/dev/ttyUSB0' to your ESP32 port)
        # In Windows, use 'COM3', etc.
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        
        # Create a timer to check serial buffer every 0.1 seconds
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("mmWave Serial to ROS 2 Bridge Started")

    def timer_callback(self):
        if self.ser.in_waiting > 0:
            try:
                # Read line, decode, and convert to float
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    heart_rate = float(line)
                    
                    msg = Float32()
                    msg.data = heart_rate
                    self.publisher_.publish(msg)
                    
                    self.get_logger().info(f'Publishing Heart Rate: {heart_rate}')
            except ValueError:
                pass # Ignore malformed serial data

def main(args=None):
    rclpy.init(args=args)
    node = MMWavePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
