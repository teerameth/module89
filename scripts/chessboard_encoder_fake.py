#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import time, math

class FakeEncoderPublisher(Node):
    def __init__(self):
        super().__init__('encoder_publisher_fake')
        self.publisher = self.create_publisher(Float32, '/chessboard/encoder', 10)
        timer_period = 1 / 30  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.rot = Float32()
        self.rot.data = 0.0
        self.last_time = time.time()
    def timer_callback(self):
        now = time.time()
        delta = now - self.last_time
        self.last_time = now
        self.rot.data += delta
        while self.rot.data >= math.pi: self.rot.data -= math.pi
        while self.rot.data < 0: self.rot.data += math.pi
        print(self.rot)
        self.publisher.publish(self.rot)
        # self.get_logger().info(str(self.rot.data))
def main():
    rclpy.init()
    fake_encoder_pub = FakeEncoderPublisher()
    rclpy.spin(fake_encoder_pub)

    fake_encoder_pub.destroy_node()
    rclpy.shutdown()
if __name__ == "__main__":
    main()