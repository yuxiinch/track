import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge

class RealSensePublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher')
        self.bridge = CvBridge()
        self.color_publisher = self.create_publisher(Image, '/camera/color', 10)
        self.depth_publisher = self.create_publisher(Image, '/camera/depth', 10)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        try:
            self.pipeline.start(self.config)
            self.get_logger().info("RealSense pipeline started successfully.")
        except RuntimeError as e:
            self.get_logger().error(f"Failed to start RealSense pipeline: {e}")
            return
        self.timer = self.create_timer(0.1, self.publish_frames)  # 10Hz publishing rate

    def publish_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            self.get_logger().warn("Failed to retrieve frames from RealSense camera.")
            return
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
        
        self.color_publisher.publish(color_msg)
        self.depth_publisher.publish(depth_msg)
        self.get_logger().info("Published RealSense color and depth images.")

    def destroy(self):
        self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePublisher()
    rclpy.spin(node)
    node.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()