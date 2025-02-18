import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
import time 
from datetime import datetime
import os


class RealSensePublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher')

        # ------------ realsense準備 -------------
        self.bridge = CvBridge()
        self.color_publisher = self.create_publisher(Image, '/camera/color', 10)
        self.depth_publisher = self.create_publisher(Image, '/camera/depth', 10)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.frame_count=0
        self.last_print_fps=0
        try:
            self.pipeline.start(self.config)
            self.get_logger().info("RealSense pipeline started successfully.")
        except RuntimeError as e:
            self.get_logger().error(f"Failed to start RealSense pipeline: {e}")
            return
        self.timer = self.create_timer(0.03, self.publish_frames) 
        # self.timer = self.create_timer(0.1, self.publish_frames) 

        # ------------ 影片初始化 -------------
        output_dir = "output_videos"
        ori_output_dir = "original_videos"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(ori_output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        video_filename = os.path.join(output_dir, f"depth_video_{timestamp}.avi")
        original_video_filename = os.path.join(ori_output_dir, f"original_video_{timestamp}.avi")

        frame_width, frame_height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        self.fps =  30

        self.video_writer = cv2.VideoWriter(video_filename, fourcc, float(self.fps), (frame_width, frame_height))
        self.out_original = cv2.VideoWriter(original_video_filename, fourcc, float(self.fps), (frame_width, frame_height))

        self.get_logger().info(f"Video will be saved to {video_filename}")
        self.get_logger().info(f"Video will be saved to {original_video_filename}")


    def video_save(self,depth_image,color_image):

        # if depth_frame is None or depth_frame.size == 0:
        #     self.get_logger().warn("Depth image is empty or None, skipping video save.")
        #     return
        
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        # 寫入影片
        self.video_writer.write(depth_colormap)
        self.out_original.write(color_image)

        # 顯示深度影像
        # cv2.imshow("Depth Stream", depth_colormap)

    def publish_frames(self):        
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        ## fps
        if time.time() - self.last_print_fps >= 1.0:
            self.get_logger().warn(f"Actual FPS: {self.frame_count}")
            self.last_print_fps = time.time()
            self.frame_count = 0
        self.frame_count += 1
        
        if not color_frame or not depth_frame:
            self.get_logger().warn("Failed to retrieve frames from RealSense camera.")
            return
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        self.video_save(depth_image,color_image)
        
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