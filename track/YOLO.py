#!/usr/bin/env python3
# yolo_infer_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

MODEL_PATH = "./src/track/track/best_new_l.engine"  # 替換為你的模型路徑
# MODEL_PATH = "./src/track/track/best_old_l.pt"

class YoloInferNode(Node):
    def __init__(self):
        super().__init__('track_YOLO')    

        self.bridge = CvBridge()
        self.color_image = None
        self.video_path = None #'/workspace/src/track/track/cameracolorr.avi'
        self.cap = None
        self.pub_detections = self.create_publisher(String, '/track/yolo', 10)
        
        self.model = YOLO(MODEL_PATH, task="detect") #.engine模型
        # self.model = YOLO(MODEL_PATH,task = 'detect').to('cuda') #.pt模型

        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.get_logger().error(f"Cannot open video: {self.video_path}")
                exit(1)
            self.get_logger().info(f"Using video file as input: {self.video_path}")
        else:
            self.create_subscription(Image, '/camera/color', self.color_callback, 10)
            self.get_logger().info("Subscribed to /camera/color")

        self.get_logger().info('YOLOv8 Inference Node started.')
        self.timer = self.create_timer(0.03, self.yolo_callback)


    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def yolo_callback(self):
        if self.cap:  # 從影片檔讀取
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().info("Video ended or cannot read frame.")
                rclpy.shutdown()
                return
        else:  # 從 RealSense 訂閱
            if self.color_image is None:
                return
            frame = self.color_image.copy()

        results = self.model(frame, device=0)
        #=======================================
        # results = self.model.predict(frame, device=0, verbose=False)

        detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else np.array([])
        scores = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else np.array([])

        result_strings = []
        for det, conf in zip(detections, scores):
            x1, y1, x2, y2 = det[:4]
            result_strings.append(f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{conf:.2f}")

        msg = String()
        msg.data = ";".join(result_strings)  # 就算是空字串也沒關係
        self.pub_detections.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloInferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
