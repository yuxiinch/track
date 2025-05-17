#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import argparse
import os
import time 
from datetime import datetime 
# import onnxruntime as ort

# ByteTrack
from track.byte_tracker import BYTETracker
from track.basetrack import BaseTrack

# YOLOv8 來自 ultralytics
from ultralytics import YOLO

###############################################################################
# 請換成你自己訓練好的 YOLOv8 權重檔
###############################################################################

# MODEL_PATH = "./src/track/track/best_old_l.engine"    #58-60
MODEL_PATH = "/workspace/src/track/track/best_new_l.engine"    #60-63
# MODEL_PATH = "./src/track/track/best_new_m.engine"    #76-80
# MODEL_PATH = "./src/track/track/best_old_l.pt"
# MODEL_PATH = "/home/ros_dev/workspace/track/track/best.pt"
a=0


def make_parser():
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack + ROS2 Demo")
    parser.add_argument("--video_path", type=str, help="Path to input video file (optional, use webcam if not provided)")
    parser.add_argument("--realsense", action="store_true", help="Use Intel RealSense camera for input")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="Tracking confidence threshold")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="Matching threshold for BYTETracker")
    parser.add_argument("--track_buffer", type=int, default=30, help="Tracking buffer size")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="Aspect ratio threshold for tracking")
    parser.add_argument("--min_box_area", type=float, default=10, help="Minimum box area for valid tracking")
    parser.add_argument("--mot20", action="store_true", help="Enable MOT20 settings for BYTETracker")
    return parser


class PeopleTrackNode(Node):
    def __init__(self, args):
        super().__init__('PeopleTrackNode')

        self.status = "UNLOCKED"
        self.locked_target = None
        self.x1, self.y1, self.x2, self.y2 = None, None, None, None
        self.has_detection = False

        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/color', self.color_callback, 10)
        self.color_image = None

        self.pub_annotated_image = self.create_publisher(Image, '/annotated_image', 10)
        self.pub_ocr_results = self.create_publisher(String, '/ocr_results', 10)
        self.get_logger().info('people_track node is up and running!')

        self.model = YOLO(MODEL_PATH)
        # self.model = YOLO(MODEL_PATH,task = 'detect').to('cuda')
        time.sleep(3)

        self.tracker_args = args

        self.yolo_thread = threading.Thread(target=self.yolo_track, daemon=True)
        self.yolo_thread.start()

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def publish_annotated_image(self, frame):
        if frame is not None:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.pub_annotated_image.publish(img_msg)

    def yolo_track(self):
        tracker = BYTETracker(self.tracker_args)
        # ------------ 狀態 -------------
        self.status = "UNLOCKED"  # 文字狀態
        self.corner_points = None
        self.locked_target = None
        self.x1, self.y1, self.x2, self.y2 = None, None, None, None
        self.depth_value = 0  
        self.has_detection = False

        # ------------ ROS 相關 -------------
        self.pub_track_data = self.create_publisher(Int32MultiArray, 'people_track', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.color_image = None
        self.depth_image = None
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/color/yu', self.color_callback, 10)
        self.create_subscription(Image, '/camera/depth', self.depth_callback, 10)
        self.get_logger().info('people_track node is up and running!')

        # ------------ YOLO 模型 -------------
        self.model = YOLO(MODEL_PATH,task = 'detect').to('cuda')
        print(f"running at: {self.model.device}")
        time.sleep(3)

        # 將 ByteTrack 參數也保存下來
        self.tracker_args = args

        # ------------ 副執行緒 -------------
        self.yolo_thread = threading.Thread(target=self.yolo_track, daemon=True)
        self.yolo_thread.start()


    def timer_callback(self):
        msg = Int32MultiArray()

        if not self.has_detection:
            self.locked_target = None
            self.depth_value = 0
            msg.data = [0, 0]
            self.get_logger().info("[timer_callback] => NO DETECTION => state=0")
        else:
            if self.locked_target is None:
                self.status = "UNLOCKED"
                self.depth_value = 0
                msg.data = [0, 0]
                self.get_logger().info("[timer_callback] => DETECT but UNLOCKED => state=0")
            else:
                self.status = "LOCKED"
                self.center_x = (self.x1 + self.x2)/2
                msg.data = [int(self.center_x), int(self.depth_value)]
                self.get_logger().info( f"[timer_callback] => LOCKED => state=1")

        self.pub_track_data.publish(msg)

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')

    def get_center_depth(self, depth_image, x1, y1, x2, y2):
        if depth_image is None:
            return 0
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if (cx < 0 or cy < 0 or cy >= depth_image.shape[0] or cx >= depth_image.shape[1]):
            return 0

        dval = depth_image[cy, cx]  
        if dval == 0 or dval > 5000: 
            return 0
        return int(dval) 

    def reset_realsense_devices(self):
        ctx = rs.context()
        for device in ctx.query_devices():
            self.get_logger().info(f"Resetting device: {device.get_info(rs.camera_info.name)}")
            device.hardware_reset()

    def yolo_track(self):
        tracker = BYTETracker(self.tracker_args)
        msg = Int32MultiArray()

        use_realsense = False
        if self.tracker_args.realsense:
            use_realsense = True
        elif self.tracker_args.video_path:
            cap = cv2.VideoCapture(self.tracker_args.video_path)
        else:
            cap = cv2.VideoCapture(0)
        # fps counter
        last_print_fps = 0
        frame_count = 0

        while rclpy.ok():
            if self.color_image is None:
                continue
            if not rclpy.ok():
                break
            ## fps
            if time.time() - last_print_fps >= 1.0:
                self.get_logger().warn(f"Actual FPS: {frame_count}")
                last_print_fps = time.time()
                a = frame_count
                frame_count = 0
            frame_count += 1


            frame = self.color_image.copy()
            results = self.model(frame, verbose=True, device=0)
            detections = results[0].boxes.xyxy
            scores = results[0].boxes.conf
            frame_count = 0
            frame_count += 1
            ## camera
            start_time = time.time()
            if use_realsense:
                if self.color_image is None:
                    continue
                frame = self.color_image.copy()
                depth_image = self.depth_image.copy() if self.depth_image is not None else None

            else:
                ret, frame = cap.read()
                if not ret:
                    self.get_logger().info("Video ended or cannot grab frame.")
                    self.has_detection = False
                    break
                depth_image = None
            self.get_logger().info(f"camera time: {int((time.time() - start_time)*1000)}")
            ## inference
            start_time = time.time()
            results = self.model(frame)
            detections = results[0].boxes.xyxy
            scores = results[0].boxes.conf
            self.get_logger().info(f"inference time: {int((time.time() - start_time)*1000)}")
            ## post process
            start_time = time.time()
            if len(detections) == 0:
                self.has_detection = False
                cv2.imshow("YOLOv8 Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            else:
                self.has_detection = True

            dets = torch.cat([
                detections[:, 0].unsqueeze(1),
                detections[:, 1].unsqueeze(1),
                detections[:, 2].unsqueeze(1),
                detections[:, 3].unsqueeze(1),
                scores.unsqueeze(1),
            ], dim=1).cpu().numpy()
            self.get_logger().info(f"post process time: {int((time.time() - start_time)*1000)}")

            img_h, img_w = frame.shape[:2]
            info_imgs = (img_h, img_w, 0)
            online_targets = tracker.update(dets, info_imgs, (img_h, img_w))

            if len(online_targets) == 0:
                self.has_detection = False
                cv2.imshow("YOLOv8 Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            self.has_detection = True

            for target in online_targets:
                x, y, w, h = target.tlwh
                track_id = target.track_id
                ix, iy, iw, ih = int(x), int(y), int(w), int(h)

                if self.locked_target is None:
                    cv2.rectangle(frame, (ix, iy), (ix + iw, iy + ih), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (ix, iy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    if self.locked_target.track_id == track_id:
                        self.x1, self.y1 = ix, iy
                        self.x2, self.y2 = ix + iw, iy + ih

                        cx = int((self.x1 + self.x2) / 2)
                        cy = int((self.y1 + self.y2) / 2)

                        output_msg = String()
                        output_msg.data = f"person,{cx},{cy}"
                        self.pub_ocr_results.publish(output_msg)
                        if use_realsense and depth_image is not None:
                            depth_val = self.get_center_depth(depth_image, self.x1, self.y1, self.x2, self.y2)
                            self.depth_value = depth_val
                        else:
                            self.depth_value = 0

                        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Locked ID: {track_id}", (self.x1, self.y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, f"fps: {a}", (self.x1, self.y2 - 20) 
                                    ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        self.publish_annotated_image(frame)
                        cv2.putText(frame, f"Depth: {self.depth_value}", (self.x1, self.y2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow("YOLOv8 Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("[INFO] Press 'q' => Break main loop.")
                break
            elif key == ord('r'):
                tracker.tracked_stracks.clear()
                tracker.lost_stracks.clear()
                tracker.removed_stracks.clear()
                BaseTrack._count = 0
                self.locked_target = None
                self.x1 = self.y1 = self.x2 = self.y2 = None
                self.corner_points = None
                self.depth_value = 0
                self.get_logger().info("[INFO] Tracker reset -> UNLOCKED")
            elif key == ord('l'):
                if self.locked_target is not None:
                    self.get_logger().info(f"[INFO] Unlocked target: ID={self.locked_target.track_id}")
                    self.locked_target = None
                    self.x1 = self.y1 = self.x2 = self.y2 = None
                    self.corner_points = None
                    self.depth_value = 0
                else:
                    center_x = frame.shape[1] // 2
                    center_y = frame.shape[0] // 2
                    min_dist = float('inf')
                    closest_target = None
                    for t in online_targets:
                        tx, ty, tw, th = t.tlwh
                        tx_center = tx + tw / 2
                        ty_center = ty + th / 2
                        dist = np.sqrt((tx_center - center_x)**2 + (ty_center - center_y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_target = t

                    if closest_target is not None:
                        self.locked_target = closest_target
                        ix, iy, iw, ih = map(int, closest_target.tlwh)
                        self.x1, self.y1 = ix, iy
                        self.x2, self.y2 = ix + iw, iy + ih
                        self.get_logger().info(f"[INFO] Locked target: ID={closest_target.track_id}")

        cv2.destroyAllWindows()
        self.get_logger().info("Tracking finished.")
        self.depth_value = 0
        self.get_logger().info(f"[INFO] Locked target: ID={closest_target.track_id}")

        if use_realsense:
            if self.subscription is not None:
                self.destroy_subscription(self.color_callback)
                self.destroy_subscription(self.depth_callback)
                self.get_logger().info("RealSense subscription stopped.")
        else:
            cap.release()

        self.video_writer.release()
        cv2.destroyAllWindows()
        self.get_logger().info("Video saved successfully.")

def main():
    args = make_parser().parse_args()
    rclpy.init()
    node = PeopleTrackNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
