import cv2
import rclpy
from rclpy.node import Node
# from geometry_msgs.msg import Tuple
from std_msgs.msg import String,Int32MultiArray
from ultralytics import YOLO
from track.byte_tracker import BYTETracker
from track.basetrack import BaseTrack
import argparse
import torch
import numpy as np
import pyrealsense2 as rs
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

#yolov8l mode

model = YOLO("/home/ros_dev/workspace/track/track/best.pt")
track_xyz = {}
matplotlib.use('TkAgg') 
x1=None
y1=None
x2=None
y2=None

def make_parser():
    parser = argparse.ArgumentParser(description="YOLOv8 and BYTETracker demo")
    parser.add_argument("--video_path", type=str, help="Path to the input video file (optional, use webcam if not provided)")
    parser.add_argument("--realsense", action="store_true", help="Use Intel RealSense camera for input")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="Tracking confidence threshold")
    parser.add_argument("--mam tch_thresh", type=float, default=0.8, help="Matching threshold for BYTETracker")
    parser.add_argument("--track_buffer", type=int, default=30, help="Tracking buffer size")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="Aspect ratio threshold for tracking")
    parser.add_argument("--min_box_area", type=float, default=10, help="Minimum box area for valid tracking")
    parser.add_argument("--mot20", action="store_true", help="Enable MOT20 settings for BYTETracker")
    return parser

class people_track_PublisherNode(Node):
    def __init__(self):
        super().__init__('people_track_publish') 
        self.publisher_ = self.create_publisher(Int32MultiArray, 'people_track', 10)  
        self.timer = self.create_timer(1.0, self.timer_callback)  
        self.get_logger().info('people_track node is up and running!')
        args = make_parser().parse_args()
        yolo_track(args)

    def timer_callback(self):
        if x1 is not None:
            msg = Int32MultiArray()
            msg.data=[x1,y1,x2,y2]
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: X1={x1}, Y1={y1},X2={x2}, Y2={y2}')

def reset_realsense_devices():
    ctx = rs.context()
    for device in ctx.query_devices():
        print(f"Resetting device: {device.get_info(rs.camera_info.name)}")
        device.hardware_reset()

def depth(pipeline, depth_image, track_id,x,y,w,h,window_size=5):
    profile = pipeline.get_active_profile()
    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
    cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy

    x2=x+w
    y2=y+h
    target_center_x = (x + x2) / 2  
    target_center_y = (y + y2) / 2 
    if not (0 <= target_center_x < depth_image.shape[1] and 0 <= target_center_y < depth_image.shape[0]):
        print(f"Track ID {track_id}: Target center out of bounds.")
        return None, None, None, None, None

    # depth_value = depth_image[int(target_center_y), int(target_center_x)] #use center depth

    depth_values = depth_image[int(y):int(y2), int(x):int(x2)]              #use x,y,x2,y2 depth
    valid_depths = depth_values[(depth_values > 0) & (depth_values <= 3000)]
    if len(valid_depths) == 0:
        print(f"Track ID {track_id}: No valid depth in target area.")
        return None, None, None, None, None

    depth_value = np.mean(valid_depths)

    Z = depth_value
    X = (target_center_x - cx) * Z / fx
    Y = (target_center_y - cy) * Z / fy
    # print(X,Y,Z)

    if track_id not in track_xyz:
        track_xyz[track_id] = []
    track_xyz[track_id].append((X, Y, Z))

    if len(track_xyz[track_id]) >= window_size:
        recent_coords = track_xyz[track_id][-window_size:]
        filtered_x = np.mean([coord[0] for coord in recent_coords])
        filtered_y = np.mean([coord[1] for coord in recent_coords])
        smoothed_depth = np.mean([coord[2] for coord in recent_coords])
        # print(recent_coords)
        # print("=========================================")
    else:
        filtered_x = X
        filtered_y = Y
        smoothed_depth = Z

    print(f"Track ID {track_id}: Measured XYZ = ({X:.2f}, {Y:.2f}, {Z:.2f}), "
          f"Filtered XYZ = ({filtered_x:.2f}, {filtered_y:.2f}, {smoothed_depth:.2f})")

    return X, Y, filtered_x, filtered_y, smoothed_depth

class YoloTracker:
    def __init__(self):
        self.visualization_initialized = False
        self.fig, self.ax = plt.subplots() #用 Matplotlib 建立窗口
        
    def visualize(self, raw_x, raw_y, smooth_x, smooth_y, z, locked_id):
        if not self.visualization_initialized:
            self.ax.set_xlim(0, 2000)
            self.ax.set_ylim(-1000, 1000)
            self.ax.set_title("RealSense Relative Position (XYZ)")
            self.ax.set_xlabel("Y (mm)")
            self.ax.set_ylabel("X (mm)")
            self.ax.grid(True)
            self.visualization_initialized = True

        self.ax.clear()
        self.ax.set_xlim(0, 2000)
        self.ax.set_ylim(-1000, 1000)

        if locked_id is not None:
            self.ax.scatter(smooth_y, smooth_x, color='red', label=f"Locked ID {locked_id}", s=150, alpha=0.8)
            self.ax.text(smooth_y, smooth_x, f"ID {locked_id}\nZ: {z:.1f} mm", fontsize=10, ha='left', color='red')

        sensor_positions = {
            0: (0,0)
        }
        for sensor_id, (sx, sy) in sensor_positions.items():
            self.ax.scatter(sy, sx, color='blue', label=f"Sensor {sensor_id}", s=100)
            self.ax.text(sy, sx, f"Sensor {sensor_id}\n({sy:.1f}, {sx:.1f})", fontsize=10, ha='center', va='bottom')
        self.ax.legend()
        plt.pause(0.01)

def yolo_track(args):
    tracker = BYTETracker(args)
    locked_target = None 
    closest_target = None
    yolo_tracker = YoloTracker() 

    if args.realsense:
        reset_realsense_devices()
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
    elif args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)

    while True:
        if args.realsense:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
        else:
            ret, frame = cap.read()
            if not ret:
                break

        results = model(frame)
        detections = results[0].boxes.xyxy
        scores = results[0].boxes.conf
        class_ids = results[0].boxes.cls

        dets = torch.cat(
            (
                detections[:, 0].unsqueeze(1),
                detections[:, 1].unsqueeze(1),
                detections[:, 2].unsqueeze(1),
                detections[:, 3].unsqueeze(1),
                scores.unsqueeze(1),
            ), dim=1).cpu().numpy()

        img_h, img_w = frame.shape[:2]
        info_imgs = (img_h, img_w, 0)
        online_targets = tracker.update(dets, info_imgs, (img_h, img_w))

        for target in online_targets:  
            x, y, w, h = target.tlwh
            track_id = target.track_id
            if locked_target is None:
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if locked_target is not None and track_id == locked_target.track_id:
                measured_x, measured_y, filtered_x, filtered_y, z_person = depth(pipeline, depth_image, track_id,x,y,w,h)
                
                if measured_x is not None and measured_y is not None:
                    yolo_tracker.visualize(measured_x, measured_y, filtered_x, filtered_y, z_person, locked_target.track_id)
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                cv2.putText(frame, f"Locked ID: {track_id}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 显示视频
        cv2.imshow("YOLOv8 Tracking", frame)

        # 处理按键事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 'q' 退出
            break
        elif key == ord('r'):  # 按 'r' 重置
            tracker.tracked_stracks = []
            tracker.lost_stracks = []
            tracker.removed_stracks = []
            BaseTrack._count = 0
            print("Tracker reset.")
        elif key == ord('l'):  # 按 'l' 锁定
            if locked_target is not None:
                locked_target = None
                print("Unlocked.")
            else:
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                width, height = frame.shape[1] // 3, frame.shape[0] // 3
                center_rect = (center_x - width // 2, center_y - height // 2, width, height)
                min_dist = float('inf')
                closest_target = None
                for target in online_targets:
                    x, y, w, h = target.tlwh
                    target_center_x = x + w / 2
                    target_center_y = y + h / 2
                    if (center_rect[0] < target_center_x < center_rect[0] + center_rect[2] and
                            center_rect[1] < target_center_y < center_rect[1] + center_rect[3]):
                        dist = np.sqrt((target_center_x - center_x) ** 2 + (target_center_y - center_y) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_target = target

                if closest_target is not None:
                    locked_target = closest_target
                    print(f"Locked target: ID {locked_target.track_id}")

    if args.realsense:
        pipeline.stop()
    else:
        cap.release()

    cv2.destroyAllWindows()
            
            
def main(args=None):
    rclpy.init(args=args)
    node = people_track_PublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__=="__main__":
    main()

    
# if __name__ == "__main__":
#     args = make_parser().parse_args()
#     yolo_track(args)
