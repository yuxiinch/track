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

# 初始化 YOLOv8 模型
model = YOLO("/home/ros_dev/workspace/track/track/best.pt")
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

def depth(pipeline,depth_image,track_id):
    profile = pipeline.get_active_profile()
    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
    cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy
    x, y = 320, 240  # 圖像中心點
    depth_value = depth_image[y, x]  # 深度值
    Z = depth_value
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    print(f"({track_id}) 的深度值: {depth_value} 毫米")
    print(f"空間座標 X={X:.2f}, Y={Y:.2f}, Z={Z:.2f} 毫米\n")

    return X,Y,Z,depth_value

def yolo_track(args):
    # 初始化
    tracker = BYTETracker(args)
    locked_target = None 
    closest_target = None
    
    if args.realsense:
        pipeline = rs.pipeline() #初始
        config = rs.config() 
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # rgb
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # rgb
        pipeline.start(config)  #start

    elif args.video_path:
        cap = cv2.VideoCapture(args.video_path)  # 影片

    else:
        cap = cv2.VideoCapture(0)  # webcam

    while True:
        if args.realsense:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame:
                continue
            if not depth_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # YOLOv8
        results = model(frame)
        detections = results[0].boxes.xyxy
        scores = results[0].boxes.conf  # 置信分數
        class_ids = results[0].boxes.cls  # ID

        # bytetrack資料
        dets = torch.cat(
        (
            detections[:, 0].unsqueeze(1),  # x1
            detections[:, 1].unsqueeze(1),  # y1
            detections[:, 2].unsqueeze(1),  # x2
            detections[:, 3].unsqueeze(1),  # y2
            scores.unsqueeze(1),  # [score]
            # class_ids.unsqueeze(1)  # 類別
        ), dim=1).cpu().numpy()

        img_h, img_w = frame.shape[:2]
        info_imgs = (img_h, img_w, 0)  #圖片資訊
        online_targets = tracker.update(dets, info_imgs, (img_h, img_w))

        # 匡匡
        for target in online_targets:
            x, y, w, h = target.tlwh
            track_id = target.track_id
            # 其他匡
            if locked_target is None:
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 鎖定匡
            if locked_target is not None and track_id == locked_target.track_id:
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                cv2.putText(frame, f"Locked ID: {track_id}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"Locked ID :{track_id}, coordinate :{int (x),int (y),int (x+w) ,int(y+h)}")
                x1=x
                y1=y
                x2=x+w
                y2=y+h


        cv2.imshow("YOLOv8 Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        # 按q退出
        if key == ord('q'):
            break
        # 按r重設
        elif key == ord('r'):
            tracker.tracked_stracks = []
            tracker.lost_stracks = []
            tracker.removed_stracks = []
            BaseTrack._count = 0
            print("Tracker reset: ID cleared and reinitialized.")
        # 按l鎖定
        elif key == ord('l'):
            if locked_target is not None:
                locked_target = None  #解除
                print("Unlocked")
            else:
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                width, height = frame.shape[1] // 3, frame.shape[0] // 3  # 鎖定區塊大小
                center_rect = (center_x - width // 2, center_y - height // 2, width, height)  # 中心座標
                min_dist = float('inf')  #最大值
                closest_target = None

                # 距離
                for target in online_targets:
                    x, y, w, h = target.tlwh
                    target_center_x = x + w / 2
                    target_center_y = y + h / 2
                    if (center_rect[0] < target_center_x < center_rect[0] + center_rect[2] and center_rect[1] < target_center_y < center_rect[1] + center_rect[3]):
                        dist = np.sqrt((target_center_x - center_x) ** 2 + (target_center_y - center_y) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_target = target

                if closest_target is not None:
                    locked_target = closest_target  #鎖定id最近 
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
