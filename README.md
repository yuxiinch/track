ROS 2 People Tracking System with YOLOv8, ByteTrack and Depth Mapping
這是一個整合 YOLOv8 物件偵測、ByteTrack 多目標追蹤、RealSense 深度資訊轉換的 ROS 2 專案。適用於人員追蹤與 2D → 3D 位置估計等應用場景。

📦 專案架構
本系統包含三個主要 ROS2 node：
YOLO.py：使用 TensorRT 加速的 YOLOv8 模型執行影像中的人員偵測。
ByteTrack.py：將 YOLO 偵測結果進行多目標追蹤與鎖定功能，可手動鎖定 ID。
depth.py：根據 RealSense 深度影像與追蹤結果估算 3D 座標。

📷 節點說明
YOLOv8 節點（YOLO.py）
訂閱：/camera/color（RGB 影像）
發布：/track/yolo（格式為 x1,y1,x2,y2,conf;...）
支援使用 .engine 模型進行 TensorRT 推論（需指定 MODEL_PATH）。

ByteTrack 節點（ByteTrack.py）
訂閱：
/track/yolo：偵測結果
/camera/color：RGB 影像（與 YOLO 共用）
發布：
/annotated_image：帶有追蹤框的影像
/track/object：PoseArray 格式的人員 2D 中心點資訊
功能：
自動追蹤與手動鎖定 ID（按鍵 l 鎖定，r 重置）

深度估算節點（depth.py）
訂閱：
/camera/depth：RealSense 深度影像
/track/object：2D 平面位置（由 ByteTrack 發布）
發布：
/track/object_3d：3D PoseArray 結果
使用 fx, fy, ppx, ppy 相機參數將 2D 位置對應至 3D 空間座標。



