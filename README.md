**RealSense 鏡頭影像發佈節點**
本專案提供一個 ROS2 節點，負責從 Intel RealSense 鏡頭擷取彩色影像與深度影像，並將其發佈至 ROS 主題，以供其他模組使用。
-------------------------
1.功能如下：
從 RealSense 鏡頭擷取 RGB 影像與深度影像
發佈影像至 ROS2 topic /camera/color（彩色影像）與 /camera/depth（深度影像）
自動儲存錄製的影像影片（彩色與深度影像）
記錄並顯示 FPS（每秒幀數）
處理 RealSense 鏡頭的初始化與錯誤管理

2.環境需求如下：
pyrealsense2（RealSense SDK）
opencv-python（影像處理）
cv_bridge（ROS 影像訊息轉換）

### 執行程式
#### 使用 RealSense 相機作為輸入
```bash
ros2 run track bytetrack --realsense
```




**人物追蹤系統（YOLOv8 + ByteTrack + ROS2）**
本專案使用 YOLOv8 物件偵測與 ByteTrack 多目標追蹤演算法，結合 Intel RealSense 深度相機，在 ROS2 環境下進行即時人物追蹤。
-------------------------
1.功能如下：
YOLOv8 目標偵測：偵測畫面中的人員
ByteTrack 目標追蹤：追蹤不同 ID 的目標
ROS2 訂閱與發佈：從 RealSense 讀取影像並發佈追蹤結果 /people_track
深度資訊計算：透過深度相機取得目標物的距離
視覺化顯示：在畫面上標記追蹤的對象與 ID

### 執行程式
#### 開啟bytetrack接收相機資訊
```bash
ros2 run track open_rs
```

![image](https://github.com/user-attachments/assets/429db481-6f6d-4282-8508-e97589ff2f03)

建議先開bytetracky再開open_rs，因為open_rs開啟後即會儲存鏡頭讀取之影片。
