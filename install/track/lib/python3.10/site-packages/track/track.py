import cv2
from ultralytics import YOLO
import sys
import numpy as np
from cv2 import KalmanFilter
import torch
import pyrealsense2 as rs

model = YOLO("datasets/best.pt")
track_history = {}
locked_target = None
coordinates_list = []
predicted_points = []  # 預測點存放
last_known_box = None  # 儲存最後一次的邊界框
last_known_center = None  # 儲存最後一次的中心點
closest_target_id = None


def init_kalman_filter():
    kf = KalmanFilter(4, 2)
    kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    kf.transitionMatrix = np.eye(4, 4, dtype=np.float32)
    kf.processNoiseCov = np.eye(4, 4, dtype=np.float32) * 1e-4
    return kf

def detect_and_track(frame):
    global locked_target, predicted_points

    # frame = cv2.resize(frame, (1280, 720))
    frame = cv2.resize(frame, (960, 540))
    results = model.track(source=frame, conf=0.35, iou=0.45, imgsz=640, persist=True)  # track 模式
    im_array = frame.copy()

    for track in results:
        boxes = track.boxes
        if boxes:
            for box in boxes:
                track_id = box.id
                if track_id is not None:
                    track_id_num = int(track_id.item())
                else:
                    track_id_num = -1  # 預設值

                xyxy = box.xyxy[0]
                center_x = (xyxy[0] + xyxy[2]) / 2
                center_y = (xyxy[1] + xyxy[3]) / 2

                # 儲存軌跡
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((center_x, center_y))

                # 處理鎖定目標
                if locked_target is None:
                    cv2.rectangle(im_array, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 3)
                    cv2.putText(im_array, f"ID: {track_id_num}", (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                elif locked_target == track_id:
                    last_known_box = xyxy
                    last_known_center = (center_x, center_y)

                    # 繪製歷史軌跡
                    for coord in coordinates_list:
                        cv2.circle(im_array, (int(coord[0]), int(coord[1])), radius=3, color=(0, 255, 0), thickness=-1)
                    cv2.rectangle(im_array, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 3)
                    cv2.putText(im_array, f"Locked: ID {track_id_num}", (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    coordinates_list.append((center_x, center_y))
                    if len(coordinates_list) > 20:
                        coordinates_list.pop(0)
                    
                # 繪製預測點
                predicted_next_point = predict(im_array, coordinates_list)
                if predicted_next_point:
                    for point in predicted_next_point:
                        predicted_points.append(point)
                    if len(predicted_points) > 5:
                        del predicted_points[:5]
                    for point in predicted_points:
                        cv2.circle(im_array, (int(point[0]), int(point[1])), radius=5, color=(255, 0, 0), thickness=-1)

    return im_array, results


def reset_tracker():
    global model, track_history, locked_target
    # 重新加載
    model = YOLO("best2.pt")  
    track_history.clear()  # 清空
    locked_target = None 
    print("Tracking state reset: All IDs reinitialized.")

def predict(im_array, coordinates_list):
    predicted_next_point = []
    if len(coordinates_list) > 1:
        last_point = coordinates_list[-1]
        second_last_point = coordinates_list[-2]

        movement_vector = [last_point[0] - second_last_point[0],
                           last_point[1] - second_last_point[1]]
        for i in range (1,5,1):
            next_point= [last_point[0] + movement_vector[0]*i,
                            last_point[1] + movement_vector[1]*i]
            predicted_next_point.append(next_point)
            print(predicted_next_point)

    return predicted_next_point

# 處理目標丟失
def handle_target_loss(im_array, results):
    global locked_target, last_known_box, last_known_center, coordinates_list

    if locked_target is not None:
        # 檢查當前追蹤結果是否包含已鎖定的目標
        target_found = False
        for track in results:
            boxes = track.boxes
            if boxes:
                for box in boxes:
                    track_id = box.id
                    if track_id == locked_target:
                        target_found = True
                        break
            if target_found:
                break

        # 如果當前追蹤結果中未找到鎖定目標，嘗試重新鎖定
        if not target_found:
            print(f"Locked target ID {locked_target} lost, attempting to relock...")

            # 預測目標下一步可能位置
            if len(coordinates_list) >= 2:
                predicted_positions = predict(im_array, coordinates_list)

                if predicted_positions:
                    best_match_id = None
                    best_match_score = float('inf')

                    # 當前追蹤的所有目標
                    for track in results:
                        boxes = track.boxes
                        if boxes:
                            for box in boxes:
                                track_id = box.id
                                box_center_x = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                                box_center_y = (box.xyxy[0][1] + box.xyxy[0][3]) / 2

                                # 計算軌跡相似性
                                if track_id in track_history:
                                    target_trajectory = track_history[track_id][-5:]  # 最近5個點
                                    if len(target_trajectory) > 1:
                                        predicted_positions_np = np.array(predicted_positions)
                                        target_trajectory_np = np.array(target_trajectory)
                                        distances = np.linalg.norm(predicted_positions_np - target_trajectory_np, axis=1)
                                        average_distance = np.mean(distances)

                                        # 更新最佳匹配目標
                                        if average_distance < best_match_score:
                                            best_match_score = average_distance
                                            best_match_id = track_id

                    # 如果找到匹配的目標，更新鎖定目標
                    if best_match_id is not None:
                        locked_target = best_match_id
                        print(f"Relocked target: ID {locked_target}")
                        last_known_box = None  # 清除先前的邊界框
                        last_known_center = None  # 清除先前的中心點
                        coordinates_list = track_history[locked_target][-5:]  # 使用新目標的歷史軌跡
                    else:
                        print("Unable to relock the lost target.")
    return im_array




def main(source="camera", input_video_path=None, output_video_path=None):
    global locked_target
    locked_target = None  # 初始狀態不鎖定任何目標

    if source == "camera":
        cap = cv2.VideoCapture(0)
    elif source == "video" and input_video_path:
        cap = cv2.VideoCapture(input_video_path)
    elif source == "realsense" :
        pipeline = rs.pipeline() #初始
        config = rs.config() 
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # rgb
        pipeline.start(config)  #start
    else:
        print("Invalid source or input path.")
        return
    target_width, target_height = 1280, 720

    if output_video_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))
    else:
        out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame or end of video.")
            break

        # 偵測與追蹤
        im_array, results = detect_and_track(frame)
        im_array = handle_target_loss(im_array, results)
        # 顯示與輸出
        if im_array is not None:
            if out:
                out.write(im_array)
            cv2.imshow('result', im_array)

        key = cv2.waitKey(1) & 0xFF
        # q退出
        if key == ord('q'):
            break
        # r重設
        elif key == ord('r'):
            reset_tracker()
            print("Tracker reset: ID cleared and reinitialized.")
        elif key == ord('l'):
            global closest_target_id
            if locked_target is not None:
                locked_target = None  #unlock
                print("Unlocked")
            else:
                print("====================================")
                center_x, center_y = target_width // 2, target_height // 2
                width, height = target_width // 2, target_height // 2  # 鎖定區塊大小
                center_rect = (center_x - width // 2, center_y - height // 2, width, height)
                cv2.rectangle(im_array, 
                              (int(center_rect[0]), int(center_rect[1])),
                              (int(center_rect[0] + width), int(center_rect[1] + height)), 
                              (0, 255, 255), 3)
                min_dist = float('inf')
                closest_target_id = None

                # 距離
                for track in results:
                    boxes = track.boxes
                    if boxes:
                        for box in boxes:
                            xyxy = box.xyxy[0]
                            box_center_x = (xyxy[0] + xyxy[2]) / 2
                            box_center_y = (xyxy[1] + xyxy[3]) / 2
                            print(box,box_center_x,box_center_y)
                            # 判斷目標在中心匡
                            if (center_rect[0] < box_center_x < center_rect[0] + center_rect[2] and
                                    center_rect[1] < box_center_y < center_rect[1] + center_rect[3]):
                                dist = np.sqrt((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_target_id = box.id

                if closest_target_id is not None:
                    locked_target = closest_target_id  # 鎖定最近的目標
                    print(f"Locked target: ID {locked_target}")
                else:
                    print("No suitable target found to lock.")

    cap.release()
    if out: 
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    source = "camera"  # 預設鏡頭
    input_video_path = None
    output_video_path = None

    if len(sys.argv) > 1:
        source = sys.argv[1]  #camera/video
    if len(sys.argv) > 2:
        input_video_path = sys.argv[2]  # 影片檔案路徑
    if len(sys.argv) > 3:
        output_video_path = sys.argv[3]  # 輸出影片路徑

    main(source, input_video_path, output_video_path) 