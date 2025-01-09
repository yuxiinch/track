import pyrealsense2 as rs
import numpy as np
import cv2

def test_realsense():
    try:
        # 建立 RealSense 管道
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        print("Starting RealSense pipeline...")
        pipeline.start(config)
        print("Pipeline started successfully!")

        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("Waiting for frames...")
                continue

            # 將幀轉換為 NumPy 陣列
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 將深度影像進行歸一化處理以便顯示
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # 拼接彩色和深度影像
            images = np.hstack((color_image, depth_colormap))

            # 顯示影像
            cv2.imshow("RealSense Color and Depth", images)

            # 按 'q' 鍵退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # 停止管道並關閉視窗
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped and windows closed.")
def main():
    test_realsense()

if __name__ == "__main__":
    main()
