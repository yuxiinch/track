import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

# 獲取設備支持的流和分辨率
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

for sensor in device.sensors:
    for profile in sensor.profiles:
        if profile.stream_type() == rs.stream.color:
            print(profile.format(), profile.fps(), profile.as_video_stream_profile().width(),
                  profile.as_video_stream_profile().height())
