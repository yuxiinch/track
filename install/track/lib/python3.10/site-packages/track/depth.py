import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('translate_depth')
        self.subscription = self.create_subscription(
            Int32MultiArray,
            '/depth',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Int32MultiArray, '/trans_depth', 10)

    def listener_callback(self, msg:Int32MultiArray):
        msg.data[0] = (msg.data[0] + msg.data[2])//2
        msg.data[1] = (msg.data[1] + msg.data[3])//2

        # if (msg.data[0] < 0  or  msg.data[1] < 0 ):
        #     msg.data[:2] = 
        
        self.get_logger().info(f"center.x = {msg.data[0]}, center.y = {msg.data[1]}")

        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# def get_center_depth(self, depth_image, x1, y1, x2, y2):
#     if depth_image is None:
#         return 0
#     cx = (x1 + x2) // 2
#     cy = (y1 + y2) // 2

#     if (cx < 0 or cy < 0 or cy >= depth_image.shape[0] or cx >= depth_image.shape[1]):
#         return 0

#     dval = depth_image[cy, cx]  
#     if dval == 0 or dval > 5000: 
#         return 0
#     return int(dval) 