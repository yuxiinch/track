import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Int32MultiArray

class track_point(Node):
    def __init__(self):
        super().__init__('track_point')
        self.subscription=self.create_subscription(Int32MultiArray,'people_track',self.listener_callback,10)

    def listener_callback(self,msg):
        self.get_logger().info(f'Receive',(msg.data))
    
def main(args=None):
    rclpy.init(args=args)
    subscrib=track_point()
    rclpy.spin(subscrib)
    subscrib.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
