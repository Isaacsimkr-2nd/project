import time
import re
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class ResponseNode(Node):
    def __init__(self):
        super().__init__('response_node')
        
        self.subscription = self.create_subscription(
            String,
            '/move_command',
            self.listener_callback,
            10
        )
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.move_flag = False 
        self.start_time = None 
        self.duration = 0.0 
        self.speed = 1.0
        self.linear = 0.0  
        self.angular = 0.0  
        
    def emotion_response_callback(self, future):        
        try:
            response = future.result() 
            self.get_logger().info(f'\nResponse received: \n{response.response}')
        except Exception as e:
            self.get_logger().error(f'\nService call failed: {e}')    

    def listener_callback(self, msg):
        data_str = msg.data
        self.get_logger().info(f"\nReceived message: {data_str}")

        try:
            act_match = re.search(r"Action:\s*(.+)", data_str)
            time_match = re.search(r"Time:\s*(.+)", data_str)

            # 결과값 저장 (없으면 빈 문자열 반환)
            act = act_match.group(1).strip() if act_match else ""
            time = time_match.group(1).strip() if time_match else ""

            self.duration = float(time)
            
            
            if act == "멈춤":
                self.linear = 0.0
                self.angular = 0.0 
            elif act == "전진":
                self.linear = -self.speed
                self.angular = 0.0 
            elif act == '후진':
                self.linear = self.speed
                self.angular = 0.0 
            elif act == '좌회전':
                self.linear = 0.0
                self.angular = -self.speed
            elif act == '우회전':
                self.linear = 0.0
                self.angular = self.speed
            elif act == '좌로돌아':
                self.linear = -self.speed
                self.angular = -self.speed 
            elif act == '우로돌아':
                self.linear = -self.speed
                self.angular = self.speed 
            
                
            self.get_logger().info(f"\nParsed values - linear: {self.linear}, angular: {self.angular}, duration: {self.duration}")

            self.move_flag = True
            self.start_time = self.get_clock().now()
        except Exception as e:
            self.get_logger().error(f"Failed to parse message: {str(e)}")

    def timer_callback(self):
        if self.move_flag:
            current_time = self.get_clock().now()
            elapsed = (current_time - self.start_time).nanoseconds / 1e9  

            if elapsed < self.duration:
                twist_msg = Twist()
                twist_msg.linear.x = self.linear
                twist_msg.angular.z = self.angular
                self.cmd_vel_publisher.publish(twist_msg)
            else:
                self.get_logger().info("\nCommand duration elapsed. Stopping cmd_vel publishing.")
                self.move_flag = False  
        else:
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_vel_publisher.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ResponseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


# ros2 run turtle_llm response_node