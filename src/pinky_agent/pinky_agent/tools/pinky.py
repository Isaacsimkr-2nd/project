#!/usr/bin/env python3
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from langchain.agents import tool
from ultralytics import YOLO

import cv2 
class PinkyAgent(Node):
    def __init__(self):
        super().__init__('pinky_agent_tools')

        # /cmd_vel 퍼블리셔 생성
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # 이동 관련 변수
        self.move_flag = False
        self.start_time = None
        self.duration = 0.0
        self.linear = 0.0
        self.angular = 0.0

        # 주기적으로 이동 상태 확인 (0.1초마다 실행)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        """
        이동 상태를 주기적으로 확인하여 duration이 지나면 정지
        """
        if self.move_flag:
            current_time = self.get_clock().now()
            elapsed = (current_time - self.start_time).nanoseconds / 1e9  # 초 단위 변환

            if elapsed < self.duration:
                twist_msg = Twist()
                twist_msg.linear.x = -self.linear
                twist_msg.angular.z = self.angular
                self.cmd_vel_pub.publish(twist_msg)
            else:
                self.get_logger().info("이동 시간 초과: 정지")
                self.move_flag = False  # 이동 중지
                self.stop_movement()

    def stop_movement(self):
        """
        즉시 정지 명령을 실행
        """
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        self.get_logger().info("즉시 정지")

    def publish_twist_to_cmd_vel(self, velocity: float, angle: float, duration: float = 1.0) -> str:
        self.linear = velocity
        self.angular = angle
        self.duration = duration
        self.start_time = self.get_clock().now()
        self.move_flag = True  # 이동 시작

        return f"pinky 이동 명령: velocity={velocity}, angle={angle}, duration={duration}s."

    def stop_pinky(self) -> str:

        self.move_flag = False
        self.stop_movement()

        return "pinky 즉시 정지 명령 실행됨."


# 글로벌 인스턴스를 관리하여 LangChain과 연결
pinky_agent = None
ros_thread = None

def get_pinky_agent():
    global pinky_agent, ros_thread

    if pinky_agent is None:
        rclpy.init(args=None)
        pinky_agent = PinkyAgent()

        # ROS 2 노드를 별도 스레드에서 실행하여 LangChain과 동시 실행
        def ros_spin():
            rclpy.spin(pinky_agent)

        ros_thread = threading.Thread(target=ros_spin, daemon=True)
        ros_thread.start()

    return pinky_agent


# LangChain과 연결되는 @tool 함수 (싱글톤 패턴 활용)
# @tool
# def publish_twist_to_cmd_vel(velocity: float, angle: float, duration: int = 1) -> str:
#     """
#     [툴 함수] pinky의 /cmd_vel 토픽에 Twist 메시지를 발행하여 이동시킵니다.
#     Use a combination of linear and angular velocities to move the pinky in the desired direction.
    
#     직전 전진 또는 직선 후진의 경우 angle는 0.0이고 velocity만 조정합니다. 
#     회전인 경우 velocity는 0.0이고 angle만 조정합니다. 
#     왼쪽으로 이동 또는 오른쪽으로 이동과 같이 방향이 정해진 전진 또는 후진 이동인 경우 velocity와 angle을 모두 조정합니다.
    
#     :param velocity: 선속도 (m/s) (양수: 전진, 음수: 후진)
#     :param angle: 각속도 (rad/s) (양수: 반시계 방향 회전, 음수: 시계 방향 회전)
#     :param duration: 이동 지속 시간 (초 단위)
#     """
#     agent = get_pinky_agent()
#     return agent.publish_twist_to_cmd_vel(velocity, angle, duration)


@tool
def forward_or_backward(velocity: float, duration: float = 1.0) -> str:
    """
    [툴 함수] pinky의 /cmd_vel 토픽에 Twist 메시지를 발행하여 전진 또는 후진 합니다.
    직전 전진 또는 직선 후진의 경우 angle는 0.0이고 velocity만 조정합니다. 
    
    :param velocity: 선속도 (m/s) (양수: 전진, 음수: 후진)
    :param duration: 이동 지속 시간 (초 단위)
    """
    agent = get_pinky_agent()
    return agent.publish_twist_to_cmd_vel(velocity, 0.0, duration)

@tool
def rotate_in_place(angle: float = 0.1, duration: float = 0.1) -> str:
    """
    [툴 함수] pinky의 /cmd_vel 토픽에 Twist 메시지를 발행하여 제자리 회전 합니다.
    회전인 경우 velocity는 0.0이고 angle만 조정합니다.
    
    
    :param angle: 각속도 (rad/s) (양수: 반시계 방향 회전, 음수: 시계 방향 회전)
    :param duration: 이동 지속 시간 (초 단위)
    """
    agent = get_pinky_agent()
    return agent.publish_twist_to_cmd_vel(0.0, angle, duration)

@tool
def move_with_direction(velocity: float, angle: float, duration: float = 1.0) -> str:
    """
    [툴 함수] pinky의 /cmd_vel 토픽에 Twist 메시지를 발행하여 방향성이 있는 전진 또는 후진 합니다.
    왼쪽으로 이동 또는 오른쪽으로 이동과 같이, '방향성이 있는 이동'인 경우 velocity와 angle을 모두 조정합니다.
    
    '왼쪽으로 이동'하는 경우 velocity와 angle이 모두 양수 입니다. 
    '오른쪽으로 이동'하는 경우 velocity는 양수 angle는 음수 입니다. 
    '~을 향해 이동'할때 사용합니다.
    
    :param velocity: 선속도 (m/s) (양수: 전진, 음수: 후진)
    :param angle: 각속도 (rad/s) (양수: 왼쪽 방향, 음수: 오른쪽 방향)
    :param duration: 이동 지속 시간 (초 단위)
    """
    agent = get_pinky_agent()
    return agent.publish_twist_to_cmd_vel(velocity, angle, duration)


@tool
def stop_pinky() -> str:
    """
    [툴 함수] pinky를 즉시 정지시킵니다.
    """
    agent = get_pinky_agent()
    return agent.stop_pinky()


yolo_model = YOLO('/home/pinky/yolo/yolo11n.pt')

@tool 
def yolo_tool():
    """
    카메라 피드를 기반으로 객체를 감지합니다.
    전방에 무엇이 보이는지 확인하고 싶을 때 해당 도구를 사용하세요.
    """
    cap = cv2.VideoCapture(0)
    for _ in range(5):
        cap.grab()
    ret, frame = cap.read()
    
    if not ret:
        return [{"error": "카메라에서 프레임을 읽지 못했습니다."}]
    # 좌우 반전 
    frame = cv2.flip(frame, 1)
    results = yolo_model(source=frame, conf=0.4, verbose=False)
    info = {}
    # 화면 정보 저장
    screen_width = frame.shape[1] 
    screen_height = frame.shape[0]  
    screen_center_x = screen_width // 2  
    screen_center_y = screen_height // 2 
    
    # 화면을 5등분하는 기준
    left_boundary = screen_width // 3        # 1/3 지점 (왼쪽과 중앙 경계)
    right_boundary = (screen_width // 3) * 2 # 2/3 지점 (중앙과 오른쪽 경계)
    # -> 3/5 지점은 중앙 
    info['screen_info'] = {
        'screen_size': screen_width*screen_width,
        'screen_center': [screen_center_x, screen_center_y]
    }
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())
            label = yolo_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x, y, w, h = map(int, box.xywh[0])
            confidence = round(box.conf[0].item(), 2)
            
            if x < left_boundary:
                position = f"left:{abs(screen_center_x-x)}"
            elif x < right_boundary:
                position = "center"
            else:
                position = f"right:{abs(screen_center_x-x)}"
                
            info[label] = {
                'location': [x, y],
                'size': w * h,
                # 'bbox': [x1, y1, x2, y2],
                # 'confidence': confidence,
                'position': position,
            }
            
    save_path = "./detections.jpg"  # 저장할 이미지 경로
    cv2.imwrite(save_path, frame)
    cap.release()  
    return [info]


@tool
def find_detection(velocity: float, angle: float, duration: float = 1.0) -> str:
    """
    [툴 함수] pinky의 /cmd_vel 토픽에 Twist 메시지를 발행하여 움직입니다. 
    객체를 찾기 위해 pinky는 움직입니다. 
    객체를 찾기 위해 제자리 회전하기도 하고 방향성 있는 이동을 하기도 합니다.  
    선속도 velocity와 각속도 angle를 통해 주변을 살피고 yolo_tool을 사용하여 객체를 찾습니다.
    무엇을 찾기 위해 사용됩니다.
    객체 위치를 참고하여 angle을 정하고 얼마나 이동할지 정하세요.
    """
    agent = get_pinky_agent() 
    detection_result = yolo_tool.invoke({})
    
    return agent.publish_twist_to_cmd_vel(velocity, angle, duration)



# ROS 2 노드를 실행하는 메인 함수
def main(args=None):
    global pinky_agent
    pinky_agent = get_pinky_agent()
    try:
        rclpy.spin(pinky_agent)
    except KeyboardInterrupt:
        pass
    finally:
        pinky_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
