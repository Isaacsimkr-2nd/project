#!/usr/bin/env python3
"""
ROS2 Humble용 turtle_agent 도구 모음
이 스크립트는 turtlesim 환경에서 거북이를 생성, 이동, 삭제, 위치 조회, 펜 설정 등 다양한 작업을 수행합니다.
각 함수에 @tool 데코레이터가 붙어 있으므로 LangChain과 같은 에이전트 프레임워크에서 도구로 사용할 수 있습니다.
"""

from math import cos, sin, sqrt, radians
from typing import List
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from langchain.agents import tool
from std_srvs.srv import Empty

from turtlesim.msg import Pose
from turtlesim.srv import Spawn, TeleportAbsolute, TeleportRelative, Kill, SetPen
import threading
# 전역 변수: 각 거북이의 cmd_vel 퍼블리셔들을 저장하는 딕셔너리
cmd_vel_pubs = {}

# 전역 노드 생성 (한 번 생성하고 재사용)
def get_global_node():
    global _global_node
    try:
        # 만약 이미 생성되어 있고, 아직 유효하면 반환
        if _global_node is not None:
            return _global_node
    except NameError:
        pass
    if not rclpy.ok():
        rclpy.init(args=None)
    _global_node = rclpy.create_node('turtle_agent_tools')
    return _global_node

def spin_node(node):
    rclpy.spin(node)

global_node = get_global_node()

# 백그라운드 스레드에서 global_node 스핀 시작
spin_thread = threading.Thread(target=spin_node, args=(global_node,), daemon=True)
spin_thread.start()

def wait_for_message(topic: str, msg_type, timeout: float = 5.0):
    """
    지정한 토픽에서 msg_type 타입의 메시지를 기다립니다.
    첫 번째 메시지가 수신되면 반환하며, 시간 초과 시 TimeoutError를 발생시킵니다.
    """
    msg_container = []

    def callback(msg):
        msg_container.append(msg)

    subscription = global_node.create_subscription(msg_type, topic, callback, 10)
    start_time = time.time()
    while time.time() - start_time < timeout:
        rclpy.spin_once(global_node, timeout_sec=0.1)
        if msg_container:
            subscription.destroy()
            return msg_container[0]
    subscription.destroy()
    raise TimeoutError(f"Timeout waiting for message on topic {topic}")

def add_cmd_vel_pub(name: str, publisher):
    """
    [툴 함수] 주어진 이름의 cmd_vel 퍼블리셔를 등록합니다.
    """
    global cmd_vel_pubs
    cmd_vel_pubs[name] = publisher

def remove_cmd_vel_pub(name: str):
    """
    [툴 함수] 등록된 cmd_vel 퍼블리셔를 제거합니다.
    """
    global cmd_vel_pubs
    cmd_vel_pubs.pop(name, None)

# 기본 turtle1 퍼블리셔 등록
add_cmd_vel_pub("turtle1", global_node.create_publisher(Twist, "/turtle1/cmd_vel", 10))

def within_bounds(x: float, y: float) -> tuple:
    """
    [툴 함수] 주어진 (x, y) 좌표가 turtlesim 환경 범위(0~11, 0~11) 내에 있는지 확인합니다.
    """
    if 0 <= x <= 11 and 0 <= y <= 11:
        return True, "Coordinates are within bounds."
    else:
        return False, f"({x}, {y}) is out of bounds. Valid range is [0, 11] for both."

def will_be_within_bounds(name: str, velocity: float, lateral: float, angle: float, duration: float = 1.0) -> tuple:
    """
    [툴 함수] 주어진 Twist 명령 후, 거북이가 환경 범위 내에 있게 되는지 예측합니다.
    """
    try:
        pose = get_turtle_pose.invoke({"names": [name]})
    except Exception as e:
        return False, f"Pose lookup failed: {e}"
    if name not in pose:
        return False, f"Pose lookup failed for {name}."
    current_x = pose[name].x
    current_y = pose[name].y
    current_theta = pose[name].theta

    if abs(angle) < 1e-6:  # 직선 이동
        new_x = current_x + (velocity * cos(current_theta) - lateral * sin(current_theta)) * duration
        new_y = current_y + (velocity * sin(current_theta) + lateral * cos(current_theta)) * duration
    else:  # 원형 경로 이동
        radius = sqrt(velocity**2 + lateral**2) / abs(angle)
        center_x = current_x - radius * sin(current_theta)
        center_y = current_y + radius * cos(current_theta)
        angle_traveled = angle * duration
        new_x = center_x + radius * sin(current_theta + angle_traveled)
        new_y = center_y - radius * cos(current_theta + angle_traveled)
        for t in range(int(duration)+1):
            angle_t = current_theta + angle * t
            x_t = center_x + radius * sin(angle_t)
            y_t = center_y - radius * cos(angle_t)
            in_bounds, _ = within_bounds(x_t, y_t)
            if not in_bounds:
                return False, f"The circular path goes out of bounds at ({x_t:.2f}, {y_t:.2f})."
    in_bounds, message = within_bounds(new_x, new_y)
    if not in_bounds:
        return False, f"This command moves the turtle out of bounds to ({new_x:.2f}, {new_y:.2f})."
    return True, f"The turtle remains within bounds at ({new_x:.2f}, {new_y:.2f})."

@tool
def spawn_turtle(name: str, x: float, y: float, theta: float) -> str:
    """
    [툴 함수] 지정한 좌표와 각도로 거북이를 생성합니다.
    Spawn a turtle at the given x, y, and theta coordinates.

    :param name: name of the turtle.
    :param x: x-coordinate.
    :param y: y-coordinate.
    :param theta: angle.
    """
    in_bounds, message = within_bounds(x, y)
    if not in_bounds:
        return message
    name = name.replace("/", "")
    try:
        # 먼저, spawn 클라이언트를 생성하고 wait_for_service()를 클라이언트에서 호출합니다.
        spawn_client = global_node.create_client(Spawn, "/spawn")
        if not spawn_client.wait_for_service(timeout_sec=5.0):
            return f"Failed to spawn {name}: /spawn service not available."
    except Exception as e:
        return f"Failed to spawn {name}: {e}"
    try:
        req = Spawn.Request()
        req.x = x
        req.y = y
        req.theta = theta
        req.name = name
        future = spawn_client.call_async(req)
        rclpy.spin_until_future_complete(global_node, future, timeout_sec=5.0)
        if future.result() is None:
            return f"Failed to spawn {name}: service call failed."
        # 등록: 새로운 거북이의 cmd_vel 퍼블리셔 생성
        add_cmd_vel_pub(name, global_node.create_publisher(Twist, f"/{name}/cmd_vel", 10))
        return f"{name} spawned at x: {x}, y: {y}, theta: {theta}."
    except Exception as e:
        return f"Failed to spawn {name}: {e}"


@tool
def kill_turtle(names: List[str]) -> str:
    """
    [툴 함수] 지정한 이름의 거북이를 turtlesim 환경에서 제거합니다.
    Removes a turtle from the turtlesim environment.

    :param names: List of names of the turtles to remove (do not include the forward slash).
    """
    names = [n.replace("/", "") for n in names]
    response = ""
    global cmd_vel_pubs
    for name in names:
        try:
            global_node.wait_for_service(f"/{name}/kill", timeout_sec=5.0)
        except Exception:
            response += f"Failed to kill {name}: /{name}/kill service not available.\n"
            continue
        try:
            kill_client = global_node.create_client(Kill, f"/{name}/kill")
            req = Kill.Request()
            future = kill_client.call_async(req)
            rclpy.spin_until_future_complete(global_node, future, timeout_sec=5.0)
            if future.result() is None:
                response += f"Failed to kill {name}: service call failed.\n"
            else:
                cmd_vel_pubs.pop(name, None)
                response += f"Successfully killed {name}.\n"
        except Exception as e:
            response += f"Failed to kill {name}: {e}\n"
    return response

@tool
def clear_turtlesim() -> str:
    """
    [툴 함수] turtlesim 배경을 지웁니다.
    Clears the turtlesim background and sets the color to the value of the background parameters.
    """
    try:
        clear_client = global_node.create_client(Empty, "/clear")
        if not clear_client.wait_for_service(timeout_sec=5.0):
            return "Failed to clear turtlesim background: /clear service not available."
    except Exception as e:
        return f"Failed to clear turtlesim background: {e}"
    try:
        req = Empty.Request()
        future = clear_client.call_async(req)
        rclpy.spin_until_future_complete(global_node, future, timeout_sec=5.0)
        if future.result() is None:
            return "Failed to clear turtlesim background: service call failed."
        return "Successfully cleared the turtlesim background."
    except Exception as e:
        return f"Failed to clear turtlesim background: {e}"


@tool
def get_turtle_pose(names: List[str]) -> dict:
    """
    [툴 함수] 지정한 거북이들의 위치(Pose)를 조회합니다.
    만약 노드가 이미 소멸(destruction)된 상태라면 재초기화합니다.
    Get the pose of one or more turtles.

    :param names: List of names of the turtles to get the pose of.
    """
    poses = {}
    names = [n.replace("/", "") for n in names]
    for name in names:
        topic = f"/{name}/pose"
        try:
            msg = wait_for_message(topic, Pose, timeout=5.0)
        except Exception as e:
            if "cannot use Destroyable" in str(e):
                # 노드가 소멸 요청되었으므로 재초기화
                global global_node
                try:
                    global_node.destroy_node()
                except Exception:
                    pass
                rclpy.shutdown()
                rclpy.init(args=None)
                global_node = rclpy.create_node('turtle_agent_tools')
                add_cmd_vel_pub("turtle1", global_node.create_publisher(Twist, "/turtle1/cmd_vel", 10))
                try:
                    msg = wait_for_message(topic, Pose, timeout=5.0)
                except Exception as e2:
                    raise Exception(f"Failed to get pose for {name} after reinitialization: {e2}")
            else:
                raise Exception(f"Failed to get pose for {name}: {e}")
        poses[name] = msg
    return poses

# @tool
# def teleport_absolute(name: str, x: float, y: float, theta: float, hide_pen: bool = True) -> str:
#     """
#     [툴 함수] 거북이를 절대 좌표 (x, y, theta)로 순간 이동시킵니다.
#     hide_pen이 True이면 이동 흔적을 남기지 않습니다.
#     """
#     in_bounds, message = within_bounds(x, y)
#     if not in_bounds:
#         return message

#     # 클라이언트 객체 생성 및 서비스 사용 가능 여부 확인
#     try:
#         teleport_client = global_node.create_client(TeleportAbsolute, f"/{name}/teleport_absolute")
#         if not teleport_client.wait_for_service(timeout_sec=5.0):
#             return f"Failed to teleport {name}: /{name}/teleport_absolute service not available."
#     except Exception as e:
#         return f"Failed to teleport {name}: {e}"

#     try:
#         req = TeleportAbsolute.Request()
#         req.x = x
#         req.y = y
#         req.theta = theta

#         if hide_pen:
#             # 펜 숨김: 이동 전 흔적 제거 (set_pen 도구는 별도로 정의되어 있어야 함)
#             set_pen.invoke({"name": name, "r": 0, "g": 0, "b": 0, "width": 1, "off": 1})
            
#         future = teleport_client.call_async(req)
#         rclpy.spin_until_future_complete(global_node, future, timeout_sec=5.0)
#         if future.result() is None:
#             return f"Failed to teleport {name}: service call failed."

#         if hide_pen:
#             # 이동 후 펜 복원
#             set_pen.invoke({"name": name, "r": 30, "g": 30, "b": 255, "width": 1, "off": 0})

#         try:
#             current_pose = get_turtle_pose.invoke({"names": [name]})
#         except Exception as e:
#             return f"{name} pose lookup failed: {e}"
#         if name not in current_pose:
#             return f"Pose lookup failed for {name}."
#         pose = current_pose.get(name)
#         return f"{name} new pose: (x={pose.x}, y={pose.y}) at {pose.theta} radians."
#     except Exception as e:
#         return f"Failed to teleport the turtle: {e}"


@tool
def teleport_relative(name: str, linear: float, angular: float) -> str:
    """
    [툴 함수] 거북이를 현재 위치 기준으로 상대 이동시킵니다.
    Teleport a turtle relative to its current position.

    :param name: name of the turtle
    :param linear: linear distance
    :param angular: angular distance
    """

    in_bounds, message = will_be_within_bounds(name, linear, 0.0, angular)
    if not in_bounds:
        return message

    try:
        teleport_client = global_node.create_client(TeleportRelative, f"/{name}/teleport_relative")
        if not teleport_client.wait_for_service(timeout_sec=5.0):
            return f"Failed to teleport {name} relatively: /{name}/teleport_relative service not available."
    except Exception as e:
        return f"Failed to create client for teleport_relative: {e}"
    
    # 서비스 호출
    req = TeleportRelative.Request()
    req.linear = linear
    req.angular = angular
    future = teleport_client.call_async(req)

    
    # 이동 후 거북이의 위치를 조회합니다.
    try:
        current_pose = get_turtle_pose.invoke({"names": [name]})
    except Exception as e:
        return f"{name} pose lookup failed: {e}"
    if name not in current_pose:
        return f"Pose lookup failed for {name}."
    pose = current_pose.get(name)
    return f"{name} new pose: (x={pose.x}, y={pose.y}) at {pose.theta} radians."

    


@tool
def publish_twist_to_cmd_vel(name: str, velocity: float, lateral: float, angle: float, steps: int = 1) -> str:
    """
    [툴 함수] 지정한 거북이의 /{name}/cmd_vel 토픽에 Twist 메시지를 발행하여 이동시킵니다.
    Publish a Twist message to the /{name}/cmd_vel topic to move a turtle robot.
    Use a combination of linear and angular velocities to move the turtle in the desired direction.

    :param name: name of the turtle (do not include the forward slash)
    :param velocity: linear velocity, where positive is forward and negative is backward
    :param lateral: lateral velocity, where positive is left and negative is right
    :param angle: angular velocity, where positive is counterclockwise and negative is clockwise
    :param steps: Number of times to publish the twist message
    """
    name = name.replace("/", "")
    in_bounds, message = will_be_within_bounds(name, velocity, lateral, angle, duration=steps)
    if not in_bounds:
        return message
    vel = Twist()
    vel.linear.x = velocity
    vel.linear.y = lateral
    vel.linear.z = 0.0
    vel.angular.x = 0.0
    vel.angular.y = 0.0
    vel.angular.z = angle
    try:
        pub = cmd_vel_pubs[name]
        for _ in range(steps):
            pub.publish(vel)
            time.sleep(1)
    except Exception as e:
        return f"Failed to publish {vel} to /{name}/cmd_vel: {e}"
    try:
        current_pose = get_turtle_pose.invoke({"names": [name]})
    except Exception as e:
        return f"{name} pose lookup failed: {e}"
    if name not in current_pose:
        return f"Pose lookup failed for {name}."
    pose = current_pose.get(name)
    return (f"New Pose ({name}): x={pose.x}, y={pose.y}, theta={pose.theta} rads, "
            f"(linear and angular velocities are not provided by turtlesim).")

@tool
def stop_turtle(name: str) -> str:
    """
    [툴 함수] 거북이를 정지시킵니다.
    Stop a turtle by publishing a Twist message with zero linear and angular velocities.

    :param name: name of the turtle
    """
    return publish_twist_to_cmd_vel.invoke({
        "name": name,
        "velocity": 0.0,
        "lateral": 0.0,
        "angle": 0.0,
        "steps": 1
    })

@tool
def reset_turtlesim() -> str:
    """
    [툴 함수] turtlesim 환경을 초기화합니다.
    Resets the turtlesim, removes all turtles, clears any markings, and creates a new default turtle at the center.
    """
    try:
        reset_client = global_node.create_client(Empty, "/reset")
        if not reset_client.wait_for_service(timeout_sec=5.0):
            return "Failed to reset turtlesim: /reset service not available."
    except Exception as e:
        return f"Failed to reset turtlesim: {e}"
    try:
        req = Empty.Request()
        future = reset_client.call_async(req)
        rclpy.spin_until_future_complete(global_node, future, timeout_sec=5.0)
        if future.result() is None:
            return "Failed to reset turtlesim: service call failed."
        global cmd_vel_pubs
        cmd_vel_pubs.clear()
        add_cmd_vel_pub("turtle1", global_node.create_publisher(Twist, "/turtle1/cmd_vel", 10))
        return "Successfully reset turtlesim. All previous commands, failures, and goals are cleared."
    except Exception as e:
        return f"Failed to reset turtlesim: {e}"


@tool
def set_pen(name: str, r: int, g: int, b: int, width: int, off: int) -> str:
    """
    [툴 함수] 거북이의 펜 색상, 두께, 상태를 설정합니다.
    Set the pen color and width for the turtle. The pen is used to draw lines on the turtlesim canvas.

    :param name: name of the turtle
    :param r: red value
    :param g: green value
    :param b: blue value
    :param width: width of the pen.
    :param off: 0=on, 1=off
    """
    name = name.replace("/", "")
    try:
        # 클라이언트 객체를 생성하고, 해당 클라이언트에서 wait_for_service()를 호출합니다.
        set_pen_client = global_node.create_client(SetPen, f"/{name}/set_pen")
        if not set_pen_client.wait_for_service(timeout_sec=5.0):
            return f"Failed to set pen for {name}: /{name}/set_pen service not available."
    except Exception as e:
        return f"Failed to set pen for {name}: {e}"
    
    try:
        req = SetPen.Request()
        req.r = r
        req.g = g
        req.b = b
        req.width = width
        req.off = off
        future = set_pen_client.call_async(req)
        rclpy.spin_until_future_complete(global_node, future, timeout_sec=5.0)
        if future.result() is None:
            return f"Failed to set pen for {name}: service call failed."
        return f"Successfully set pen for {name}."
    except Exception as e:
        return f"Failed to set pen for {name}: {e}"


@tool
def has_moved_to_expected_coordinates(name: str, expected_x: float, expected_y: float, tolerance: float = 0.1) -> str:
    """
    [툴 함수] 거북이가 예상한 좌표로 이동했는지 확인합니다.
    Check if the turtle has moved to the expected position.

    :param name: name of the turtle
    :param expected_x: expected x-coordinate
    :param expected_y: expected y-coordinate
    :param tolerance: tolerance level for the comparison
    """
    try:
        current_pose = get_turtle_pose.invoke({"names": [name]})
    except Exception as e:
        return f"Pose lookup failed for {name}: {e}"
    if name not in current_pose:
        return f"Pose lookup failed for {name}."
    pose = current_pose.get(name)
    current_x = pose.x
    current_y = pose.y
    distance = sqrt((current_x - expected_x) ** 2 + (current_y - expected_y) ** 2)
    if distance <= tolerance:
        return f"{name} has moved to the expected position ({expected_x}, {expected_y})."
    else:
        return f"{name} has NOT moved to the expected position ({expected_x}, {expected_y})."

# 만약 이 모듈을 직접 실행할 경우, turtle1의 포즈를 출력하고 노드를 종료합니다.
if __name__ == '__main__':
    try:
        pose = get_turtle_pose.invoke({"names": ["turtle1"]})
        print(f"turtle1's pose: {pose['turtle1']}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 여기에서는 노드를 종료하지 않습니다. 에이전트 실행 중에는 노드를 계속 유지해야 합니다.
        pass
