from langchain.agents import tool
from typing import List, Dict
import cv2 
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
yolo_model = YOLO("./yolo/yolo11n.pt")

# YOLO 도구 정의 (도구 데코레이터 사용)
@tool
def cmd_vel_tool(action: str='멈춤') -> str:
    """
    해당 도구는 로봇 바퀴를 움직일 수 있게하는 도구 함수입니다. 
    로봇을 움직여야 할때 사용하세요. 
    이동 명령을 받아서 로봇을 움직이는 동작을 반환합니다.
    로봇은 "전진", "후진", "좌회전", "우회전", "멈춤"의 동작을 할 수 있습니다.
    이동할 필요가 없는 경우 '멈춤'을 유지하세요.
    """
    actions = {"전진":"전진", "후진" : "후진", "좌회전" : "좌회전", "우회전" : "우회전", "멈춤" : "멈춤"}
    final_action = f"ACTION:{actions[action]}"
    
    return final_action

@tool
def emotion_tool(emotion: str='신남') -> str:
    """
    해당 도구는 로봇이 자신의 감정을 표현할때 사용합니다.
    로봇의 행동, 로봇의 대화에 적합한 감정을 선택하여 감정을 반환합니다. 
    로봇은 "신남", "화남", "슬픔", "기쁨", "무표정"의 감정을 나타낼 수 있습니다.
    """
    emotions = {"신남":"신남", "화남":"화남", "슬픔":"슬픔", "기쁨":"기쁨", "무표정":"무표정"}
    final_emotion = f"EMOTION:{emotions[emotion]}"
    
    return final_emotion


@tool
def watching_tool() -> List[Dict[str, str]]:
    """
    해당 도구는 로봇이 전방을 보고 객체를 탐지하는 도구 함수입니다. 
    로봇이 시각 정보가 필요할때 사용하세요.
    현재 카메라 피드를 기반으로 객체를 탐지합니다.
    현재 상황을 확인하고 싶거나 현재 보이는 것을 알고자 할 때 이 도구를 사용하세요.
    """
    for _ in range(5):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        return [{"error": "카메라에서 프레임을 읽지 못했습니다."}]
    # 좌우 반전 (미러 효과)
    frame = cv2.flip(frame, 1)
    results = yolo_model(source=frame, conf=0.4, verbose=False)
    info = {}
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())
            label = yolo_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x, y, w, h = map(int, box.xywh[0])
            confidence = round(box.conf[0].item(), 2)
            info[label] = {
                'full screen size': frame.shape[0] * frame.shape[1],
                'object location': [x, y],
                'object size': w * h,
                'object bbox': [x1, y1, x2, y2],
                'detection confidence': confidence,
            }
    
    return [info]