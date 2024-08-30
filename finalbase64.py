import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

# Keras 모델 및 설정 초기화
actions = ['hello', 'meet', 'greet', 'what', 'nextweek', 'today', 'tomorrow', 'weather']
seq_length = 30  # 모델이 기대하는 시퀀스 길이
model = load_model('model7.keras')

# MediaPipe Hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

seq = []
action_seq = []
previous_action = None  # 이전 프레임의 예측 결과를 저장할 변수


def process_frame(frame):
    global seq, previous_action  # seq와 previous_action 변수를 전역 변수로 사용
    if frame.dtype == np.float64:
        frame = (frame * 255).astype(np.uint8)  # 이미지 값이 0~1 범위일 경우 0~255로 변환

    joint = frame.reshape((21, 4))

    # Calculate angles between joints
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
    v = v2 - v1
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
    angle = np.degrees(angle)

    d = np.concatenate([joint.flatten(), angle])

    # 시퀀스에 추가
    seq.append(d)
    if len(seq) < seq_length:
        return None

    # 입력 시퀀스 준비
    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)  # 시퀀스 준비
    y_pred = model.predict(input_data).squeeze()
    i_pred = int(np.argmax(y_pred))
    conf = y_pred[i_pred]

    print(f"Predicted action: {actions[i_pred]}, Confidence: {conf}")

    if conf < 0.7:
        return '?'

    action = actions[i_pred]
    action_seq.append(action)

    # 이전 액션과 현재 액션이 다르면 출력
    if action != previous_action:
        previous_action = action  # 이전 액션을 현재 액션으로 업데이트
        if len(action_seq) >= 2 and action_seq[-1] == action_seq[-2]:
            print(f"Detected action: {action}")
            return action
        else:
            return '?'

    return None  # 이전 액션과 동일하면 None 반환


def process_video_from_npy(npy_filename):
    # .npy 파일에서 비디오 데이터 로드
    video_data = np.load(npy_filename)
    print(f"Loaded video data with shape: {video_data.shape}")

    frame_count = 0

    for frame in video_data:
        frame_count += 1
        if frame_count % 15 != 0:
            continue

        print(f"Processing frame {frame_count}")
        action = process_frame(frame)
        if action:
            print(f"Detected action: {action}")


# 사용 예시
npy_filename = 'dataset/hand_data_1724985080.npy'
process_video_from_npy(npy_filename)
