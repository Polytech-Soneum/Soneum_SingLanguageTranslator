import time
import tensorflow as tf
import mediapipe as mp
import numpy as np
import os
from collections import Counter

# 동작 리스트와 시퀀스 설정
action_label_map = {
    '안녕하다': 0,
    '만나다': 1,
    '반갑다': 2,
    '무엇': 3,
    '다음주': 4,
    '오늘': 5,
    '내일': 6,
    '날씨': 7,
    '그렇다': 8,
    '아니다': 9,
    '고맙다': 10,
    '먹다': 11,
    '월요일': 12,
    '화요일': 13,
    '수요일': 14,
    '목요일': 15,
    '금요일': 16,
    '토요일': 17,
    '일요일': 18,
    '좋다': 19,
    '나쁘다': 20,
    '어제': 21,
    '알아요': 22,
    '모르다': 23,
    '괜찮다': 24,
    '미안하다': 25,
    '오전': 26,
    '오후': 27,
    '나': 28,
    '너': 29,
    '기분': 30,
    '싫다': 31
}

actions = list(action_label_map.keys())
seq_length = 30

# 모델을 전역 변수로 선언
model = None

# MediaPipe 손 및 포즈 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

seq = []


def load_model_once():
    """
    TensorFlow 모델을 서버 시작 시에 한 번만 로드
    """
    global model
    if model is None:
        model = tf.keras.models.load_model('./utils/models/sign_translator.keras')

def process_frame_sequence():
    """
    30개의 프레임 시퀀스를 처리하여 동작 예측
    """
    global seq, model

    if len(seq) < seq_length:
        return None  # 시퀀스가 채워지지 않음

    # 프레임을 관리하기 위한 인덱스
    start = 0
    th = 10
    end = 20

    # seq의 프레임에서 필요한 부분만 선택
    selected_frames = seq[th:end]  # 10에서 20 프레임 선택

    # 시퀀스 데이터 준비
    input_data = np.expand_dims(np.array(selected_frames, dtype=np.float32), axis=0)  # (1, 10, 156) 형태로 변환
    y_pred = model.predict(input_data).squeeze()
    i_pred = int(np.argmax(y_pred))
    conf = y_pred[i_pred]

    if conf < 0.9:
        return 'conf error'

    action = actions[i_pred]

    # 예측된 동작 출력
    print(f"예측된 동작: {action}")

    return action


def save_actions_to_npy(actions, filename):
    # 동작 리스트를 numpy 배열로 변환하고 저장
    np.save(filename, actions)
    print(f"Actions saved to {filename}")


def process_video_from_npy(npy_filename):
    global seq  # seq를 전역 변수로 사용
    # 지정된 .npy 파일 로드
    video_data = np.load(npy_filename)

    # 전체 프레임 수 확인
    print(f"전체 프레임 수: {len(video_data)}")

    action_counts = Counter()

    # 모든 프레임에 대해 동작 처리
    for frame_idx, frame in enumerate(video_data):
        seq.append(frame)  # 프레임 추가

        # 현재 seq 상태 출력
        print(f"프레임 {frame_idx + 1} 추가: 현재 시퀀스 길이 = {len(seq)}")

        # 시퀀스 길이가 seq_length에 도달하면 동작 예측
        if len(seq) == seq_length:
            action = process_frame_sequence()
            if action and action != 'conf error':
                action_counts[action] += 1
            seq = []  # 시퀀스 초기화 (다음 시퀀스를 위해)

    # 최종적으로 카운트된 모든 동작들 출력
    print(f"각 동작의 최종 카운트: {action_counts}")

    actions = [action for action, count in action_counts.items() if count > 0]
    print(f"최종 추출된 동작 리스트: {actions}")

    # 추출된 동작들을 콤마로 구분한 문자열로 반환
    final_actions = ','.join(actions)

    return final_actions
