import time
import tensorflow as tf
import mediapipe as mp
import numpy as np
from collections import Counter

# 동작 리스트와 시퀀스 설정
action_label_map = {
    '안녕하다': 0, '만나다': 1, '반갑다': 2, '무엇': 3, '다음주': 4,
    '오늘': 5, '내일': 6, '날씨': 7, '그렇다': 8, '아니다': 9,
    '고맙다': 10, '먹다': 11, '월요일': 12, '화요일': 13, '수요일': 14,
    '목요일': 15, '금요일': 16, '토요일': 17, '일요일': 18, '좋다': 19,
    '나쁘다': 20, '어제': 21, '알아요': 22, '모르다': 23, '괜찮다': 24,
    '미안하다': 25, '오전': 26, '오후': 27, '나': 28, '너': 29,
    '기분': 30, '싫다': 31
}

actions = list(action_label_map.keys())
seq_length = 30

# 모델을 전역 변수로 선언
model = None

# MediaPipe 손 및 포즈 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

seq = []
predictions = []


def load_model_once():
    global model
    if model is None:
        model = tf.keras.models.load_model('./utils/models/sign_translator.keras')


def process_frame_sequence():
    global seq, model, predictions

    if len(seq) < seq_length:
        return None  # 시퀀스가 채워지지 않음

    # 중간 구간 프레임만 사용 (10번째부터 20번째까지)
    selected_frames = seq[10:20]

    # 예측을 위한 입력 데이터 생성
    input_data = np.expand_dims(np.array(selected_frames, dtype=np.float32), axis=0)
    y_pred = model.predict(input_data).squeeze()
    i_pred = int(np.argmax(y_pred))
    conf = y_pred[i_pred]

    # 신뢰도가 0.9 이상일 때만 유효한 예측으로 간주
    if conf < 0.9:
        return None

    action = actions[i_pred]
    predictions.append(action)

    # 중복된 예측 중 5개 이상 동일한 경우만 동작 출력
    if predictions:
        most_common_action, count = Counter(predictions).most_common(1)[0]
        if count >= 5:
            predictions.clear()  # 예측 리스트 초기화
            return most_common_action
    return None


def process_video_data(video_data):
    global seq, predictions
    action_counts = Counter()
    total_frames = len(video_data)
    skip_initial = True  # 첫 번째 시퀀스를 예측에서 제외하기 위한 플래그

    # 첫 번째 시퀀스를 건너뛰기 위해 30번째 프레임부터 처리 시작
    for frame_idx in range(30, total_frames - 30):  # 마지막 30 프레임은 날림
        frame = video_data[frame_idx]
        seq.append(frame)

        # 현재 시퀀스 길이 출력
        print(f"프레임 {frame_idx + 1} 추가: 현재 시퀀스 길이 = {len(seq)}")

        # 시퀀스 길이가 seq_length에 도달하면 동작 예측 수행
        if len(seq) == seq_length:
            # 첫 번째 시퀀스는 건너뛰고 다음 시퀀스부터 예측
            if skip_initial:
                skip_initial = False
                seq = seq[1:]  # 첫 번째 시퀀스 이동
                continue

            action = process_frame_sequence()
            if action:
                action_counts[action] += 1
            seq = []  # 시퀀스 초기화 (다음 시퀀스를 위해)

    # 각 동작의 최종 카운트 출력
    actions_list = [action for action, count in action_counts.items() if count > 0]

    # 추출된 동작들을 콤마로 구분한 문자열로 반환
    final_actions = ','.join(actions_list)
    print(final_actions)
    return final_actions
