import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import Counter

# 미디어파이프 핸드 모듈 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# 액션 라벨 매핑
action_label_map = {
    'hello': 0,
    'meet': 1,
    'greet': 2,
    'how_is_it': 3,
    'nextweek': 4,
    'today': 5,
    'tomorrow': 6,
    'weather': 7,
    'yes': 8,
    'no': 9,
    'thanks': 10,
    'eat': 11,
    'Monday': 12,
    'Tuesday': 13,
    'Wednesday': 14,
    'Thursday': 15,
    'Friday': 16,
    'Saturday': 17,
    'sunday' : 18,
    'good': 19,
    'hate': 20,
    'yesterday':21,
    'know': 22,        # 추가됨
    'do_not_know': 23, # 추가됨
    'fine': 24,        # 추가됨
    'sorry': 25,       # 추가됨
    'AM': 26,          # 추가됨
    'PM': 27,          # 추가됨
    'me': 28,          # 추가됨
    'you': 29,         # 추가됨
    'feeling': 30,     # 추가됨
    'hate': 31        # 추가됨
}
actions = list(action_label_map.keys())
seq_length = 30

# 모델 로드 (모델 경로를 수정해야 합니다)
model = tf.keras.models.load_model('model1.keras')

# 이전 액션을 저장할 변수 초기화
previous_action = None
action_seq = []
seq = []  # 프레임 데이터를 저장할 리스트 초기화

# 액션을 저장할 변수 추가
save = []


def process_frame(frame):
    global previous_action, action_seq, seq

    # 현재 프레임 데이터를 seq에 추가
    seq.append(frame)

    if len(seq) < seq_length:
        return None

    # 입력 시퀀스 준비
    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)  # 시퀀스 준비
    y_pred = model.predict(input_data).squeeze()
    i_pred = int(np.argmax(y_pred))
    conf = y_pred[i_pred]

    if conf < 0.5:
        return 'conf error'

    action = actions[i_pred]
    action_seq.append(action)

    return action  # 액션을 반환


def process_video_from_npy(npy_filename):
    global save

    # .npy 파일에서 비디오 데이터 로드
    video_data = np.load(npy_filename)

    for frame in video_data:
        action = process_frame(frame)

    # 모든 프레임 처리 후 최종 저장된 액션들을 10개씩 나누어 출력
    combined_actions = []  # 최종 출력할 액션 리스트
    for i in range(0, len(action_seq), 10):
        action_subset = action_seq[i:i + 10]  # 10개씩 나누기
        if action_subset:
            most_common_action = Counter(action_subset).most_common(1)
            if most_common_action:
                action, count = most_common_action[0]

                # 같은 단어가 연속으로 나타나는 경우 합치기
                if combined_actions and combined_actions[-1][0] == action:
                    combined_actions[-1][1] += count  # 카운트 합산
                else:
                    combined_actions.append([action, count])  # 새로운 액션 추가



    # 최종 저장된 액션들 출력 (단어만 나열)
    final_actions = ' '.join(action[0] for action in combined_actions)  # 액션만 추출하여 문자열로 결합
    print(final_actions)
    return final_actions  # 최종 결과 반환


# 실제 .npy 파일 경로로 변경하여 호출
npy_filename = '20241010_163715_412.npy'
process_video_from_npy(npy_filename)
