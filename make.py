import cv2
import numpy as np
import mediapipe as mp
import time
import os

# 미디어파이프 핸드 모듈 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 액션 수집 시간 (초)
secs_for_action = 90  # 액션 수집 시간 설정
wait_time_before_next_action = 3  # 다음 액션으로 넘어가기 전 대기 시간 (초)

# 라벨을 숫자로 매핑
action_label_map = {

}
actions = list(action_label_map.keys())  # 수행할 액션 리스트

# 데이터셋 폴더 생성
dataset_folder = 'dataset'
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# 각 액션에 대한 서브폴더 생성
for action in actions:
    action_folder = os.path.join(dataset_folder, action)
    if not os.path.exists(action_folder):
        os.makedirs(action_folder)

# 각 액션에 대한 데이터 저장을 위한 딕셔너리 초기화
data_dict = {action: [] for action in actions}

# 웹캠 입력 시작
cap = cv2.VideoCapture(1)

for action in actions:
    label = action_label_map[action]  # 해당 액션에 대한 숫자 라벨
    start_time = time.time()  # 시작 시간 저장

    while True:
        ret, frame = cap.read()
        # 프레임 처리
        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 각 손의 랜드마크 처리
        x_values = []
        y_values = []
        z_values = []
        angles = []  # 각도 값 저장용

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                # 랜드마크 그리기
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 부모-자식 관절 설정
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]

                # 벡터 계산 및 정규화
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 각도 계산
                angle = np.arccos(np.einsum('nt,nt->n',
                                              v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                              v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                angle = np.degrees(angle)

                # 랜드마크 좌표 저장
                for lm in hand_landmarks.landmark:
                    x_values.append(lm.x)
                    y_values.append(lm.y)
                    z_values.append(lm.z)

                angles.extend(angle.tolist())

        # 랜드마크가 감지되지 않은 경우도 처리
        if len(x_values) == 0:
            # 랜드마크가 감지되지 않으면 0으로 채움
            x_values = [0.0] * 21
            y_values = [0.0] * 21
            z_values = [0.0] * 21
            angles = [0.0] * 30  # 각도도 0으로 채움

        # 랜드마크가 42개가 되도록 0으로 채움
        while len(x_values) < 42:
            x_values.append(0.0)
            y_values.append(0.0)
            z_values.append(0.0)

        # 각도 값이 없는 경우 0으로 채움
        if len(angles) < 30:
            angles += [0.0] * (30 - len(angles))  # 부족한 만큼 0으로 채움

        # 숫자로 변환된 라벨 추가
        label_float = float(label)

        # 데이터에 추가
        # 데이터 길이를 157로 고정
        frame_data = x_values + y_values + z_values + angles + [label_float]
        if len(frame_data) == 157:  # 157개로 맞춰지면 추가
            data_dict[action].append(frame_data)

        # 데이터 개수 출력
        print(f"Action: {action} | Frame Count: {len(data_dict[action])}")

        # 화면에 텍스트 출력
        cv2.putText(image, f"Action: {action} | Frame Count: {len(data_dict[action])}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 화면에 출력
        cv2.imshow('MediaPipe Hands', image)

        # secs_for_action 초가 지나면 다음 액션으로 넘어감
        if time.time() - start_time > secs_for_action:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # 다음 액션으로 넘어가기 전 3초 대기
    wait_message = f"'{action}' 액션 완료. 다음 액션으로 넘어갑니다..."
    print(wait_message)
    for i in range(wait_time_before_next_action, 0, -1):
        cv2.putText(image, wait_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f"다음 액션 {i}초 후 시작", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('MediaPipe Hands', image)
        cv2.waitKey(1000)  # 1초 대기

# 모든 액션에 대한 데이터를 numpy 배열로 변환하고 저장
for action, data in data_dict.items():
    try:
        data_array = np.array(data, dtype=float)  # dtype=float으로 변경하여 데이터 배열 통일
        # 데이터셋 폴더에 각 액션별로 npy 파일로 저장
        np.save(os.path.join(dataset_folder, action, f"{action}_hand_data.npy"), data_array)
        print(f"Saved data for action: {action} | Total frames: {len(data)}")
    except ValueError as e:
        print(f"Error converting data for action {action} to numpy array: {e}")

cap.release()
cv2.destroyAllWindows()
