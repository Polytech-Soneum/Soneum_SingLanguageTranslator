import cv2
import mediapipe as mp
import numpy as np
import time
import os

# 동작 리스트와 시퀀스 설정
actions = ['hello','meet','greet','howisit','nextweek','today','tomorrow','weather']
seq_length = 30
secs_for_action = 30

# MediaPipe 손 및 포즈 모델 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 비디오 캡처
cap = cv2.VideoCapture(1)

# 데이터 저장 폴더 생성
created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        print(f'Collecting data for {action} action...')

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result_hands = hands.process(img_rgb)
            result_pose = pose.process(img_rgb)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            hand_data = []
            pose_data = []

            # 손 데이터 처리
            if result_hands.multi_hand_landmarks:
                for res in result_hands.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # 관절 간 각도 계산
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                    v = v2 - v1
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                    angle = np.degrees(angle)

                    angle_label = np.concatenate([angle, [idx]])

                    d = np.concatenate([joint.flatten(), angle_label])
                    hand_data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # 포즈 데이터 처리
            if result_pose.pose_landmarks:
                selected_landmarks = [result_pose.pose_landmarks.landmark[i] for i in [11, 12, 13, 14, 15, 16]]
                pose_joint = np.zeros((6, 4))

                for j, lm in enumerate(selected_landmarks):
                    pose_joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # 랜드마크 시각화
                for lm in selected_landmarks:
                    cx, cy = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)

                # 랜드마크 간 연결 (정확한 인덱스 사용)
                connections = [(0, 2), (2, 4), (1, 3), (3, 5), (1, 0)]
                for (start_idx, end_idx) in connections:
                    start = selected_landmarks[start_idx]
                    end = selected_landmarks[end_idx]
                    start_point = (int(start.x * img.shape[1]), int(start.y * img.shape[0]))
                    end_point = (int(end.x * img.shape[1]), int(end.y * img.shape[0]))
                    cv2.line(img, start_point, end_point, (255, 0, 0), 3)

                # 포즈 데이터와 손 데이터 결합
                if hand_data:
                    for hand_d in hand_data:
                        d = np.concatenate([pose_joint.flatten(), hand_d])
                        data.append(d)

            # 손과 포즈 랜드마크가 없을 경우 경고 메시지 표시
            if not hand_data and not result_pose.pose_landmarks:
                cv2.putText(img, 'No hands or pose detected!', org=(10, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        # 시퀀스 데이터 생성
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(f'{action} action sequence data: {full_seq_data.shape}')
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)

    break

cap.release()
cv2.destroyAllWindows()
