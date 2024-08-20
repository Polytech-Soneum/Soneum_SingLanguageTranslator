import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ['hello', 'meet', 'greet', 'howisit', 'nextweek', 'today', 'tomorrow', 'weather']
seq_length = 30

# 모델 로드
model = load_model('model.keras')

# MediaPipe hands 및 pose 모델 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

seq = []
action_seq = []

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img0 = img.copy()

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
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]  # 부모 관절
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]  # 자식 관절
            v = v2 - v1  # [20, 3]
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # 각도 계산
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))  # [15,]
            angle = np.degrees(angle)  # 라디안을 도로 변환

            hand_data.append(np.concatenate([joint.flatten(), angle]))

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

        pose_data.append(pose_joint.flatten())

    # 포즈와 손 데이터 결합
    if hand_data and pose_data:
        combined_data = np.concatenate([pose_data[0], hand_data[0]])
        seq.append(combined_data)

        if len(seq) >= seq_length:
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf >= 0.9:
                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) >= 5:
                    # 마지막 5개의 동작이 동일한지 확인
                    if all(a == action_seq[-1] for a in action_seq[-5:]):
                        cv2.putText(img, f'{action.upper()}', org=(50, 50),  # 화면의 고정된 위치에 동작 이름 표시
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        print(action)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
