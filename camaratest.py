import cv2
import mediapipe as mp
import numpy as np
import time
import os

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

data = []

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    cv2.putText(img, 'Collecting hand landmarks...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.imshow('img', img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            d = joint.flatten()
            data.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 데이터 배열을 numpy array로 변환
data = np.array(data)
print(f'Data shape: {data.shape}')

# .npy 파일로 저장
filename = os.path.join('dataset', f'hand_data_{created_time}.npy')
np.save(filename, data)

# Create sequence data


