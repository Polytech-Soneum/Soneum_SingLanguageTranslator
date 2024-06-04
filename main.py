import cv2
import mediapipe as mp
import os
import numpy as np

def draw_hand_connections(frame, hand_landmarks, hand_connections):
    for connection in hand_connections:
        start_idx, end_idx = connection
        start_landmark = hand_landmarks.landmark[start_idx]
        end_landmark = hand_landmarks.landmark[end_idx]

        start_x, start_y = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
        end_x, end_y = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])

        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)

# MediaPipe 모델 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
pose = mp_pose.Pose()

video_path = #'.mp4'
cap = cv2.VideoCapture(video_path)

output_dir = #'./meet_12_image_jpg'
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    if not hands_results.multi_hand_landmarks or len(hands_results.multi_hand_landmarks) != 2:
        continue  # 다음 프레임으로 넘어감

    hand_landmarks = hands_results.multi_hand_landmarks[0]  # 첫 번째 감지된 손을 사용
    index_finger_tip_y = hand_landmarks.landmark[8].y  # 검지손가락 끝의 y좌표
    if index_finger_tip_y > 0.9:  # 프레임 높이의 90% 이상 위치에 있다면
        continue  # 이 프레임은 건너뜀
    frame_count += 1

    if hands_results.multi_hand_landmarks and pose_results.pose_landmarks:
        frame_black = np.zeros_like(frame)

        for hand_landmarks in hands_results.multi_hand_landmarks:
            draw_hand_connections(frame_black, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmark_indices = [15, 13, 11, 12, 14, 16]
        for i in range(len(landmark_indices) - 1):
            start_index = landmark_indices[i]
            end_index = landmark_indices[i + 1]
            x1, y1 = int(pose_results.pose_landmarks.landmark[start_index].x * frame.shape[1]), int(
                pose_results.pose_landmarks.landmark[start_index].y * frame.shape[0])
            x2, y2 = int(pose_results.pose_landmarks.landmark[end_index].x * frame.shape[1]), int(
                pose_results.pose_landmarks.landmark[end_index].y * frame.shape[0])
            cv2.line(frame_black, (x1, y1), (x2, y2), (0, 0, 255), 3)

        image_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(image_path, frame_black)

cap.release()


