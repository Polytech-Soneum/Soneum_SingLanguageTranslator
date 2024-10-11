import os
import cv2
import numpy as np
import glob

# 데이터셋 폴더 경로 설정
dataset_folder = '/dataset'  # 정제 폴더 안의 2 폴더 경로
output_folder = './Refined_dataset'   # 처리된 .npy 파일을 저장할 폴더

# 모든 액션에 대한 .npy 파일 경로 가져오기
npy_files = glob.glob(os.path.join(dataset_folder, '*/*.npy'))  # 2 폴더에서 .npy 파일만 가져오기

# 각 .npy 파일 처리
for npy_file in npy_files:
    print(f"Loading file: {npy_file}")  # 로드할 파일 경로 출력

    try:
        # .npy 파일에서 데이터 로드
        data = np.load(npy_file)
        print(f"Data shape: {data.shape}")  # 데이터의 형태 출력

        # 시각화할 랜드마크 수 설정
        num_landmarks = 42  # 랜드마크 수

        valid_frames = []  # 유효한 프레임을 저장할 리스트
        deleted_frame_indices = []  # 삭제할 프레임 인덱스 저장
        current_frame = 0  # 현재 프레임 인덱스

        # 프레임별로 시각화 및 처리
        while current_frame < len(data):
            frame_data = data[current_frame]  # 현재 프레임 데이터

            # 랜드마크 좌표 추출
            x_values = frame_data[:num_landmarks]
            y_values = frame_data[num_landmarks:num_landmarks*2]
            label = frame_data[-1]  # 액션 라벨

            # 빈 화면 생성
            image = np.zeros((480, 640, 3), dtype=np.uint8)  # 640x480 크기의 빈 이미지

            # 랜드마크 시각화
            for x, y in zip(x_values, y_values):
                if x != 0.0 and y != 0.0:  # 랜드마크가 감지된 경우만 그리기
                    image_height = int(y * image.shape[0])
                    image_width = int(x * image.shape[1])
                    cv2.circle(image, (image_width, image_height), 5, (255, 255, 255), -1)  # 랜드마크 그리기

            # 액션 라벨 표시
            cv2.putText(image, f'Action: {int(label)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 프레임 출력
            cv2.imshow('Landmark Visualization', image)

            # 키 입력 처리
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                current_frame += 1  # 다음 프레임으로 이동
            elif key == ord('d'):
                deleted_frame_indices.append(current_frame)  # 삭제할 프레임 인덱스 추가
                current_frame += 1  # 다음 프레임으로 이동
            elif key == ord('r'):
                if current_frame > 0:
                    current_frame -= 1  # 현재 프레임 인덱스 감소
                continue  # 반복문 계속 진행

        # 삭제할 프레임 인덱스를 제외한 유효한 프레임 저장
        valid_frames = np.array([data[i] for i in range(len(data)) if i not in deleted_frame_indices])

        # 액션 라벨에 맞는 하위 폴더 생성
        action_label = int(label)
        action_folder = os.path.join(output_folder, str(action_label))
        if not os.path.exists(action_folder):
            os.makedirs(action_folder)

        # 수정된 데이터 numpy 배열로 저장
        output_file = os.path.join(action_folder, os.path.basename(npy_file).replace('.npy', '_pypy.npy'))
        np.save(output_file, valid_frames)
        print(f"Saved cleaned data to: {output_file}")

    except Exception as e:
        print(f"Error loading {npy_file}: {e}")

# 모든 창 닫기
cv2.destroyAllWindows()
