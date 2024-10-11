import os
import numpy as np
import glob

# 데이터셋 폴더 경로 설정
input_folder = 'Refined_dataset'  # 액션별 .npy 파일들이 있는 폴더
output_folder = 'sequencedatas'  # 시퀀스 데이터를 저장할 폴더

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 모든 액션에 대한 .npy 파일 경로 가져오기
npy_files = glob.glob(os.path.join(input_folder, '*.npy'))  # 하위 디렉토리에서 모든 .npy 파일 가져오기

# 시퀀스 길이 설정
seq_length = 30  # 시퀀스 길이

# 각 .npy 파일 처리
for npy_file in npy_files:
    print(f"Loading file: {npy_file}")  # 로드할 파일 경로 출력
    if not os.path.exists(npy_file):
        print(f"File does not exist: {npy_file}")
        continue

    try:
        # .npy 파일에서 데이터 로드
        data = np.load(npy_file)

        # 시퀀스를 생성하여 numpy 배열로 변환
        if len(data) >= seq_length:
            sequences = []
            for i in range(0, len(data) - seq_length + 1):
                sequences.append(data[i:i + seq_length])

            # 시퀀스를 numpy 배열로 변환
            sequences = np.array(sequences)

            # 수정된 데이터 numpy 배열로 저장
            output_file = os.path.join(output_folder, os.path.basename(npy_file).replace('.npy', '_sequences.npy'))
            np.save(output_file, sequences)
            print(f"Saved sequences to: {output_file}")
        else:
            print(f"Data length is less than {seq_length} for file: {npy_file}")

    except Exception as e:
        print(f"Error loading {npy_file}: {e}")

