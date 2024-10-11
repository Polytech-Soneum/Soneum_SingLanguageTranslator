import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 액션 목록
actions = [
    'hello', 'meet', 'greet', 'how_is_it', 'nextweek', 'today',
    'tomorrow', 'weather', 'yes', 'no', 'thanks', 'eat', 'monday',
    'tuesday', 'wednesday', 'thursday', 'friday', 'saturday','sunday', 'good',
    'bad', 'yesterday','know','do_not_know','fine','sorry','AM','PM','me','you','feeling','hate'
]


# cleaned_sequences 폴더에서 .npy 파일 가져오기
data_files = glob.glob('/content/drive/MyDrive/sequencedatas/*.npy')
print(data_files)

# 데이터 로드 및 결합
data = []
for file in data_files:
    loaded_data = np.load(file)
    data.append(loaded_data)

# 모든 데이터를 하나로 결합
data = np.concatenate(data, axis=0)

# 데이터의 형태 확인
print("Original data shape:", data.shape)

# x_data와 labels 생성
x_data = data[:, :, :-1]
labels = data[:, 0, -1]
print("x_data shape:", x_data.shape)
print("labels shape:", labels.shape)

# 레이블을 원-핫 인코딩
y_data = to_categorical(labels, num_classes=len(actions))

# 데이터 타입 변환
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

# 훈련 및 검증 데이터 분리
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2021)

print("Training data shape:", x_train.shape, y_train.shape)
print("Validation data shape:", x_val.shape, y_val.shape)

# 초기 러닝 레이트 설정
initial_learning_rate = 0.001  # 원하는 초기 러닝 레이트로 설정

# 모델 정의
model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

# 모델 컴파일 (러닝 레이트 적용)
model.compile(optimizer=Adam(learning_rate=initial_learning_rate),
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()

# 모델 학습
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/model32.keras', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=20, verbose=1, mode='auto')
    ]
)

# 훈련 과정 시각화 (선택 사항)
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()