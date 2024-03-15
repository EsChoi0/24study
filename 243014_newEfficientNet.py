from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 크기 정의 (모델에 따라 조정해야 할 수 있음)
IMG_WIDTH, IMG_HEIGHT = 224, 224

train_data_dir = 'resized_image/train'  # 훈련 데이터셋 경로
validation_data_dir = 'resized_image/val'  # 검증 데이터셋 경로
test_data_dir = 'resized_image/test'  # 테스트 데이터셋 경로

# 훈련 데이터를 위한 데이터 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# 검증 및 테스트 데이터를 위한 데이터 전처리 (여기서는 단순 리스케일링만 적용)
test_datagen = ImageDataGenerator(rescale=1./255)

# 훈련 데이터 생성기
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=32,
    class_mode='categorical')

# 검증 데이터 생성기
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=32,
    class_mode='categorical')

# 테스트 데이터 생성기
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)  # 테스트 데이터의 경우 순서를 유지해야 하므로 shuffle을 False로 설정


# EfficientNetB0 모델 불러오기 (사전 훈련된 가중치 없이)
base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))

# 모델 커스터마이징
x = base_model.output
x = GlobalAveragePooling2D()(x)  # 평균 풀링 층 추가
x = Dense(1024, activation='relu')(x)  # 새로운 FC 층 추가
predictions = Dense(4, activation='softmax')(x)  # 최종 출력 층 (클래스 수에 따라 변경해야 함)

# 전체 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, mode='min')


# 모델 컴파일
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = model.fit(
    train_generator,
    epochs=50,  # 훈련 에폭 수
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)
