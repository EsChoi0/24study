from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tf_explain.core.grad_cam import GradCAM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import glob
import numpy as np

# EfficientNetB0 베이스 모델 로드
base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))

# 베이스 모델 위에 커스텀 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # 최종 클래스 수에 따라 조정

# 커스텀 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 훈련된 가중치 로드
model.load_weights('best_model.h5')

# Grad-CAM 적용
explainer = GradCAM()

for img_path in glob.glob('./data/img/*.jpg'):  # 이미지 경로 수정
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    
    # EfficientNet에 맞는 전처리 대신 훈련 때 사용한 리스케일링 적용
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)

    # 모델의 예측
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])  # 가장 확률이 높은 클래스의 인덱스

    # Grad-CAM
    data = (img_array, None)
    grid = explainer.explain(data, model, class_index=pred_index)  # 실제 예측 인덱스 사용

    # Grad-CAM 결과 이미지 저장
    save_path = img_path.replace('.jpg', '_cam.jpg')  # 원본 이미지 파일명에 '_cam' 추가
    explainer.save(grid, ".", save_path)
