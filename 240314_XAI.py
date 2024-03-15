from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tf_explain.core.grad_cam import GradCAM
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# 모델 로드
model = EfficientNetB0(weights='imagenet', include_top=True, input_shape=(224, 224, 3))  # include_top=True로 변경

# Grad-CAM 적용
explainer = GradCAM()

for img_path in glob.glob('./data/img/*.jpg'):  # 이미지 경로 수정
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # EfficientNet에 맞는 전처리
    img_array = np.expand_dims(img_array, axis=0)

    # 모델의 예측
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])  # 가장 확률이 높은 클래스의 인덱스

    # Grad-CAM
    data = (img_array, None)
    grid = explainer.explain(data, model, class_index=pred_index)  # imgnet_index 대신 실제 예측 인덱스 사용

    # Grad-CAM 결과 이미지 저장
    save_path = img_path.replace('.jpg', '_cam.jpg')  # 원본 이미지 파일명에 '_cam' 추가
    explainer.save(grid, ".", save_path)

# Grad-CAM 결과 출력
images_cams = []
for img_path in glob.glob('./Grad-CAM/*_cam.jpg'):  # Grad-CAM 이미지 경로 수정
    images_cams.append(mpimg.imread(img_path))
