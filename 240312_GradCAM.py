import os
from traceback import print_tb
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import matplotlib.cm as cm

# 모델 파일 경로 및 모델 불러오기
model_path = 'EfficientNetB0_finetune.h5'
model = load_model(model_path)

# 마지막 합성곱 층의 이름 확인
# last_conv_layer_name = 'top_conv' # EfficientNet.finetune last layer name

# 폴더 경로
# 클래스별 이미지 폴더 경로
class_folders = {
    'a': 'C:\\Users\\AI-Esc\\Desktop\\Python\\01',
    'b': 'C:\\Users\\AI-Esc\\Desktop\\Python\\03',
    'c': 'C:\\Users\\AI-Esc\\Desktop\\Python\\a7',
    'd': 'C:\\Users\\AI-Esc\\Desktop\\Python\\a8',
}
save_folder = "C:\\Users\\AI-Esc\\Desktop\\Python\\Grad-CAM"

# 불러온 모델에서 'efficientnetb0' layer의 내부 layer를 가져옴
efficientnetb0_layer = model.get_layer('efficientnetb0')
# 'efficientnetb0' 내부 layer 중 마지막 Covolution layer의 이름인 'top_conv'를 가져옴
last_conv_layer = efficientnetb0_layer.get_layer('top_conv')

print(last_conv_layer.output)


# last_conv_layer_model = Model(inputs=model.inputs, outputs=last_conv_layer.output) # 얘가 안됨
# print(last_conv_layer_model)

# last_conv_layer_model = tf.keras.models.Model(
#     [model.inputs], [model.get_layer('top_conv').output, model.output]
# ) 
# print(last_conv_layer_model)



# # Grad-CAM 히트맵 생성 함수
# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     # 불러온 모델에서 'efficientnetb0' layer의 내부 layer를 가져옴
#     efficientnetb0_layer = model.get_layer('efficientnetb0')
#     # 'efficientnetb0' 내부 layer 중 마지막 Covolution layer의 이름인 'top_conv'를 가져옴
#     last_conv_layer = efficientnetb0_layer.get_layer('top_conv')
#     #########여기까지 됨됨
#     # 마지막 컨볼루션 레이어의 출력을 포착합니다.
#     # last_conv_layer_model = Model(inputs=model.inputs, outputs=last_conv_layer.output) # 얘가 안됨
#     last_conv_layer_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     ) 
#     print(last_conv_layer_model)



#     # GradientTape를 사용하여 'top_conv' 레이어의 출력에 대한 예측 클래스의 그래디언트를 계산합니다.
#     with tf.GradientTape() as tape:
#         last_conv_layer_output = last_conv_layer_model(img_array)
#         tape.watch(last_conv_layer_output)
#         # model의 출력을 가져옵니다.
#         preds = model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         top_class_channel = preds[:, pred_index]

#     # 지정된 클래스에 대한 'top_conv' 레이어의 출력의 그래디언트를 계산합니다.
#     grads = tape.gradient(top_class_channel, last_conv_layer_output)

#     # 그래디언트의 풀링 평균을 계산합니다.
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     # 'top_conv' 레이어의 출력에 대한 가중치 풀링 그래디언트를 계산합니다.
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     # 히트맵을 0과 1 사이로 정규화합니다.
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()



# for class_name, folder_path in class_folders.items():
#     save_class_folder = os.path.join(save_folder, class_name)
#     if not os.path.exists(save_class_folder):
#         os.makedirs(save_class_folder)
    
#     for img_file in os.listdir(folder_path):
#         img_path = os.path.join(folder_path, img_file)
#         img = load_img(img_path, target_size=(224, 224))
#         img_array = img_to_array(img)
#         img_array = preprocess_input(img_array)
#         img_array = np.expand_dims(img_array, axis=0)

#         # 가장 확신하는 클래스의 인덱스 찾기
#         preds = model.predict(img_array)
#         pred_index = np.argmax(preds[0])

#         # Grad-CAM 히트맵 생성
#         heatmap = make_gradcam_heatmap(img_array, model, 'top_conv', pred_index)
#         heatmap = np.uint8(255 * heatmap)
#         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#         # 원본 이미지에 히트맵 오버레이
#         img_original = cv2.imread(img_path)
#         img_original = cv2.resize(img_original, (224, 224))
#         superimposed_img = heatmap * 0.4 + img_original
#         superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

#         # 결과 저장
#         save_path = os.path.join(save_class_folder, img_file)
#         cv2.imwrite(save_path, superimposed_img)