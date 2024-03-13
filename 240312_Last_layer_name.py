from tensorflow.keras.models import load_model

# 저장된 모델 로드
model_path = 'EfficientNetB0_finetune.h5'
model = load_model(model_path)

# 모델 레이어 요약 정보 출력
model.summary()

# EfficientNetB0 내부 레이어 탐색
efficientnetb0_layer = None
for layer in model.layers:
    if 'efficientnetb0' in layer.name:
        efficientnetb0_layer = layer
        break

if efficientnetb0_layer is not None:
    # EfficientNetB0 내부 레이어 요약 정보 출력
    efficientnetb0_layer.summary()
    # 마지막 Convolutional Layer 이름 찾기
    for layer in reversed(efficientnetb0_layer.layers):
        if 'conv' in layer.name:
            print("마지막 Convolutional Layer의 이름:", layer.name)
            break
