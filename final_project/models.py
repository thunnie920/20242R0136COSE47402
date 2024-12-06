import torch
from transformers import CLIPModel
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, get_default_qat_qconfig, convert


# Teacher Model: Pre-trained CLIP
def load_teacher_model():
    """
    Pre-trained CLIP 모델(교사 모델)을 로드합니다.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print("Teacher model loaded.")
    return model


# Student Model: Quantized CLIP
def load_student_model(quantized=True):
    """
    양자화된 CLIP 모델(학생 모델)을 로드합니다.
    양자화 옵션을 사용하지 않을 경우, 기본 모델을 로드합니다.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    if quantized:
        torch.backends.quantized.engine = "qnnpack" 
        # Apply dynamic quantization to reduce model size and improve efficiency
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        print("Quantized student model loaded.")
    else:
        print("Non-quantized student model loaded.")
    return model


def final_distilled_model(quantized=False, model_path = '/Users/seoeun/Desktop/DeepLearning/final_project/final_model.pth'):
    """
    학습된 Distilled 모델을 로드합니다.
    Args:
        quantized (bool): True이면 양자화된 모델을 로드합니다.
        model_path (str): 학습된 모델의 가중치가 저장된 파일 경로입니다.
    Returns:
        model (CLIPModel): 로드된 Distilled 모델입니다.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    if quantized:
        torch.backends.quantized.engine = "qnnpack"
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        print("Quantized distilled model loaded.")
    else:
        print("Non-quantized distilled model loaded.")

    # 학습된 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Distilled model weights loaded from: {model_path}")
    return model