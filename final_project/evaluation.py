import argparse
import torch
import sys
import os
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import time
# 상위 디렉토리를 Python 경로에 추가
# sys.path.append("..")  # training 디렉토리의 상위 디렉토리를 추가
from models import load_teacher_model, load_student_model  # models.py에서 함수 가져오기
from dataset import load_coco_dataset, preprocess_coco_dataset, split_train_validation
from utils import get_data_loaders, evaluate_model_i2t, evaluate_model_t2i, get_model_size

torch.backends.quantized.engine = "qnnpack" 

# 1. 명령어 파싱 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLIP models")
    parser.add_argument("--model", type=str, required=True, choices=["teacher", "student", "distilled"],
                        help="Choose the model type: 'teacher' for the base CLIP model, 'student' for the quantized CLIP model.")
    parser.add_argument('--model_name', type=str, default=None,
                    help='Filename of the trained model to load (required for distilled model)')

    return parser.parse_args()

# 2. 메인 실행 함수
def main(args):
    # CPU 고정
    device = torch.device("cpu")
    print("Using device: CPU")

    # 모델 로드
    if args.model == "teacher":
        print("Loading base teacher CLIP model...")
        model = load_teacher_model().to(device)
    elif args.model == "student":
        print("Loading quantized student CLIP model...")
        model = load_student_model(quantized=True).to(device)
    elif args.model == "distilled":
        if args.model_name is None:
            raise ValueError("For distilled model, --model_name must be specified.")
        print("Loading distilled student CLIP model...")
        model = load_student_model(quantized=False).to(device)
        model.load_state_dict(torch.load(args.model_name, map_location=device))
        model.eval()
        # 양자화 적용
        torch.backends.quantized.engine = "qnnpack"
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        print("Quantized distilled model loaded.")

    # 각 레이어와 데이터 타입 출력
    # quantization 확인용
    print("\nChecking model layers and data types:")
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Data Type: {param.dtype}")

    # Processor 로드
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 데이터 로딩
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32, validation_size=0.2)

    # 평가: Image-to-Text (I2T)
    print("Evaluating Image-to-Text (I2T)...")
    recall_at_1_i2t, recall_at_5_i2t, latency_i2t = evaluate_model_i2t(model, val_loader, processor, device)

    # 평가: Text-to-Image (T2I)
    print("Evaluating Text-to-Image (T2I)...")
    recall_at_1_t2i, recall_at_5_t2i, latency_t2i = evaluate_model_t2i(model, val_loader, processor, device)

    # 모델 크기 계산
    model_size = get_model_size(model)

    # 결과 출력
    print(f"\nResults for {args.model.upper()} Model:")
    print(f"I2T - Recall@1: {recall_at_1_i2t * 100:.2f}%, Recall@5: {recall_at_5_i2t * 100:.2f}%, Latency: {latency_i2t:.2f} ms")
    print(f"T2I - Recall@1: {recall_at_1_t2i * 100:.2f}%, Recall@5: {recall_at_5_t2i * 100:.2f}%, Latency: {latency_t2i:.2f} ms")
    print(f"Model Size: {model_size:.2f} MB")

if __name__ == "__main__":
    args = parse_args()
    main(args)