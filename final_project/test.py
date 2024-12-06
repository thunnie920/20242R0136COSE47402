import argparse
import torch
from transformers import CLIPProcessor
from models import load_teacher_model, load_student_model  # 필요한 함수들 import
from PIL import Image
from torch.utils.data import DataLoader

from models import load_teacher_model, load_student_model
from dataset import load_coco_dataset, split_train_validation, preprocess_coco_dataset_labels
from utils import evaluate_model_i2t, evaluate_model_t2i, get_model_size



# 명령어 파싱 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLIP models")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--distilled_model_path", type=str, default=None, help="Path to the distilled model file")
    return parser.parse_args()

def generate_text_from_image(model, processor, image, all_texts, device="cpu"):
    """
    이미지를 입력받아 텍스트를 생성하는 함수
    """
    model.eval()
    
    # 이미지 입력 처리
    image_inputs = processor(images=image, return_tensors="pt", padding=True)
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    
    # 텍스트 입력 처리
    text_inputs = processor(text=all_texts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # 이미지와 텍스트 임베딩 생성
    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_inputs)  # [1, D]
        text_embeddings = model.get_text_features(**text_inputs)  # [N, D]

    # 유사도 계산
    similarities = torch.matmul(image_embeddings, text_embeddings.T)  # [1, N]
    predicted_index = similarities.argmax(dim=-1).item()
    return all_texts[predicted_index]

def main(args):
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 모델 로드
    print("Loading Teacher model...")
    teacher_model = load_teacher_model().to(device)
    
    print("Loading Student model...")
    student_model = load_student_model(quantized=True).to(device)

    distilled_model = None
    if args.distilled_model_path:
        print("Loading Distilled model...")
        distilled_model = load_student_model(quantized=False).to(device)
        distilled_model.load_state_dict(torch.load(args.distilled_model_path, map_location=device))

    # Processor 로드
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 테스트 이미지 로드
    print(f"Loading image from {args.image_path}...")
    image = Image.open(args.image_path).convert("RGB")

    # 모든 텍스트 후보
    """
    all_texts = ["A dog playing in the park", "A cat sitting on the couch", 
                 "A car driving on the highway", "A person riding a bike", 
                 "A beautiful sunset over the ocean"]
    """
    # 데이터셋 로드 및 DataLoader 생성
    train_dataset = load_coco_dataset(split="train")
    train_dataset = preprocess_coco_dataset_labels(train_dataset)
    batch_size = 32
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모든 텍스트 후보 생성
    all_texts = list(set([example["text"] for example in train_dataset]))


    # Teacher 모델 결과 생성
    print("Generating output from Teacher model...")
    teacher_output = generate_text_from_image(teacher_model, processor, image, all_texts, device)

    # Student 모델 결과 생성
    print("Generating output from Student model...")
    student_output = generate_text_from_image(student_model, processor, image, all_texts, device)

    # Distilled 모델 결과 생성 (선택 사항)
    distilled_output = None
    if distilled_model:
        print("Generating output from Distilled model...")
        distilled_output = generate_text_from_image(distilled_model, processor, image, all_texts, device)

    # 결과 출력
    print("\n=== Model Outputs ===")
    print(f"Teacher Model: {teacher_output}")
    print(f"Student Model: {student_output}")
    if distilled_model:
        print(f"Distilled Model: {distilled_output}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
