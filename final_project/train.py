import torch
import sys
import torch.nn.functional as F
from torch.nn import KLDivLoss
from tqdm import tqdm
from transformers import CLIPProcessor
from torch.utils.data import DataLoader

from teacher_generation import generate_soft_labels_log_probabilities
from student_generation import generate_predictions

from models import load_teacher_model, load_student_model
from dataset import load_coco_dataset, split_train_validation, preprocess_coco_dataset_labels
from utils import evaluate_model_i2t, evaluate_model_t2i, get_model_size


def train_student_model(student_model, teacher_model, data_loader, processor, all_texts, alpha=0.5, temperature=5.0, optimizer=None, device='cpu'):
    student_model.train()
    teacher_model.eval()  # Teacher 모델은 업데이트되지 않으므로 eval 모드로 설정
    total_loss = 0.0
    total_samples = 0

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_kd = KLDivLoss(reduction="batchmean", log_target=True)

    # 텍스트 -> 인덱스 매핑
    text_to_index = {text: idx for idx, text in enumerate(all_texts)}

    # 텍스트 임베딩 미리 계산 (학습 시 gradient가 필요 없으므로 with torch.no_grad() 사용)
    with torch.no_grad():
        text_inputs = processor(
            text=all_texts,
            images=None,
            return_tensors="pt",
            padding=True
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        teacher_text_embeddings = teacher_model.get_text_features(**text_inputs)  # [N, D]
        student_text_embeddings = student_model.get_text_features(**text_inputs)  # [N, D]

    for batch in tqdm(data_loader, desc="Training Student Model"):
        images = batch['image'].to(device)
        ground_truth_texts = batch['text']

        # Ground Truth 인덱스
        ground_truth_indices = [text_to_index[text] for text in ground_truth_texts]
        ground_truth_indices = torch.tensor(ground_truth_indices).to(device)

        # 이미지 입력 처리
        image_inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

        # Student 모델의 이미지 임베딩 계산
        student_image_embeddings = student_model.get_image_features(**image_inputs)  # [batch_size, D]

        # --- Student Loss 계산 (Cross-Entropy Loss) ---
        # 유사도 계산 (logits)
        student_logits_ce = torch.matmul(student_image_embeddings, student_text_embeddings.T)  # [batch_size, N]

        # Student Loss 계산
        student_loss = criterion_ce(student_logits_ce, ground_truth_indices)
        # ---------------------------------------------

        #  --- Distillation Loss 계산 ---
        # Student 모델의 logits에 온도 적용 및 로그 확률 계산
        student_logits = student_logits_ce / temperature  # [batch_size, N]
        student_log_probs = F.log_softmax(student_logits, dim=-1)  # [batch_size, N]

        # Teacher 모델의 로그 확률 계산 (gradient 불필요)
        with torch.no_grad():
            teacher_image_embeddings = teacher_model.get_image_features(**image_inputs)  # [batch_size, D]
            teacher_logits = torch.matmul(teacher_image_embeddings, teacher_text_embeddings.T) / temperature  # [batch_size, N]
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        # Distillation Loss 계산
        distillation_loss = criterion_kd(student_log_probs, teacher_log_probs)
        # ------------------------------

        # 최종 손실 계산
        final_loss = alpha * distillation_loss + (1 - alpha) * student_loss

        # 역전파 및 모델 업데이트
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        total_loss += final_loss.item() * images.size(0)
        total_samples += images.size(0)

    average_loss = total_loss / total_samples
    return average_loss

if __name__ == "__main__":
    # 모델 로드 및 설정
    teacher_model = load_teacher_model().to("cpu")
    teacher_model.eval()
    student_model = load_student_model(quantized=False).to("cpu")
    student_model.train()

    # 데이터셋 로드 및 DataLoader 생성
    train_dataset = load_coco_dataset(split="train")
    train_dataset = preprocess_coco_dataset_labels(train_dataset)
    batch_size = 32
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모든 텍스트 후보 생성
    all_texts = list(set([example["text"] for example in train_dataset]))

    # Processor 초기화
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 옵티마이저 설정
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)

    # 학습 루프
    alpha = 0.25
    num_epochs = 5
    for epoch in range(num_epochs):
        average_loss = train_student_model(
            student_model=student_model,
            teacher_model=teacher_model,
            data_loader=data_loader,
            processor=processor,
            all_texts=all_texts,
            alpha=alpha,
            temperature=3.0,
            optimizer=optimizer,
            device="cpu"
        )
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")

# 학습된 모델 저장 후 양자화 적용
student_model.eval()  # 양자화 전 eval 모드로 전환
torch.backends.quantized.engine = "qnnpack"
student_model_quantized = torch.quantization.quantize_dynamic(
    student_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)
print("Quantization applied to the student model.")


# 학습된 모델 저장
model_save_path = '/Users/seoeun/Desktop/DeepLearning/final_project/final_model.pth'
torch.save(student_model.state_dict(), model_save_path)
print(f"Trained model saved to '{model_save_path}'")