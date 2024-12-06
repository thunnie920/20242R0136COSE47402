import torch
import time
import os
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader
from dataset import load_coco_dataset, preprocess_coco_dataset, split_train_validation


# 데이터셋 로드 및 전처리
def get_data_loaders(batch_size=32, validation_size=0.2):
    train_dataset = load_coco_dataset("train")
    train_dataset, val_dataset = split_train_validation(train_dataset, validation_size=validation_size)
    test_dataset = load_coco_dataset("test")

    train_dataset = preprocess_coco_dataset(train_dataset)
    val_dataset = preprocess_coco_dataset(val_dataset)
    test_dataset = preprocess_coco_dataset(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_model_i2t(model, dataloader, processor, device):
    """
    Image-to-Text (I2T) 평가: Recall@1, Recall@5 및 Latency 계산
    """
    model.eval()
    total_recall_at_1 = 0
    total_recall_at_5 = 0
    total_samples = 0
    total_latency = 0

    with torch.no_grad():
        for batch in dataloader:
            start_time = time.time()

            images = batch["image"].to(device)
            texts = batch["text"]

            inputs = processor(text=texts, images=None, return_tensors="pt", padding=True).to(device)
            text_features = model.get_text_features(**inputs)
            image_features = model.get_image_features(images)

            similarities = torch.matmul(image_features, text_features.T)

            end_time = time.time()
            total_latency += (end_time - start_time)

            for i, similarity in enumerate(similarities):
                ranked_indices = torch.argsort(similarity, descending=True)
                total_samples += 1

                if i in ranked_indices[:1]:  # R@1
                    total_recall_at_1 += 1
                if i in ranked_indices[:5]:  # R@5
                    total_recall_at_5 += 1

    recall_at_1 = total_recall_at_1 / total_samples
    recall_at_5 = total_recall_at_5 / total_samples
    avg_latency_per_batch = (total_latency / len(dataloader)) * 1000  # ms 단위

    return recall_at_1, recall_at_5, avg_latency_per_batch

def evaluate_model_t2i(model, dataloader, processor, device):
    """
    Text-to-Image (T2I) 평가: Recall@1, Recall@5 및 Latency 계산
    """
    model.eval()
    total_recall_at_1 = 0
    total_recall_at_5 = 0
    total_samples = 0
    total_latency = 0

    with torch.no_grad():
        for batch in dataloader:
            start_time = time.time()

            images = batch["image"].to(device)
            texts = batch["text"]

            inputs = processor(text=texts, images=None, return_tensors="pt", padding=True).to(device)
            text_features = model.get_text_features(**inputs)
            image_features = model.get_image_features(images)

            similarities = torch.matmul(text_features, image_features.T)

            end_time = time.time()
            total_latency += (end_time - start_time)

            for i, similarity in enumerate(similarities):
                ranked_indices = torch.argsort(similarity, descending=True)
                total_samples += 1

                if i in ranked_indices[:1]:  # R@1
                    total_recall_at_1 += 1
                if i in ranked_indices[:5]:  # R@5
                    total_recall_at_5 += 1

    recall_at_1 = total_recall_at_1 / total_samples
    recall_at_5 = total_recall_at_5 / total_samples
    avg_latency_per_batch = (total_latency / len(dataloader)) * 1000  # ms 단위

    return recall_at_1, recall_at_5, avg_latency_per_batch

def get_model_size(model):
    """
    모델 크기 계산 (MB 단위)
    """
    temp_path = "temp_model.bin"
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_mb

