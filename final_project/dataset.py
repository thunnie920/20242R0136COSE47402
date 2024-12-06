from datasets import load_dataset
from torchvision import transforms

def load_coco_dataset(split="train"):
    """
    COCO 데이터셋을 로드합니다.
    Args:
        split (str): 로드할 dataset의 분할 ('train', 'validation', 'test').

    Returns:
        Dataset: Hugging Face dataset 객체.
    """
    try:
        dataset = load_dataset("JotDe/mscoco_15k", split=split)
        print(f"COCO dataset ({split}) loaded successfully.")
        return dataset
    except Exception as e:
        print(f"Error loading COCO dataset: {e}")
        return None


def preprocess_coco_dataset(dataset):
    """
    COCO 데이터셋 전처리 (이미지 텐서 변환 및 캡션 정리).
    Args:
        dataset (Dataset): COCO 데이터셋.

    Returns:
        Dataset: 전처리된 데이터셋 (PyTorch 텐서 형식 설정 포함).
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def preprocess(example):
        # PIL 이미지를 PyTorch 텐서로 변환
        example["image"] = transform(example["image"])
        return example

    # 병렬 처리 활성화
    dataset = dataset.select(range(1000))
    dataset = dataset.map(preprocess, batched=False, num_proc=4)  # 4개의 CPU 코어 사용
    dataset.set_format(type="torch", columns=["image", "text"])
    return dataset

# COCO 데이터셋 전처리 함수
def preprocess_coco_dataset_labels(dataset):
    """
    COCO 데이터셋 전처리 (이미지 텐서 변환 및 캡션 정리).
    Args:
        dataset (Dataset): COCO 데이터셋.

    Returns:
        Dataset: 전처리된 데이터셋 (PyTorch 텐서 형식 설정 포함).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
    ])

    def preprocess(example):
        """
        COCO 데이터셋의 이미지와 텍스트를 전처리합니다.
        Args:
            example: 데이터셋의 하나의 샘플.

        Returns:
            example: 전처리된 샘플.
        """
        example["image"] = transform(example["image"])
        return example

    # 병렬 처리 활성화
    dataset = dataset.select(range(1000))
    dataset = dataset.map(preprocess, batched=False, num_proc=4)  # 4개의 CPU 코어 사용
    dataset.set_format(type="torch", columns=["image", "text"])
    return dataset



def split_train_validation(dataset, validation_size=0.2):
    """
    학습 데이터에서 검증 데이터를 분리합니다.
    Args:
        dataset: 전체 학습 데이터셋.
        validation_size (float): 검증 데이터 비율 (예: 0.2 -> 20%).
    Returns:
        train_dataset, validation_dataset
    """
    dataset = dataset.train_test_split(test_size=validation_size, seed=42)
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Validation dataset size: {len(dataset['test'])}")
    return dataset['train'], dataset['test']