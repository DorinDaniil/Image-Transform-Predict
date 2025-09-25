# main.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from model.predictor import ImageTransformPredictor
from model.tokenizer import TransformTokenizer
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass

# ----------------------------
# 1. СИНТЕТИЧЕСКИЙ ДАТАСЕТ
# ----------------------------
class SyntheticImageTransformDataset(Dataset):
    def __init__(self, n_samples=1000, img_size=(224, 224), max_seq_len=5, noop_prob=0.3):
        self.n_samples = n_samples
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.noop_prob = noop_prob

        self.transforms = [
            "noop",
            "grayscale", "rotate_90", "rotate_180", "rotate_270",
            "color_jitter", "noise_adding", "crop",
            "horizontal_flip", "vertical_flip"
        ]
        self.transform_to_id = {t: i + 3 for i, t in enumerate(self.transforms)}  # 3 = noop ID

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        is_noop = np.random.rand() < self.noop_prob

        if is_noop:
            # Одинаковые изображения
            base_img = torch.randn(3, *self.img_size)
            img1 = base_img.clone()
            img2 = base_img.clone()
            token_ids = [1, 3, 2]  # [START], noop, [END]
        else:
            # Разные изображения
            img1 = torch.randn(3, *self.img_size)
            img2 = torch.randn(3, *self.img_size)
            seq_len = np.random.randint(1, self.max_seq_len + 1)
            seq = np.random.choice(self.transforms, size=seq_len, replace=True).tolist()
            token_ids = [1] + [self.transform_to_id[t] for t in seq] + [2]

        # Паддинг до 10
        while len(token_ids) < 10:
            token_ids.append(0)

        return {
            "image_batch_1": img1,
            "image_batch_2": img2,
            "target_tokens": torch.tensor(token_ids, dtype=torch.long),
        }

# ----------------------------
# 2. DATA COLLATOR
# ----------------------------
@dataclass
class ImageTransformDataCollator:
    def __call__(self, examples: list) -> dict:
        return {
            "image_batch_1": torch.stack([ex["image_batch_1"] for ex in examples]),
            "image_batch_2": torch.stack([ex["image_batch_2"] for ex in examples]),
            "target_tokens": torch.stack([ex["target_tokens"] for ex in examples]),
        }

# ----------------------------
# 3. ЗАПУСК
# ----------------------------
if __name__ == "__main__":
    # Инициализация модели
    model = ImageTransformPredictor(
        embedding_dim=512,
        num_heads=8,
        dim_feedforward=1024,
        num_layers=3,
        max_seq_length=10,
        freeze_image_encoder=True,
        unfreeze_n_layers=2,
    )

    # Токенизатор (для декодирования)
    tokenizer = TransformTokenizer()

    # Датасет
    dataset = SyntheticImageTransformDataset(n_samples=200, noop_prob=0.3)

    # Коллатор
    collator = ImageTransformDataCollator()

    # Training args
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=15,
        learning_rate=3e-4,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # Тренировщик
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("🚀 Начало обучения...")
    trainer.train()

    print("\n💾 Сохранение модели...")
    model.save_pretrained("./final_model")

    print("\n🧪 Загрузка и тестирование генерации...")
    loaded_model = ImageTransformPredictor.from_pretrained("./final_model")

    # Генерация на примере
    dummy_img = torch.randn(1, 3, 224, 224)
    generated_ids = loaded_model.generate(dummy_img, dummy_img, max_length=10)
    tokens = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("🔮 Предсказание:", tokens)