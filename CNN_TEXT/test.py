import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
from datasets import load_dataset
from flwr.common.logger import log
from logging import INFO
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def load_data(max_length: int):
    # Tải dữ liệu từ dataset
    dataset = load_dataset("notaphoenix/shakespeare_dataset")
    train_data = dataset["training"]
    test_data = dataset["test"]

    # Xây dựng vocabulary từ tập train
    word_counts = Counter(word for text in train_data["text"] for word in text.split())
    vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common())}
    vocab["<PAD>"] = 0  # Token padding
    vocab_size = len(vocab)  # Lấy vocab_size từ tập train
    log(INFO, f"Vocab size from train dataset: {vocab_size}")

    def tokenize_and_pad(batch):
        sequences = [
            [vocab.get(word, 0) for word in text.split()] for text in batch["text"]
        ]
        padded_sequences = [
            (
                seq + [0] * (max_length - len(seq))
                if len(seq) < max_length
                else seq[:max_length]
            )
            for seq in sequences
        ]
        batch["padded"] = np.array(padded_sequences, dtype=np.int64)
        return batch

    # Áp dụng tokenization và padding
    train_data = train_data.map(tokenize_and_pad, batched=True).remove_columns("text")
    test_data = test_data.map(tokenize_and_pad, batched=True).remove_columns("text")

    # Chuyển thành tensor
    def apply_tensor(batch):
        batch["padded"] = torch.tensor(batch["padded"], dtype=torch.long)
        batch["label"] = torch.tensor(batch["label"], dtype=torch.float32)
        return batch

    train_data = train_data.with_transform(apply_tensor)
    test_split = test_data.train_test_split(test_size=0.2, seed=42)
    test_split = test_split.with_transform(apply_tensor)

    # Tạo DataLoader
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)
    valloader = DataLoader(test_split["test"], batch_size=32, num_workers=8)
    return trainloader, valloader, vocab_size

if __name__ == "__main__":
    trainloader, valloader, vocab_size = load_data(512)
    print(f"Vocab size: {vocab_size}, Train and Val DataLoaders ready")