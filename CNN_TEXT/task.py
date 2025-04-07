import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
import numpy as np
from collections import Counter
from flwr.common.logger import log
from logging import INFO
import warnings
from collections import OrderedDict
import json
from datetime import datetime
from pathlib import Path
from flwr.common.typing import UserConfig

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

fds = None
global_vocab = None

def build_global_vocab(num_partitions: int):
    global fds, global_vocab
    if fds is None:
        # partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=1.0,
            seed=42,
            min_partition_size=10,
        )
        fds = FederatedDataset(dataset="notaphoenix/shakespeare_dataset", partitioners={"training": partitioner})
    
    if global_vocab is None:
        all_words = []
        for partition_id in range(num_partitions):
            partition = fds.load_partition(partition_id=partition_id)
            all_words.extend(word for text in partition["text"] for word in text.split())
        word_counts = Counter(all_words)
        global_vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common())}
        global_vocab["<PAD>"] = 0
        log(INFO, f"Global vocab size: {len(global_vocab)}")

def load_data(partition_id: int, num_partitions: int, max_length: int):
    global fds, global_vocab
    if global_vocab is None:
        build_global_vocab(num_partitions)
    
    partition = fds.load_partition(partition_id=partition_id)

    def tokenize_and_pad(batch):
        sequences = [
            [global_vocab.get(word, 0) for word in text.split()] for text in batch["text"]
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

    partition = partition.map(tokenize_and_pad, batched=True)
    partition = partition.remove_columns("text")

    def apply_tensor(batch):
        batch["padded"] = torch.tensor(batch["padded"], dtype=torch.long)
        batch["label"] = torch.tensor(batch["label"], dtype=torch.float32)
        return batch

    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_tensor)

    trainloader = DataLoader(
        partition_train_test["train"], batch_size=32, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, valloader, len(global_vocab)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_size, max_length):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size
        )
        self.pool = nn.MaxPool1d(kernel_size=max_length - kernel_size + 1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x)
        x = x.squeeze(2)
        x = self.fc(x)
        return x

def train(net, trainloader, epochs: int, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-5)
    net.train()
    total_loss = 0.0
    for epoch in range(epochs):
        for batch in trainloader:
            optimizer.zero_grad()
            inputs = batch["padded"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    avg_loss = total_loss / len(trainloader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return avg_loss

def test(net, valloader, device):
    net.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in valloader:
            inputs = batch["padded"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = total_loss / len(valloader)
    print(f"Validation Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def create_run_dir(config: UserConfig) -> Path:
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)
    return save_path, run_dir

if __name__ == "__main__":
    for i in range(10):
        trainloader, valloader, vocab_size = load_data(i, 10, 512)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        net = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=256,
            num_filters=128,
            kernel_size=2,
            max_length=512,
        )
        net.to(device)
        train(net, trainloader, 10, device)
        test(net, valloader, device)