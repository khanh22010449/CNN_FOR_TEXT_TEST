"""pytorch: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, RecordSet, array_from_numpy
from CNN_TEXT.task import TextCNN, get_weights, load_data, set_weights, train, test
from flwr.common.logger import log
from logging import INFO

class FlowerClient(NumPyClient):
    def __init__(self, model, client_state: RecordSet, trainloader, valloader, local_epochs):
        self.model = model
        self.client_state = client_state
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.local_layer_name = "classification-head"

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        self._load_layer_weights_from_state()
        train_loss = train(self.model, self.trainloader, self.local_epochs, self.device)
        self._save_layer_weights_to_state()
        return get_weights(self.model), len(self.trainloader), {"train_loss": train_loss}

    def _save_layer_weights_to_state(self):
        state_dict_arrays = {}
        for k, v in self.model.fc.state_dict().items():
            state_dict_arrays[k] = array_from_numpy(v.cpu().numpy())
        self.client_state.parameters_records[self.local_layer_name] = ParametersRecord(state_dict_arrays)
"""pytorch: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, RecordSet, array_from_numpy
from CNN_TEXT.task import TextCNN, get_weights, load_data, set_weights, train, test
from flwr.common.logger import log
from logging import INFO
import toml  # Thêm import toml

def load_best_params() -> dict:
    """Đọc tham số tối ưu từ pyproject.toml."""
    with open("pyproject.toml", "r") as f:
        config = toml.load(f)
    best_params = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {}).get("model-params")
    if not best_params:
        raise ValueError("Model parameters not found in pyproject.toml. Please ensure server has optimized them.")
    return best_params

class FlowerClient(NumPyClient):
    def __init__(self, model, client_state: RecordSet, trainloader, valloader, local_epochs):
        self.model = model
        self.client_state = client_state
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.local_layer_name = "classification-head"

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        self._load_layer_weights_from_state()
        train_loss = train(self.model, self.trainloader, self.local_epochs, self.device)
        self._save_layer_weights_to_state()
        return get_weights(self.model), len(self.trainloader), {"train_loss": train_loss}

    def _save_layer_weights_to_state(self):
        state_dict_arrays = {}
        for k, v in self.model.fc.state_dict().items():
            state_dict_arrays[k] = array_from_numpy(v.cpu().numpy())
        self.client_state.parameters_records[self.local_layer_name] = ParametersRecord(state_dict_arrays)

    def _load_layer_weights_from_state(self):
        if self.local_layer_name not in self.client_state.parameters_records:
            return
        state_dict = {}
        param_records = self.client_state.parameters_records
        for k, v in param_records[self.local_layer_name].items():
            state_dict[k] = torch.from_numpy(v.numpy())
        self.model.fc.load_state_dict(state_dict, strict=True)

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self._load_layer_weights_from_state()
        loss, accuracy = test(self.model, self.valloader, self.device)
        return loss, len(self.valloader), {"accuracy": accuracy}

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Đọc best_params từ pyproject.toml
    best_params = load_best_params()
    log(INFO, f"Client {partition_id} - Loaded best params: {best_params}")

    # Tải dữ liệu với max_length tối ưu
    trainloader, valloader, vocab_size = load_data(partition_id, num_partitions, best_params["max_length"])
    log(INFO, f"Client {partition_id} - Vocab size: {vocab_size}")

    # Khởi tạo mô hình với tham số tối ưu
    model = TextCNN(
        vocab_size=best_params["vocab_size"],
        embedding_dim=best_params["embedding_dim"],
        num_filters=best_params["num_filters"],
        kernel_size=best_params["kernel_size"],
        max_length=best_params["max_length"],
    )

    local_epochs = context.run_config["local-epochs"]
    client_state = context.state
    return FlowerClient(model, client_state, trainloader, valloader, local_epochs).to_client()

app = ClientApp(client_fn)
    def _load_layer_weights_from_state(self):
        if self.local_layer_name not in self.client_state.parameters_records:
            return
        state_dict = {}
        param_records = self.client_state.parameters_records
        for k, v in param_records[self.local_layer_name].items():
            state_dict[k] = torch.from_numpy(v.numpy())
        self.model.fc.load_state_dict(state_dict, strict=True)

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self._load_layer_weights_from_state()
        loss, accuracy = test(self.model, self.valloader, self.device)
        return loss, len(self.valloader), {"accuracy": accuracy}

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, vocab_size = load_data(partition_id, num_partitions, 512)
    log(INFO, f"Client {partition_id} - Vocab size: {vocab_size}")

    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=256,  # Giá trị mặc định, sẽ được server tối ưu sau
        num_filters=128,
        kernel_size=2,
        max_length=512,
    )

    local_epochs = context.run_config["local-epochs"]
    client_state = context.state
    return FlowerClient(model, client_state, trainloader, valloader, local_epochs).to_client()

app = ClientApp(client_fn)