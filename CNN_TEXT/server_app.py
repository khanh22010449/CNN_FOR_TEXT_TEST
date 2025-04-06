"""pytorch: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAdagrad, FedAdam, FedAvg
from typing import List, Tuple
from CNN_TEXT.task import TextCNN, get_weights, set_weights, test, train
from CNN_TEXT.strategy import CustomFedAvg
from CNN_TEXT.test import load_data
import torch
from torch.utils.data import DataLoader
import itertools
import time
import toml  # Thêm import toml để xử lý file .toml
import os

def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function '{func.__name__}' took {elapsed_time:.2f} seconds to execute.")
    return result, elapsed_time

def find_best_params(device: torch.device, epochs: int = 5) -> dict:
    param_grid = {
        'embedding_dim': [64, 128, 256],
        'num_filters': [64, 128, 256],
        'kernel_size': [2, 3, 4],
        'max_length': [256, 512, 1024]
    }

    best_accuracy = 0.0
    best_params = None

    for embedding_dim, num_filters, kernel_size, max_length in itertools.product(
        param_grid['embedding_dim'], param_grid['num_filters'], 
        param_grid['kernel_size'], param_grid['max_length']
    ):
        print(f"Testing params: embedding_dim={embedding_dim}, num_filters={num_filters}, "
              f"kernel_size={kernel_size}, max_length={max_length}")
        
        trainloader, valloader, vocab_size = load_data(max_length)
        
        net = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            max_length=max_length
        )
        net.to(device)

        (avg_loss, _) = measure_time(train, net, trainloader, epochs, device)
        (_, accuracy), _ = measure_time(test, net, valloader, device)
        print(f"Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'num_filters': num_filters,
                'kernel_size': kernel_size,
                'max_length': max_length
            }
    
    print(f"Best params: {best_params}, Best accuracy: {best_accuracy:.4f}")
    
    # Cập nhật pyproject.toml với tham số tối ưu
    with open("pyproject.toml", "r") as f:
        config = toml.load(f)
    
    # Thêm hoặc cập nhật phần [tool.flwr.app.config.model-params]
    if "tool" not in config:
        config["tool"] = {}
    if "flwr" not in config["tool"]:
        config["tool"]["flwr"] = {}
    if "app" not in config["tool"]["flwr"]:
        config["tool"]["flwr"]["app"] = {}
    if "config" not in config["tool"]["flwr"]["app"]:
        config["tool"]["flwr"]["app"]["config"] = {}
    
    config["tool"]["flwr"]["app"]["config"]["model-params"] = best_params
    
    with open("pyproject.toml", "w") as f:
        toml.dump(config, f)
    print("Saved best parameters to pyproject.toml under [tool.flwr.app.config.model-params]")
    
    return best_params

def load_best_params() -> dict:
    """Đọc tham số tối ưu từ pyproject.toml nếu tồn tại."""
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "r") as f:
            config = toml.load(f)
        best_params = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {}).get("model-params")
        if best_params:
            print("Loaded best parameters from pyproject.toml")
            return best_params
    raise FileNotFoundError("Model parameters not found in pyproject.toml. Please run parameter optimization first.")

def gen_evaluate_fn(testloader: DataLoader, device: torch.device, config: dict):
    def evaluate(server_round, parameters_ndarrays, config_dict):
        net = TextCNN(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_filters=config["num_filters"],
            kernel_size=config["kernel_size"],
            max_length=config["max_length"],
        )
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        return loss, {"centralized_accuracy": accuracy}
    return evaluate

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def on_fit_config(server_round: int):
    lr = 0.1
    if server_round > 5:
        lr /= 2
    return {"lr": lr}

def server_fn(context: Context):
    device = context.run_config["server-device"]
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Kiểm tra nếu đã có model-params trong pyproject.toml thì đọc, nếu không thì tìm mới
    try:
        best_params = load_best_params()
    except FileNotFoundError:
        best_params = find_best_params(device, epochs=5)

    # Tải testloader với max_length tối ưu
    trainloader, testloader, _ = load_data(best_params["max_length"])

    # Khởi tạo tham số mô hình
    ndarrays = get_weights(
        TextCNN(
            vocab_size=best_params["vocab_size"],
            embedding_dim=best_params["embedding_dim"],
            num_filters=best_params["num_filters"],
            kernel_size=best_params["kernel_size"],
            max_length=best_params["max_length"],
        )
    )
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = CustomFedAvg(
        run_config=context.run_config,
        use_wandb=context.run_config.get("use-wandb", False),
        model_params=best_params,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.5,
        min_available_clients=2,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=gen_evaluate_fn(testloader, device=device, config=best_params),
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)