"""pytorch-example: A Flower / PyTorch app."""

import json
from logging import INFO
import torch
import wandb
from CNN_TEXT.task import TextCNN, create_run_dir, set_weights
from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg

PROJECT_NAME = "CNN for TEXT - shakespeare dataset - DirichletPartitioner"

class CustomFedAvg(FedAvg):
    def __init__(self, run_config: UserConfig, use_wandb: bool, model_params: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.use_wandb = use_wandb
        self.model_params = model_params  # Tham số mô hình từ server
        if use_wandb:
            self._init_wandb_project()
        self.best_acc_so_far = 0.0
        self.results = {}

    def _init_wandb_project(self):
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, results_dict):
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, round, accuracy, parameters):
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            ndarrays = parameters_to_ndarrays(parameters)
            model = TextCNN(
                vocab_size=self.model_params["vocab_size"],
                embedding_dim=self.model_params["embedding_dim"],
                num_filters=self.model_params["num_filters"],
                kernel_size=self.model_params["kernel_size"],
                max_length=self.model_params["max_length"],
            )
            set_weights(model, ndarrays)
            file_name = f"model_state_acc_{accuracy}_round_{round}.pth"
            torch.save(model.state_dict(), self.save_path / file_name)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        self._store_results(tag=tag, results_dict={"round": server_round, **results_dict})
        if self.use_wandb:
            wandb.log(results_dict, step=server_round)

    def evaluate(self, server_round, parameters):
        loss, metrics = super().evaluate(server_round, parameters)
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics