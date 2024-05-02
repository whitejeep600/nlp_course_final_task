from pathlib import Path

import torch
from matplotlib import pyplot as plt
from numpy import mean
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import ACCURACY, EVAL, LOSS, METRICS, MODES, TRAIN


class Trainer:

    def __init__(
        self,
        model: Module,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        device: torch.device,
        loss_function: _Loss,
        optimizer: Optimizer,
        plots_save_path: Path,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.plots_save_path = plots_save_path
        self.epochs_elapsed = 0
        self.metrics: dict[str, dict[str, list[float]]] = {
            mode: {metric: [] for metric in METRICS} for mode in MODES
        }

    def iteration(
        self,
        mode: str,
    ) -> None:
        if mode == TRAIN:
            self.model.train()
            dataloader = self.train_dataloader
        elif mode == EVAL:
            self.model.eval()
            dataloader = self.dev_dataloader
        else:
            raise ValueError(f"Unsupported mode {mode}, expected one of {MODES}")

        batch_losses: list[float] = []
        n_correctly_classified_samples = 0

        for batch_no, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"{mode} iteration",
            leave=False,
            position=1,
        ):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            token_type_ids = (
                batch["token_type_ids"].to(self.device) if "token_type_ids" in batch else None
            )
            output_logits = self.model(input_ids, attention_mask, token_type_ids)
            loss = self.loss_function(output_logits, labels)

            batch_losses.append(loss.item())
            n_correctly_classified_samples += (
                (torch.argmax(output_logits, dim=1) == labels).sum().item()
            )

            if mode == TRAIN:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        mean_loss = float(mean(batch_losses))
        n_all_samples = len(dataloader) * dataloader.batch_size  # type: ignore
        accuracy = n_correctly_classified_samples / n_all_samples

        print(
            f"\nEpoch {self.epochs_elapsed}, {mode} loss: {mean_loss}, accuracy: {accuracy:.4f}\n"
        )

        self.metrics[mode][LOSS].append(mean_loss)
        self.metrics[mode][ACCURACY].append(accuracy)

        if mode == EVAL:
            self.epochs_elapsed += 1

    def save_plots(self) -> None:
        self.plots_save_path.mkdir(exist_ok=True, parents=True)
        for mode in MODES:
            for metric in METRICS:
                save_path = self.plots_save_path / f"{mode}_{metric}.png"
                title = f"{mode} {metric}"
                ys = self.metrics[mode][metric]
                xs = range(self.epochs_elapsed)
                plt.title(title)
                plt.plot(xs, ys, linewidth=0.5)
                plt.xlabel("Iteration")
                plt.ylabel(metric)
                plt.savefig(save_path, dpi=256)
                plt.clf()
