from pathlib import Path

import torch
import yaml
from matplotlib import pyplot as plt
from numpy import mean
from torch.nn import CrossEntropyLoss, Module
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.constants import ACCURACY, EVAL, LOSS, METRICS, MODES, TRAIN
from src.datasets.claim_detection_dataset import ClaimDetectionDataset
from src.models.classification_bert import ClassificationBert
from src.utils import get_cuda_device_if_available


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
            output_logits = self.model(input_ids, attention_mask)
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


def main(
    train_set_json_path: Path,
    dev_set_json_path: Path,
    tuned_model_name: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    plots_save_path: Path,
) -> None:
    device = get_cuda_device_if_available()

    model = ClassificationBert(tuned_model_name, 2, device)
    tokenizer = AutoTokenizer.from_pretrained(tuned_model_name)

    train_dataset = ClaimDetectionDataset(train_set_json_path, tokenizer)
    dev_dataset = ClaimDetectionDataset(dev_set_json_path, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

    loss_function = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    trainer = Trainer(
        model, train_dataloader, dev_dataloader, device, loss_function, optimizer, plots_save_path
    )

    for _ in tqdm(range(n_epochs), desc="Training", position=0):

        trainer.iteration(TRAIN)
        with torch.no_grad():
            trainer.iteration(EVAL)

    trainer.save_plots()


if __name__ == "__main__":

    params = yaml.safe_load(open("params.yaml"))["src.training.claim_detection"]

    train_set_json_path = Path(params["train_set_json_path"])
    dev_set_json_path = Path(params["dev_set_json_path"])
    tuned_model_name = params["tuned_model_name"]
    n_epochs = int(params["n_epochs"])
    batch_size = int(params["batch_size"])
    lr = float(params["lr"])
    plots_save_path = Path(params["plots_save_path"])

    main(
        train_set_json_path=train_set_json_path,
        dev_set_json_path=dev_set_json_path,
        tuned_model_name=tuned_model_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        plots_save_path=plots_save_path,
    )
