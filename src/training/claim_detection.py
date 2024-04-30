from pathlib import Path

import torch
import yaml
from numpy import mean
from torch.nn import CrossEntropyLoss, Module
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.constants import EVAL, MODES, TRAIN
from src.datasets.claim_detection_dataset import ClaimDetectionDataset
from src.models.classification_bert import ClassificationBert
from src.utils import get_cuda_device_if_available


def iteration(
    epoch_number: int,
    mode: str,
    model: Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_function: _Loss,
    optimizer: Optimizer | None = None,
) -> None:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode {mode}, expected one of {MODES}")
    if mode == TRAIN:
        model.train()
    elif mode == EVAL:
        model.eval()

    batch_losses: list[float] = []
    n_correctly_classified_samples = 0

    for batch_no, batch in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"{mode} iteration",
        leave=False,
        position=1,
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        output_logits = model(input_ids, attention_mask)
        loss = loss_function(output_logits, labels)

        batch_losses.append(loss.item())
        n_correctly_classified_samples += (
            (torch.argmax(output_logits, dim=1) == labels).sum().item()
        )

        if mode == TRAIN:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    mean_loss = float(mean(batch_losses))
    n_all_samples = len(dataloader) * dataloader.batch_size
    accuracy = n_correctly_classified_samples / n_all_samples
    print(f"Epoch {epoch_number}, {mode} loss: {mean_loss}, accuracy: {accuracy:.4f}")


def main(
    train_set_json_path: Path,
    dev_set_json_path: Path,
    tuned_model_name: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
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

    for epoch_number in tqdm(range(n_epochs), desc="Training", position=0):

        iteration(epoch_number, TRAIN, model, train_dataloader, device, loss_function, optimizer)
        with torch.no_grad():
            iteration(epoch_number, EVAL, model, dev_dataloader, device, loss_function)


if __name__ == "__main__":

    params = yaml.safe_load(open("params.yaml"))["src.training.claim_detection"]

    train_set_json_path = Path(params["train_set_json_path"])
    dev_set_json_path = Path(params["dev_set_json_path"])
    tuned_model_name = params["tuned_model_name"]
    n_epochs = int(params["n_epochs"])
    batch_size = int(params["batch_size"])
    lr = float(params["lr"])

    main(
        train_set_json_path=train_set_json_path,
        dev_set_json_path=dev_set_json_path,
        tuned_model_name=tuned_model_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
    )
