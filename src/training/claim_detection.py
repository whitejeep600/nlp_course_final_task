from pathlib import Path

import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.constants import EVAL, TRAIN
from src.datasets.claim_detection_dataset import ClaimDetectionDataset
from src.models.classification_bert import ClassificationBert
from src.training.trainer import Trainer
from src.utils import get_cuda_device_if_available


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
        model,
        train_dataloader,
        dev_dataloader,
        device,
        loss_function,
        optimizer,
        plots_save_path,
        "claim detection",
    )

    for _ in tqdm(range(n_epochs), desc="Training", position=0):

        trainer.iteration(TRAIN)
        with torch.no_grad():
            trainer.iteration(EVAL)

    trainer.save_plots()


if __name__ == "__main__":

    params = yaml.safe_load(open("params.yaml"))["src.training.claim_detection_task2"]

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
