import os
import sys
from pathlib import Path
import torch
import yaml
import csv
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  #get the "src" directory
sys.path.append(grandparent_dir)
from src.constants import EVAL, TRAIN
from src.datasets.argument_relation_detection_dataset import ArgumentRelationDetectionDataset
from src.models.classification_bert import ClassificationBert
from src.training.trainer import Trainer
from src.utils import get_cuda_device_if_available


def main(
    train_set_json_path: Path,
    dev_set_json_path: Path,
    test_set_json_path: Path,
    test_set_output_csv_path: Path,
    tuned_model_name: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    plots_save_path: Path,
) -> None:
    device = get_cuda_device_if_available()

    model = ClassificationBert(tuned_model_name, 3, device)
    tokenizer = AutoTokenizer.from_pretrained(tuned_model_name)

    train_dataset = ArgumentRelationDetectionDataset(train_set_json_path, tokenizer, test_mode=False)
    dev_dataset = ArgumentRelationDetectionDataset(dev_set_json_path, tokenizer, test_mode=False)
    test_dataset = ArgumentRelationDetectionDataset(test_set_json_path, tokenizer, test_mode=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
        "argument relation detection",
    )

    for _ in tqdm(range(n_epochs), desc="Training", position=0):

        trainer.iteration(TRAIN)
        with torch.no_grad():
            trainer.iteration(EVAL)

    trainer.save_plots()

    ####Start to predict####

    trainer.model.eval()

    test_predictions = []

    for batch in tqdm(test_dataloader, desc="Testing", position=0):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.no_grad():
            output_logits = trainer.model(input_ids, attention_mask, token_type_ids)

        probabilities = torch.softmax(output_logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        test_predictions.extend(predicted_classes.cpu().tolist())

    ####Output the prediction in .csv format####
    
    #print(test_predictions)

    output_csv_file = Path(test_set_output_csv_path) / "predictions.csv"
    output_csv_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_file, mode='w', newline='') as file:
        fieldnames = ['ID', 'y_pred']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for i, prediction in enumerate(test_predictions, start=1):
            writer.writerow({"ID": i, "y_pred": prediction})

if __name__ == "__main__":

    ####                                                            ####
    ####    change the "config_name" to get the different settings  ####
    ####                                                            ####
    
    config_name = "src.training.argument_relation_detection_sec-bert-base"
    params = yaml.safe_load(open("default_params.yaml"))[config_name]

    train_set_json_path = Path(params["train_set_json_path"])
    dev_set_json_path = Path(params["dev_set_json_path"])
    test_set_json_path = Path(params["test_set_json_path"])
    test_set_output_csv_path = Path(params["test_set_output_csv_path"])
    tuned_model_name = params["tuned_model_name"]
    n_epochs = int(params["n_epochs"])
    batch_size = int(params["batch_size"])
    lr = float(params["lr"])
    plots_save_path = Path(params["plots_save_path"])

    main(
        train_set_json_path=train_set_json_path,
        dev_set_json_path=dev_set_json_path,
        test_set_json_path=test_set_json_path,
        test_set_output_csv_path=test_set_output_csv_path,
        tuned_model_name=tuned_model_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        plots_save_path=plots_save_path,
    )

    