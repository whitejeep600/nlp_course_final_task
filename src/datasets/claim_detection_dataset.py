import json
from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

from src.datasets.text_classification_dataset import TextClassificationDataset


class ClaimDetectionDataset(TextClassificationDataset):
    def __init__(
        self,
        json_path: Path,
        tokenizer: PreTrainedTokenizer,
    ):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.texts: list[str] = [sample[0] for sample in data]
        self.labels: list[int] = [sample[1] for sample in data]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text = self.texts[i]
        label = self.labels[i]
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": tokenized_text["input_ids"].flatten(),
            "attention_mask": tokenized_text["attention_mask"].flatten(),
            "label": torch.tensor(label),
        }
