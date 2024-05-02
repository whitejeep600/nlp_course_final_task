import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ArgumentRelationDetectionDataset(Dataset):
    def __init__(
        self,
        json_path: Path,
        tokenizer: PreTrainedTokenizer,
    ):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.texts: list[tuple[str, str]] = [(sample[0], sample[1]) for sample in data]
        self.labels: list[int] = [sample[2] for sample in data]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        texts = self.texts[i]
        label = self.labels[i]
        tokenized_text = self.tokenizer(
            *texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": tokenized_text["input_ids"].flatten(),
            "attention_mask": tokenized_text["attention_mask"].flatten(),
            "token_type_ids": tokenized_text["token_type_ids"].flatten(),
            "label": torch.tensor(label),
        }
