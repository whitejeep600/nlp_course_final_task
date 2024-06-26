import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.constants import ATTENTION_MASK, INPUT_IDS, LABEL, TOKEN_TYPE_IDS


class ArgumentRelationDetectionDataset(Dataset):
    def __init__(self, json_path: Path, tokenizer: PreTrainedTokenizer, task: str, max_length: int = 256):
        with open(json_path, "r") as f:
            data = json.load(f)
        if task == "task1":  # The dataset format is different between task1 and task2
            texts: list[tuple[str, str]] = [(sample[1], sample[2]) for sample in data]
            labels: list[int] | None = (
                [sample[3] for sample in data] if len(data) > 0 and len(data[0]) > 3 else None
            )
        elif task == "task2":
            texts = [(sample[0], sample[1]) for sample in data]
            labels = [sample[2] for sample in data]
        else:
            raise ValueError("You should specify task1 or task2.")

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def is_test_mode(self) -> bool:
        return self.labels is None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        texts = self.texts[i]
        label = self.labels[i] if not self.is_test_mode() else None
        tokenized_text = self.tokenizer(
            *texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        data_dict = {
            INPUT_IDS: tokenized_text[INPUT_IDS].flatten(),
            ATTENTION_MASK: tokenized_text[ATTENTION_MASK].flatten(),
            TOKEN_TYPE_IDS: tokenized_text[TOKEN_TYPE_IDS].flatten(),
        }
        if not self.is_test_mode():
            data_dict[LABEL] = torch.tensor(label)

        return data_dict
