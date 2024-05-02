import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.constants import ATTENTION_MASK, INPUT_IDS, LABEL


class ClaimDetectionDataset(Dataset):
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
        return len(self.labels)

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
            INPUT_IDS: tokenized_text[INPUT_IDS].flatten(),
            ATTENTION_MASK: tokenized_text[ATTENTION_MASK].flatten(),
            LABEL: torch.tensor(label),
        }
