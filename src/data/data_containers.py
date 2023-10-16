import torch
from torch import Tensor
from dataclasses import dataclass
from typing import NamedTuple


# TODO: restructure Batch and EncodedBatch so that collate_fn is the right type
class Batch(NamedTuple):
    src_text: list[str]
    tgt_text: list[str]


class EncodedBatch(NamedTuple):
    src_sequence_ids: Tensor
    src_attention_mask: Tensor
    tgt_sequence_ids: Tensor
    tgt_attention_mask: Tensor


class InputDataset(torch.utils.data.Dataset):
    def __init__(self, train_data, target_data):
        self.train_data = train_data
        self.target_data = target_data

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, index):
        return Batch(
            src_text=self.train_data[index]["text"],
            tgt_text=self.target_data[index]["text"]
        )
