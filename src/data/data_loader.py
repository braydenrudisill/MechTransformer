from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit

from torch import tensor
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

from pathlib import Path
from os.path import isfile

from src.data.data_containers import InputDataset, Batch, EncodedBatch

def get_data_split(data_path: Path, *prefixes):
    return ({
        'train': [str(data_path / f'{prefix}-train.txt')],
        'val': [str(data_path / f'{prefix}-val.txt')]
    } for prefix in prefixes)


def load(data_path: Path, batch_size: int):
    assert isfile(token_file := data_path / 'tokenizer.json'), f"Could not find a tokenizer file in {data_path}"

    tokenizer: Tokenizer = Tokenizer.from_file(str(token_file))
    tokenizer.pre_tokenizer = WhitespaceSplit()  # noqa: Setting property
    tokenizer.enable_padding(pad_id=2)

    def collate_fn(batch: Batch):
        batch = default_collate(batch)
        encoded_src = tokenizer.encode_batch(batch.src_text)
        encoded_tgt = tokenizer.encode_batch(batch.tgt_text)
        src_sequence_ids = tensor([elem.ids for elem in encoded_src])
        tgt_sequence_ids = tensor([elem.ids for elem in encoded_tgt])
        src_attention_mask = tensor([elem.attention_mask for elem in encoded_src])
        tgt_attention_mask = tensor([elem.attention_mask for elem in encoded_tgt])
        encoded_batch = EncodedBatch(src_sequence_ids, src_attention_mask, tgt_sequence_ids, tgt_attention_mask)
        return encoded_batch

    src_files, tgt_files = get_data_split(data_path, 'src', 'tgt')

    src_dataset = load_dataset('text', data_files=src_files)
    tgt_dataset = load_dataset('text', data_files=tgt_files)

    def wrap_with_startend_tokens(x: dict):
        return {'text': f"[START] {x['text']} [END]"}

    src_dataset = src_dataset.map(wrap_with_startend_tokens)
    tgt_dataset = tgt_dataset.map(wrap_with_startend_tokens)

    train_dataset = InputDataset(src_dataset['train'], tgt_dataset['train'])
    val_dataset = InputDataset(src_dataset['val'], tgt_dataset['val'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, tokenizer




