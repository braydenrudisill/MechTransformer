# %%
# Dataloading
# In the data_files={...}, specify train, val, and test
from datasets import load_dataset
from pathlib import Path

# Since we loaded source and target separately
# Here we need to make a class for creating combined torch dataset
from torch.utils.data import Dataset

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

import wandb

from typing import NamedTuple, List

from timeit import default_timer as timer
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
SRC_VOCAB_SIZE = 298
TGT_VOCAB_SIZE = 298
EMB_SIZE = 256
NHEAD = 1
FFN_HID_DIM = 2048
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
NUM_EPOCHS = 1
LEARNING_RATE = 0.0001


class Batch(NamedTuple):
    src_text: List[str]
    tgt_text: List[str]


class InputDataset(Dataset):
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



class EncodedBatch(NamedTuple):
    src_sequence_ids: torch.tensor
    src_attention_mask: torch.tensor
    tgt_sequence_ids: torch.tensor
    tgt_attention_mask: torch.tensor


def collate_fn(batch:Batch):
    batch = default_collate(batch)
    encoded_src = tokenizer.encode_batch(batch.src_text)
    encoded_tgt = tokenizer.encode_batch(batch.tgt_text)
    src_sequence_ids = torch.tensor([elem.ids for elem in encoded_src]).T
    tgt_sequence_ids = torch.tensor([elem.ids for elem in encoded_tgt]).T
    src_attention_mask = torch.tensor([elem.attention_mask for elem in encoded_src]).T
    tgt_attention_mask = torch.tensor([elem.attention_mask for elem in encoded_tgt]).T
    encoded_batch = EncodedBatch(src_sequence_ids, src_attention_mask, tgt_sequence_ids, tgt_attention_mask)
    return encoded_batch

# %%



# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# MechTransformer Network
class MechTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(MechTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor = None):

        # TODO: Re-add self.positional_encoding()
        src_emb = self.src_tok_emb(src)
        tgt_emb = self.tgt_tok_emb(trg)

        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)

        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        # TODO: Re-add self.positional_encoding()
        return self.transformer.encoder(self.src_tok_emb(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # TODO: Re-add self.positional_encoding()
        return self.transformer.decoder(self.tgt_tok_emb(tgt), memory, tgt_mask)

# %%
# Make the masks

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    PAD_IDX = 2

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    batch: EncodedBatch
    for batch in train_dataloader:
        src = batch.src_sequence_ids.to(DEVICE)
        tgt = batch.tgt_sequence_ids.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        tgt_mask = tgt_mask < 0
        
        logits = model(
            src = src, 
            trg = tgt_input, 
            src_mask = src_mask, 
            tgt_mask = tgt_mask,
            src_padding_mask = src_padding_mask, 
            tgt_padding_mask = tgt_padding_mask, 
            memory_key_padding_mask = src_padding_mask
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        wandb.log({"train_batch_losses": loss.item()})

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for batch in val_dataloader:
        src = batch.src_sequence_ids.to(DEVICE)
        tgt = batch.tgt_sequence_ids.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


def main():
    data_path = Path("data/MIT_separated_augm")

    dataset_src = load_dataset("text", data_files={"train": [str(data_path / "src-train.txt")],
                                                   "val": [str(data_path / "src-val.txt")]})
    dataset_tgt = load_dataset("text", data_files={"train": [str(data_path / "tgt-train.txt")],
                                                   "val": [str(data_path / "tgt-val.txt")]})

    # Add [START] and [END] tokens for all SMILES
    dataset_src = dataset_src.map(lambda x: {"text": "[START] " + x["text"] + " [END]"})
    dataset_tgt = dataset_tgt.map(lambda x: {"text": "[START] " + x["text"] + " [END]"})
    # %%
    # Using the InputDataset class to create train_iter and val_iter
    train_iter = InputDataset(dataset_src["train"], dataset_tgt["train"])
    val_iter = InputDataset(dataset_src["val"], dataset_tgt["val"])

    # Check the format is now (src_text,tgt_text)
    # train_iter[0]

    # %%
    # # Make the custom dictionary for our vocabs
    from itertools import chain
    # with open('../mirana_data/train_src.txt') as file:
    #     lines = file.readlines()

    #     tokens = sorted(set(chain.from_iterable(map(str.split, map(str.strip, lines)))))

    #     # Here I moved the SMILE vocabs back by 4 spaces in the dictionary
    #     # In order to add the following tokens into the dictionary
    #     dictionary = dict(zip(tokens, range(4, len(tokens) + 4)))
    #     dictionary["[START]"] = 0
    #     dictionary["[END]"] = 1
    #     dictionary["[PAD]"] = 2
    #     dictionary["[UNK]"] = 3

    # # %%
    # # Making the tokenizer

    # tokenizer = Tokenizer(WordLevel(vocab = dictionary, unk_token="[UNK]"))
    # tokenizer.save("tokenizer.json")

    tokenizer = Tokenizer.from_file('tokenizer_from_USPTO_MIT.json')

    # Modifying the tokenizer

    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.enable_padding(pad_id=2)

    # %%
    # Now we have the tokenizer, we need to define the collate function
    # The collate function will be called by the DataLoader

    from torch.utils.data.dataloader import default_collate
    # from torch.utils.data import default_collate
    torch.manual_seed(0)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="USPTO_MIT_TESTING",

        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "Mechformer",
            "dataset": str(data_path / "src-train.txt"),
            "tokenizer": "",
            "SRC_VOCAB_SIZE": SRC_VOCAB_SIZE,
            "TGT_VOCAB_SIZE": TGT_VOCAB_SIZE,
            "epochs": NUM_EPOCHS,
            "EMB_SIZE": EMB_SIZE,
            "NHEAD": NHEAD,
            "FFN_HID_DIM": FFN_HID_DIM,
            "BATCH_SIZE": BATCH_SIZE,
            "NUM_ENCODER_LAYERS": NUM_ENCODER_LAYERS,
            "NUM_DECODER_LAYERS": NUM_DECODER_LAYERS,
        }
    )

    transformer = MechTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE,
                                  TGT_VOCAB_SIZE, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # transformer.load_state_dict(torch.load('/baldig/chemistry/2023_rp/mirana_testing_mechformer/trained_models/myer_pretrain_sept_8.pth'))
    # transformer.eval()

    transformer = transformer.to(DEVICE)

    PAD_IDX = 2

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

    # %%
    # Define Train,Val,Decode,Test

    from torch.utils.data import DataLoader

    train_losses = []
    val_losses = []
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        wandb.log({"train_losses": train_loss, "val_losses": val_loss})
        # accuracy = compute_accuracy("mirana_data/test_src.txt","mirana_data/test_tgt.txt")
        print(
            f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")
        torch.save(transformer.state_dict(), 'trained_models/USPTO_MIT_OCT_4.pth')

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig("/baldig/chemistry/2023_rp/mirana_testing_mechformer/loss_pics/USPTO_MIT_OCT_4.png")
