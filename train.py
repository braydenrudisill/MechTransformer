import torch
from torch import nn

import math

from src.models.transformer import TransformerModel, create_mask
from src.data import EncodedBatch
from src.data import load

import yaml
from pathlib import Path

import os
import wandb
from tqdm import tqdm
from timeit import default_timer as timer


torch.manual_seed(0)

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

    # Update config
    config['data_path'] = Path('data/' + config['data_set'])
    if not torch.cuda.is_available():
        config['device'] = 'cpu'


train_data, val_data, tokenizer = load(config['data_path'], config['batch_size'])

config['src_vocab_size'] = config['tgt_vocab_size'] = tokenizer.get_vocab_size()
print(config['src_vocab_size'])


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project='USPTO_MIT_TESTING',

    # track hyperparameters and run metadata
    config={
        'architecture': 'Mechformer',
        'dataset': config['data_path'] / 'src-train.txt',
        'tokenizer': config['data_path'] / 'tokenizer.json',
        'SRC_VOCAB_SIZE': config['src_vocab_size'],
        'TGT_VOCAB_SIZE': config['tgt_vocab_size'],
        'epochs': config['num_epochs'],
        'EMB_SIZE': config['emb_size'],
        'NHEAD': config['nhead'],
        'FFN_HID_DIM': config['ffn_hid_dim'],
        'BATCH_SIZE': config['batch_size'],
        'NUM_ENCODER_LAYERS': config['num_encoder_layers'],
        'NUM_DECODER_LAYERS': config['num_decoder_layers'],
    }
)

transformer = TransformerModel(
    config['num_encoder_layers'],
    config['num_decoder_layers'],
    config['emb_size'],
    config['nhead'],
    config['src_vocab_size'],
    config['tgt_vocab_size'],
    config['ffn_hid_dim'],
)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(config['device'])

PAD_IDX = 2

optimizer = torch.optim.Adam(transformer.parameters(), betas=(0.9, 0.998), eps=1e-8)


def warmup(step: int):
    step += 1
    learning_rate = config['emb_size'] ** (-0.5) * min(step ** (-0.5), step * config['warmup_steps'] ** (-1.5))
    wandb.log({"learning_rate": learning_rate})
    return learning_rate


lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train_epoch(model, optimizer):
    model.train()
    losses = 0

    batch: EncodedBatch
    for i, batch in enumerate(train_data):
        src = batch.src_sequence_ids.to(config['device'])
        tgt = batch.tgt_sequence_ids.to(config['device'])

        tgt_input = tgt[:, :-1]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, config['device'])

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[:, 1:]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        losses += loss.item()
        wandb.log({"train_batch_losses": loss.item()})
        if i % 5 == 0:
            if not os.path.isdir(folder := f'src/models/transformer/pretrained/USPTO_MIT_OCT_18/epoch-{i}/'):
                os.mkdir(folder)

            torch.save(transformer.state_dict(), folder + f'batch_{i}.pth')

    return losses / len(list(train_data))


def evaluate(model):
    model.eval()
    losses = 0

    batch: EncodedBatch
    for batch in val_data:
        src = batch.src_sequence_ids.to(config['device'])
        tgt = batch.tgt_sequence_ids.to(config['device'])

        tgt_input = tgt[:, :-1]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device=config['device'])

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[:, 1:]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        losses += loss.item()

    return losses / len(list(val_data))


train_losses = []
val_losses = []
for epoch in range(1, config['num_epochs'] + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    wandb.log({'train_losses': train_loss, 'val_losses': val_loss})

    # accuracy = compute_accuracy('mirana_data/test_src.txt','mirana_data/test_tgt.txt')
    print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")

    if not os.path.isdir(folder := 'src/models/transformer/pretrained/USPTO_MIT_OCT_18/'):
        os.mkdir(folder)
    torch.save(transformer.state_dict(), folder + f'epoch-{epoch}.pth')

