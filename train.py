import torch
from torch import nn

from src.models.transformer import TransformerModel, create_mask
from src.data import EncodedBatch
from src.data import load

import yaml
from pathlib import Path

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
        'learning_rate': config['learning_rate'],
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

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)


def train_epoch(model, optimizer):
    model.train()
    losses = 0

    batch: EncodedBatch
    for batch in tqdm(train_data):
        src = batch.src_sequence_ids.to(config['device'])
        tgt = batch.tgt_sequence_ids.to(config['device'])

        tgt_input = tgt

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, config['device'])

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        wandb.log({"train_batch_losses": loss.item()})

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
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
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
    torch.save(transformer.state_dict(), 'trained_models/USPTO_MIT_OCT_4.pth')

