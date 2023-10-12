from temporary_train import MechTransformer
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

transformer = MechTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

PAD_IDX = 2

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

torch.save(transformer.state_dict(), 'trained_models/USPTO_MIT_OCT_4.pth')
