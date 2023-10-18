import torch
import math

maxlen = 5
emb_size = 6

den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
pos = torch.arange(0, maxlen).reshape(maxlen, 1)

pos_embedding = torch.zeros((maxlen, emb_size))

pos_embedding[:, 0::2] = torch.sin(pos * den)
# pos_embedding[:, 1::2] = torch.cos(pos * den)

print(pos_embedding.shape)


