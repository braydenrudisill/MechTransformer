import torch

src, tgt = torch.ones(5, 5), torch.ones(5, 5)

pad_idx = 3

src[0, 1] = 3
src[2, 3] = 3

tgt[4, 1] = 3
tgt[0, 3] = 3

# print(src, tgt)


src_padding_mask, tgt_padding_mask = (
        mask
        .float()
        .masked_fill(mask == pad_idx, float('-inf'))
        .masked_fill(mask != pad_idx, 0.0)
        for mask in (src, tgt)
    )

spm2 = (src == pad_idx)
tpm2 = (tgt == pad_idx)

print(src_padding_mask + src)
print(tgt_padding_mask + tgt)
#
# print(spm2)
# print(tpm2)


