from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from itertools import chain
from pathlib import Path
from glob import glob


def create_tokenizer_for(data_path: Path):
    """Creates a tokenizer file cataloging all tokens in the data."""
    txt_files = glob(f'{data_path}/*.txt')
    lines = chain.from_iterable(open(file).readlines() for file in txt_files)

    tokens = sorted(set(chain.from_iterable(map(str.split, map(str.strip, lines)))))

    # Here I moved the SMILE vocabs back by 4 spaces in the dictionary
    # In order to add the following tokens into the dictionary
    dictionary = {token: i+4 for i, token in enumerate(tokens)}
    dictionary['[START]'] = 0
    dictionary['[END]'] = 1
    dictionary['[PAD]'] = 2
    dictionary['[UNK]'] = 3

    assert len(dictionary.values()) == len(set(dictionary.values())), "Token values are not unique."

    tokenizer = Tokenizer(WordLevel(vocab=dictionary, unk_token='[UNK]'))
    tokenizer.save(data_path / 'tokenizer.json')
