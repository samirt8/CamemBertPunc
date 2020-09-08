import numpy as np
import torch
import array
from torch.utils.data import TensorDataset, DataLoader

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

def encode_data(data, tokenizer, puncs, punctuation_enc, segment_size):
    """
    Converts words to (BERT) tokens and punctuation to given encoding.
    Note that words can be composed of multiple tokens.
    """
    X = []
    Y = []
    for line in data:
        x = []
        y = []
        for i, word in enumerate(line.split()):
            if word in puncs:
                #on masque le caractère ponctué
                x.append("[MASK]")
                y.append(punctuation_enc[word])
            else:
                x.insert(i+1, word)
                y.append(punctuation_enc['O'])
        if x:
            x = " ".join(x)
            x = tokenizer.tokenize(x)
            x = tokenizer.convert_tokens_to_ids(x)
            x = tokenizer.encode(x, pad_to_max_length=True, truncation=True, padding_side="right", max_length=segment_size)
            current_len_y = len(y)
            y = y + [0]*(len(x) - current_len_y)
            y = y[:len(x)]
            X.append(x)
            Y.append(y)
    return X, Y

def create_data_loader(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(np.array(X)).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader