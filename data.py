
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

def encode_data(data, tokenizer, punctuation_enc):
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
        tokens = tokenizer.tokenize(line)
        X.append(x)
        Y.append(y)
    return np.array(X), Y

def preprocess_data(data, tokenizer, punctuation_enc, segment_size):
    X, y = encode_data(data, tokenizer, punctuation_enc)
    return X, y

def create_data_loader(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader
