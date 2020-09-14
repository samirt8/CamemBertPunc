import random
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
        x_token = line.split()
        y = []
        if len(x_token) > 1:
            x_token = " ".join(x_token)
            x_token = tokenizer.tokenize(x_token)
            #number of [MASK] we add, 15% of the sentence
            nb_masks = int(0.15*len(x_token)) + 1
            #indices we will mask
            random_indices = random.sample(range(1, len(x_token)), nb_masks)
            for j in random_indices:
                x_token[j] = '<mask>'
            x_token = " ".join(x_token)
            x = tokenizer.encode_plus(x_token, pad_to_max_length=True, add_special_tokens=True, truncation=True,
                                      padding_side="right", max_length=segment_size, return_attention_mask=True)
            x_token = tokenizer.convert_ids_to_tokens(x["input_ids"])
            for i, word in enumerate(x_token):
                if word in puncs:
                    y.append(punctuation_enc[word])
                else:
                    y.append(punctuation_enc['TOKEN'])
            X.append(x)
            Y.append(y)
    return X, Y

def create_data_loader(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(np.array(X)).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader