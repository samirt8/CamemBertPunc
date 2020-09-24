import random
import numpy as np
import torch
import array
from torch.utils.data import TensorDataset, DataLoader
from keras.preprocessing.sequence import pad_sequences

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

def load_file2(filename, segment_word):
    """
    In this version of load_file, we split the sentence every x tokens and deal with it
    segment word is the number of words in each segment
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        huge_list = f.read().split()
        len_huge_list = len(huge_list)
        for i in range(len_huge_list//segment_word):
            data.append(" ".join(huge_list[i*segment_word:(i+1)*segment_word]))
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
                                      max_length=segment_size, return_attention_mask=True)
            x_token = tokenizer.convert_ids_to_tokens(x["input_ids"])
            for i, word in enumerate(x_token):
                if word in puncs:
                    y.append(punctuation_enc[word])
                else:
                    y.append(punctuation_enc['TOKEN'])
            X.append(x["input_ids"])
            Y.append(y)
    return X, Y

def encode_data2(data, tokenizer, puncs, punctuation_enc, segment_size):
    """
    Converts words to (BERT) tokens and punctuation to given encoding.
    Note that words can be composed of multiple tokens. We add a <mask> token
    between every token and try to predict
    """
    X = []
    Y = []
    for line in data:
        x_token = line.split()
        y = []
        if len(x_token) > 1:
            x_token = " ".join(x_token)
            x_token = tokenizer.tokenize(x_token)
            x_token = " ".join(x_token)
            x = tokenizer.encode_plus(x_token, pad_to_max_length=True, add_special_tokens=True, truncation=True,
                                      max_length=segment_size, return_attention_mask=True)
            x_token = tokenizer.convert_ids_to_tokens(x["input_ids"])
            for j in range(len(x_token)-1):
                x_token.insert(2*j + 1, '<mask>')
            for i, word in enumerate(x_token):
                if word in puncs:
                    y.append(punctuation_enc[word])
                else:
                    y.append(punctuation_enc['TOKEN'])
            x = tokenizer.encode_plus(x_token, pad_to_max_length=True, add_special_tokens=True, truncation=True,
                                      max_length=segment_size, return_attention_mask=True)
            X.append(x["input_ids"])
            Y.append(y)
    return X, Y

def encode_data3(data, tokenizer, puncs, punctuation_enc, segment_size):
    """
    Converts words to (BERT) tokens and punctuation to given encoding.
    Note that words can be composed of multiple tokens. This is the data
    preprocessing for NER, we delete all punctuation in x
    """
    X = []
    Y = []
    for line in data:
        if len(line.split()) > 5:
            x = tokenizer.encode_plus(line, pad_to_max_length=False, add_special_tokens=False, truncation=False,
                                      return_attention_mask=True)
            y = []
            x_token = tokenizer.convert_ids_to_tokens(x["input_ids"])
            x_token_without_punc = []
            i = 0
            while(i < len(x_token)):
                if x_token[i] in puncs:
                    y.append(punctuation_enc[x_token[i]])
                    del x_token[i]
                else:
                    x_token_without_punc.append(x_token[i])
                    y.append(punctuation_enc['TOKEN'])
                    i+=1
            x = tokenizer.encode_plus(x_token_without_punc, pad_to_max_length=True, add_special_tokens=True, truncation=True, max_length=segment_size, return_attention_mask=True)
            X.append(x["input_ids"])
            Y.append(y)
    Y = pad_sequences([y for y in Y], maxlen=segment_size, dtype="long", value=0,
                        truncating="post", padding="post")
    return X, Y

def create_data_loader(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(np.array(X)).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader
