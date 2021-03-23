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

#for inference mode
def load_file_sentence(text, segment_word):
    data = []
    text = text.split()
    len_list = len(text)
    for i in range(len_list//segment_word):
        data.append(" ".join(list[i*segment_word:(i+1)*segment_word]))
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
    sum_x_token = 0
    sum_data = 0
    for line in data:
        #print("line : ", line)
        if len(line.split()) > 5:
            x = tokenizer.encode_plus(line, pad_to_max_length=False, add_special_tokens=False, truncation=False, return_attention_mask=True)
            y = []
            x_token = tokenizer.convert_ids_to_tokens(x["input_ids"])
            x_token = [x for x in x_token if x != '▁']
            sum_x_token += len(x_token)
            sum_data += 1
            #for i in range(len(x_token)):
            #    if x_token[i] == ".":
            #        x_token.insert(i+1, "</s>")
            #if the first element of x_token is a punc, we delete it
            if x_token[0] in puncs:
                del x_token[0]
            #list x_token without the punctuation
            x_token_without_punc = []
            for i in range(len(x_token)):
                if x_token[i] in puncs:
                    y.append(punctuation_enc[x_token[i]])
                else:
                    y.append(punctuation_enc["TOKEN"])
                    x_token_without_punc.append(x_token[i])
            j = 1
            while(j < len(y)):
                #if y[j] != punctuation_enc["TOKEN"] and y[j] != punctuation_enc["."]:
                if y[j] != punctuation_enc["TOKEN"]:
                    del y[j-1]
                #elif y[j] == punctuation_enc["."]:
                #    del y[j-1]
                #    j += 1
                else:
                    j += 1
            if x_token_without_punc != []:
                #number of [MASK] we add, 15% of the sentence
                #nb_masks = int(0.15*len(x_token_without_punc)) + 1
                #indices we will mask
                #random_indices = random.sample(range(1, len(x_token_without_punc)), nb_masks)
                #for j in random_indices:
                #    x_token_without_punc[j] = '<mask>'
                x_token_without_punc = " ".join(x_token_without_punc)
                x = tokenizer.encode_plus(x_token_without_punc, pad_to_max_length=True, add_special_tokens=False, truncation=True, max_length=segment_size, return_attention_mask=True)
                x_decode = tokenizer.convert_ids_to_tokens(x["input_ids"])
                X.append(x)
                #print("x : ", x)
                #print("x_decode : ", x_decode)
                #print("y : ", y)
                Y.append(y)
    Y = pad_sequences([y for y in Y], maxlen=segment_size, dtype="long", value=0,
                        truncating="post", padding="post")
    return X, Y

def create_data_loader(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(np.array([x["input_ids"] for x in X])).long(), torch.from_numpy(np.array([x["attention_mask"] for x in X])).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def create_data_loader_without_attentions(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(np.array([x["input_ids"] for x in X])).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader

