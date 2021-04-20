import random
import numpy as np
import pandas as pd
import torch
import array
from torch.utils.data import TensorDataset, DataLoader
from transformers import CamembertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset


class PunctuationDataset(Dataset):
    """Dataset to infer punctuation"""

    def __init__(self, txt_file, segment_word, segment_size, puncs):
        """
        :param csv_file: csv file where data is stored
        :param segment_word: length in words in each sentence
        :param segment_size: length in tokens in each sentence
        """
        self.segment_word = segment_word
        self.segment_size = segment_size
        self.txt_file = txt_file
        data = []
        with open(self.txt_file, "r", encoding="utf-8") as f:
            # we store the all file
            huge_list = f.read().split()
            len_huge_list = len(huge_list)
            for i in range(len_huge_list//self.segment_word):
                data.append(" ".join(huge_list[i*self.segment_word:(i+1)*self.segment_word]))
        self.data = data
        # need to change this
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.puncs = puncs
        self.punctuation_enc = {k:i for i,k in enumerate(self.puncs)}

    def __len__(self):
        # number of lines in the file
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.tolist())


        if len(self.data[idx].split()) <= 5:
            return None
        else:
            x = self.tokenizer.encode_plus(self.data[idx], pad_to_max_length=False, add_special_tokens=False, truncation=False, return_attention_mask=True)
            y = []
            x_token = self.tokenizer.convert_ids_to_tokens(x["input_ids"])
            for i, element in enumerate(x_token):
                if element == "▁;":
                    x_token[i] = "."
                elif element == "▁:":
                    x_token[i] == ","
                elif element == "▁":
                    del x_token[i]
                elif element in list(map(lambda x: "▁"+x, ["'", "#", "$", "%", "&", "'", "(", ")", "*", "+", "-", "/", "<", "=", ">", "@", "[", "^", "_", "`", "|", "~" , "'"])):
                    del x_token[i]
                else:
                    continue
            #x_token = [x for x in x_token if x != "▁"]

            # we delete the first element if it's punctuation
            if x_token[0] in self.puncs:
                del x_token[0]
            # list x_token without the punctuation
            x_token_without_punc = []
            attention_mask = []
            for i in range(len(x_token)):
                if x_token[i] in self.puncs:
                    y.append(self.punctuation_enc[x_token[i]])
                else:
                    y.append(self.punctuation_enc["TOKEN"])
                    x_token_without_punc.append(x_token[i])
            j = 1
            while(j < len(y)):
                if y[j] != self.punctuation_enc["TOKEN"]:
                    del y[j-1]
                else:
                    j += 1
            if x_token_without_punc == []:
                return None

            #print("x : ", x_token_without_punc)
            #print("y : ", y)

            # {input_ids: [...], attention_mask: [...]}
            x = self.tokenizer.encode_plus(x_token_without_punc, pad_to_max_length=True,
                                        add_special_tokens=False, truncation=True,
                                        max_length=self.segment_size, return_attention_mask=True,
                                        padding="max_length")

            # [...]
            zeros = (self.segment_size - len(y))*[0]
            y = y + zeros

            return {"inputs": torch.from_numpy(np.array(x["input_ids"])), 
                    "attentions": torch.from_numpy(np.array(x["attention_mask"])), 
                    "labels": torch.from_numpy(np.array(y[:self.segment_size]))}
