import random
import numpy as np
import pandas as pd
import torch
import array
from torch.utils.data import TensorDataset, DataLoader
from transformers import CamembertTokenizer
from keras.preprocessing.sequence import pad_sequences


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

    def __get_item__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if len(self.data[idx].split()) <= 5:
            return None
        else:
            x = self.tokenizer.encode_plus(self.data[idx], pad_to_max_length=False, add_special_tokens=False, truncation=False, return_attention_mask=True)
            y = []
            x_token = self.tokenizer.convert_ids_to_tokens(x["inputs_ids"])
            x_token = [x for x in x_token if x != "▁"]

            # we delete the first element if it's punctuation
            if x_token[0] in self.puncs:
                del x_token[0]
            # list x_token without the punctuation
            x_token_without_punc = []
            for i in range(len(x_token)):
                if x_token[i] in self.puncs:
                    y.append(self.punctuation_enc[x_token[i]])
                    del y[-2]
                    # if there is a comma, an exclamation point or interrogation point, we add an end of sentence
                    # we don't use it
                    #if x_token[i] in [".", "?", "!"]:
                    #    x_token_without_punc.append("</s>")
                    #    y.append(self.punctuation_enc["TOKEN"])
                else:
                    y.append(self.punctuation_enc["TOKEN"])
                    x_token_without_punc.append(x_token[i])
            if x_token_without_punc != []:
                x_token_without_punc = " ".join(x_token_without_punc)

                # {input_ids: [...], attention_mask: [...]}
                x = self.tokenizer.encode_plus(x_token_without_punc, pad_to_max_length=True,
                                          add_special_tokens=False, truncation=True,
                                          max_length=self.segment_size, return_attention_mask=True,
                                          padding="max_length")

            # [...]
            y = pad_sequences([y], maxlen=self.segment_size, dtype="long", value=0, truncating="post",
                                padding="post")[0]

            return {"input_values": x, "labels": y}