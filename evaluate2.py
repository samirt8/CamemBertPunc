from datetime import datetime
startTime = datetime.now()

import os
import re
import gc
import pandas as pd
import numpy as np
from glob import glob
from transformers import CamembertTokenizer
import torch
from torch import nn
#%matplotlib inline
import json
from tqdm import tqdm
from sklearn import metrics

from model import BertPunc, BertPunc_ner
from data import load_file, load_file2, encode_data3, create_data_loader

segment_word = 10

tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

punctuation_enc = {
    'PAD': 0,
    'TOKEN': 1,
    ',': 2,
    '.': 3
}

inv_punctuation_enc = {v: k for k, v in punctuation_enc.items()}

inv_punctuation_enc_modify = {
        0: '',
        1: '',
        2: ',',
        3: '.'
}

puncs = [
    'PAD', 'TOKEN', ',', '.']

segment_size = 25
output_size = len(punctuation_enc)

bert_punc = BertPunc_ner(segment_size, output_size)

#underfit => 20201001_143458 (fonctionne bien)
#overfit => 0200930_111537
MODEL_PATH = "models/20201001_143458/model_modif"
checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
#checkpoint = torch.load(MODEL_PATH, map_location="cpu", encoding='latin1')

from collections import OrderedDict
new_checkpoint = OrderedDict()
for k, v in checkpoint.items():
    new_k = k[7:]
    new_checkpoint[new_k] = v

#load params
bert_punc.load_state_dict(new_checkpoint)

batch_size = 256

def predictions(data_loader):
    all_result_sentences = []
    y_pred = []
    y_true = []
    for inputs, labels in tqdm(data_loader, total=len(data_loader)):
        with torch.no_grad():
            #inputs, labels = inputs.cuda(), labels.cuda()
            print("len : ", len(data_loader))
            print("inputs : ", inputs)
            print("shape : ", inputs.shape)
            output = bert_punc(inputs)
            print("output : ", output)
            print("output shape : ", output.shape)
            y_pred += list(output.argmax(dim=1).cpu().data.numpy().flatten())
            y_true += list(labels.cpu().data.numpy().flatten())
            result_sentences = []
            for i, sentence in enumerate(inputs):
                sentence_input = tokenizer.convert_ids_to_tokens(sentence)
                sentence_output = ['']+[inv_punctuation_enc_modify.get(item,item) for item in output.argmax(dim=1)[i].cpu().data.numpy()][:-1]
                #sentence_output = [inv_punctuation_enc_modify.get(item,item) for item in output.argmax(dim=1)[i].cpu().data.numpy()]
                result_sentence = [None]*(len(sentence_input)+len(sentence_output))
                result_sentence[::2] = sentence_input
                result_sentence[1::2] = sentence_output
                result_sentence = list(filter(lambda a: a != '', result_sentence))
                result_sentence = tokenizer.convert_tokens_to_ids(result_sentence)
                result_sentence = tokenizer.decode(result_sentence, skip_special_tokens=True)
                result_sentences.append(result_sentence)
            result_sentences = " ".join(result_sentences)
            #print(result_sentences)
            result_sentences = result_sentences.split(".")
            result_sentences = [result_sentence[0]+result_sentence[1].upper()+result_sentence[2:]+"." if len(result_sentence) > 1 else result_sentence for result_sentence in result_sentences]
            #print("".join(result_sentences))
            all_result_sentences.append("".join(result_sentences))
    #return y_pred, y_true
    return "\n".join(all_result_sentences)

def evaluation(y_pred, y_test):
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[2, 3])
    overall = metrics.precision_recall_fscore_support(
        y_test, y_pred, average='macro', labels=[2, 3])
    result = pd.DataFrame(
        np.array([precision, recall, f1]),
        columns=list(punctuation_enc.keys())[2:],
        index=['Precision', 'Recall', 'F1']
    )
    result['OVERALL'] = overall[:3]
    return result



for file in os.listdir("inputs/macron/"):
    print("dealing with file : ", file)

    data_test = load_file2(os.path.join("inputs/macron", file), segment_word)

    X_test, y_test = encode_data3(data_test, tokenizer, puncs, punctuation_enc, segment_size)

    data_loader_test = create_data_loader(X_test, y_test, False, batch_size)

    #y_pred_test, y_true_test = predictions(data_loader_test)
    all_result_sentences = predictions(data_loader_test)

    #eval_test = evaluation(y_pred_test, y_true_test)
    #print(eval_test)

    #with open("results_"+file, "r") as f:
    #    lines = f.read()

    #lines = lines[0].upper() + lines[1].lower() + lines[2:]
    all_result_sentences = all_result_sentences[0].upper() + all_result_sentences[1].lower() + all_result_sentences[2:]

    with open("outputs/macron/results_"+file, "w") as f:
        #f.write(lines)
        f.write(all_result_sentences)

    #print(datetime.now() - startTime)
