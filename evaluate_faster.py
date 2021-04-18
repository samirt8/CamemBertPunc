from datetime import datetime
startTime = datetime.now()

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

from torch.utils.data import DataLoader

from model_faster import BertPunc_ner
from data_faster import PunctuationDataset


test_file = "dev.ester.clean"

segment_word = 12
segment_size = 30

puncs = [
        'PAD', 'TOKEN', ',', '.', '▁?','▁!']
#puncs = ['PAD', 'TOKEN', ',', '.']


test_set = PunctuationDataset(test_file, segment_word, segment_size, puncs)
test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

punctuation_enc = {
    'PAD': 0,
    'TOKEN': 1,
    ',': 2,
    '.': 3,
    '▁?': 4,
    '▁!': 5
}
#punctuation_enc = {'PAD':0, 'TOKEN':1, ',':2, '.':3}

inv_punctuation_enc = {v: k for k, v in punctuation_enc.items()}

inv_punctuation_enc_modify = {
        0: '',
        1: '',
        2: ',',
        3: '.',
        4: '▁?',
        5: '▁!'
}
#inv_punctuation_enc_modify = {0:'', 1:'', 2:',', 3:'.'}

output_size = len(punctuation_enc)
bert_punc = BertPunc_ner(output_size)

#underfit => 20201001_143458 (fonctionne bien)
#overfit => 0200930_111537
MODEL_PATH = "/media/nas/samir-data/CamemBertPunc/models_faster/20210415_153743/model"
checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

from collections import OrderedDict
new_checkpoint = OrderedDict()
for k, v in checkpoint.items():
    new_k = k[7:]
    new_checkpoint[new_k] = v

#load params
bert_punc.load_state_dict(checkpoint)

def predictions(data_loader):
    y_pred = []
    y_true = []
    for data in tqdm(data_loader, total=len(data_loader)):
        with torch.no_grad():
            inputs, attentions, labels = data["inputs"], data["attentions"], data["labels"]
            x = {"input_ids":inputs, "attention_mask":attentions}
            output = bert_punc(x, labels=labels)[1]
            #print("output : ", output)
            y_pred += list(output.argmax(dim=2).cpu().data.numpy().flatten())
            y_true += list(labels.cpu().data.numpy().flatten())
            result_sentences = []
            for i, sentence in enumerate(inputs):
                sentence_input = tokenizer.convert_ids_to_tokens(sentence)
                sentence_output = [inv_punctuation_enc_modify.get(item,item) for item in output.argmax(dim=2)[i].cpu().data.numpy()]
                result_sentence = [None]*(len(sentence_input)+len(sentence_output))
                result_sentence[::2] = sentence_input
                result_sentence[1::2] = sentence_output
                result_sentence = list(filter(lambda a: a != '', result_sentence))
                result_sentence = tokenizer.convert_tokens_to_ids(result_sentence)
                result_sentence = tokenizer.decode(result_sentence, skip_special_tokens=True)
                result_sentences.append(result_sentence)
            result_sentences = " ".join(result_sentences)
            #result_sentences = result_sentences.split(".")
            #result_sentences = [result_sentence[0]+result_sentence[1].upper()+result_sentence[2:]+"." for result_sentence in result_sentences]
            print(result_sentences)
    return y_pred, y_true


def evaluation(y_pred, y_test):
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[2, 3, 4, 5])
    overall = metrics.precision_recall_fscore_support(
        y_test, y_pred, average='macro', labels=[2, 3, 4, 5])
    result = pd.DataFrame(
        np.array([precision, recall, f1]),
        columns=list(punctuation_enc.keys())[2:],
        index=['Precision', 'Recall', 'F1']
    )
    result['OVERALL'] = overall[:3]
    return result

y_pred_test, y_true_test = predictions(test_loader)

eval_test = evaluation(y_pred_test, y_true_test)

print(datetime.now() - startTime)
