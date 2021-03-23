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

from model import BertPunc, BertPunc_ner
from data import load_file, load_file2, encode_data3, create_data_loader, create_data_loader_without_attentions

segment_word = 12

data_test = load_file("dev.ester.clean")
#data_test = load_file2("test_ilyes_segment_long1.txt", segment_word)
#data_test = load_file("/media/nas/samir-data/punctuation/all_datasets/data_dir_punctuator_v3/subset_cleaned_leMonde_with_punct_v2_for_punctuator.test.txt")

tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

punctuation_enc = {
    'PAD': 0,
    'TOKEN': 1,
    ',': 2,
    '.': 3,
    '▁?': 4
}

#punctuation_enc = {
#	'PAD': 0,
#        'TOKEN': 1,
#        ',': 2,
#        '.': 3,
#        '▁?': 4,
#        '▁:': 5,
#        '▁!': 6,
#        '▁;': 7
#}

inv_punctuation_enc = {v: k for k, v in punctuation_enc.items()}

#inv_punctuation_enc_modify = {
#	0: '',
#	1: '',
#	2: ',',
#	3: '.',
#	4: '▁?',
#	5: '▁:',
#	6: '▁!',
#	7: '▁;'
#}

inv_punctuation_enc_modify = {
        0: '',
        1: '',
        2: ',',
        3: '.',
        4: '▁?'
}

puncs = [
    'PAD', 'TOKEN', ',', '.', '▁?']

#puncs = [
#    'PAD', 'TOKEN', ',', '.', '▁?', '▁:', '▁!', '▁;']

segment_size = 25

X_test, y_test = encode_data3(data_test, tokenizer, puncs, punctuation_enc, segment_size)

output_size = len(punctuation_enc)
#dropout = 0.3
#bert_punc = nn.DataParallel(BertPunc_ner(segment_size, output_size).cpu())
bert_punc = BertPunc_ner(segment_size, output_size)

#underfit => 20201001_143458 (fonctionne bien)
#overfit => 0200930_111537
MODEL_PATH = "/media/nas/samir-data/CamemBertPunc/models/20201001_143458/model"
checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

from collections import OrderedDict
new_checkpoint = OrderedDict()
for k, v in checkpoint.items():
    new_k = k[7:]
    new_checkpoint[new_k] = v

#load params
bert_punc.load_state_dict(new_checkpoint)

batch_size = 64
data_loader_test = create_data_loader(X_test, y_test, False, batch_size)
#data_loader_test_asr = create_data_loader(X_test_asr, y_test_asr, False, batch_size)

def predictions(data_loader):
    y_pred = []
    y_true = []
    for inputs, attentions, labels in tqdm(data_loader, total=len(data_loader)):
        with torch.no_grad():
            #inputs, labels = inputs.cuda(), labels.cuda()
            output = bert_punc(inputs, attentions)
            #print("output0 : ", output[0])
            #print("output1 : ", output[1])
            #print("output2 : ", output[2])
            #print("shape output : ", output.shape)
            y_pred += list(output.argmax(dim=1).cpu().data.numpy().flatten())
            y_true += list(labels.cpu().data.numpy().flatten())
            result_sentences = []
            for i, sentence in enumerate(inputs):
                sentence_input = tokenizer.convert_ids_to_tokens(sentence)
                #sentence_output = ['']+[inv_punctuation_enc_modify.get(item,item) for item in output.argmax(dim=1)[i].cpu().data.numpy()][:-1]
                sentence_output = [inv_punctuation_enc_modify.get(item,item) for item in output.argmax(dim=1)[i].cpu().data.numpy()]
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
            result_sentences = [result_sentence[0]+result_sentence[1].upper()+result_sentence[2:]+"." for result_sentence in result_sentences]
            print(result_sentences)
    return y_pred, y_true
    #return ""
    #return "".join(result_sentences)

def evaluation(y_pred, y_test):
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[2, 3, 4])
    overall = metrics.precision_recall_fscore_support(
        y_test, y_pred, average='macro', labels=[2, 3, 4])
    result = pd.DataFrame(
        np.array([precision, recall, f1]),
        columns=list(punctuation_enc.keys())[2:],
        index=['Precision', 'Recall', 'F1']
    )
    result['OVERALL'] = overall[:3]
    return result

print(predictions(data_loader_test))
y_pred_test, y_true_test = predictions(data_loader_test)

eval_test = evaluation(y_pred_test, y_true_test)
print(eval_test)

print(datetime.now() - startTime)
