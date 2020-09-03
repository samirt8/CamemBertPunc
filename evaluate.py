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

from model import BertPunc
from data import load_file, preprocess_data, create_data_loader

data_test = load_file('/media/nas/samir-data/punctuation/all_datasets/data_dir_punctuator_v3/sub_subset_cleaned_leMonde_with_punct_v2_for_punctuator.test.txt_clean')

tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

punctuation_enc = {
    'O': 0,
    ',COMMA': 1,
    '.PERIOD': 2,
    '?QUESTIONMARK': 3,
    ':COLON': 4,
    '!EXCLAMATIONMARK': 5,
    ';SEMICOLON': 6
}

# punctuation_enc = {
#     'O': 0,
#     'PERIOD': 1
# }

#segment_size = hyperparameters['segment_size']
segment_size = 32

X_test, y_test = preprocess_data(data_test, tokenizer, punctuation_enc, segment_size)

print("data_test : ", data_test)
print("X_test : ", X_test)
print("y_test : ", y_test)

output_size = len(punctuation_enc)
dropout = 0.3
bert_punc = nn.DataParallel(BertPunc(segment_size, output_size, dropout).cuda())
#bert_punc = BertPunc(segment_size, output_size, dropout)

MODEL_PATH = "/media/nas/samir-data/CamembertPunc/models_test/20200902_173211/model"
#checkpoint = torch.load(MODEL_PATH, map_location="cpu")

#bert_punc.load_state_dict(checkpoint, strict=False)

checkpoint = torch.load(MODEL_PATH)

batch_size = 16
data_loader_test = create_data_loader(X_test, y_test, False, batch_size)
#data_loader_test_asr = create_data_loader(X_test_asr, y_test_asr, False, batch_size)

def predictions(data_loader):
    y_pred = []
    y_true = []
    for inputs, labels in tqdm(data_loader, total=len(data_loader)):
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            output = bert_punc(inputs)
            y_pred += list(output.argmax(dim=1).cpu().data.numpy().flatten())
            y_true += list(labels.cpu().data.numpy().flatten())
            #print("y_pred : ", y_pred)
            #print("y_true : ", y_true)
    return y_pred, y_true

def evaluation(y_pred, y_test):
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[1, 2, 3, 4, 5, 6])
    overall = metrics.precision_recall_fscore_support(
        y_test, y_pred, average='macro', labels=[1, 2, 3, 4, 5, 6])
    result = pd.DataFrame(
        np.array([precision, recall, f1]),
        columns=list(punctuation_enc.keys())[1:],
        index=['Precision', 'Recall', 'F1']
    )
    result['OVERALL'] = overall[:3]
    return result

y_pred_test, y_true_test = predictions(data_loader_test)

eval_test = evaluation(y_pred_test, y_true_test)
eval_test
