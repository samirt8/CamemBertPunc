import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter
import os
import json
from datetime import datetime

import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import CamembertTokenizer
# from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
#from torchsample.callbacks import EarlyStopping

from sklearn import metrics  # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from model import BertPunc, BertPunc_ner
from data import load_file, load_file2, encode_data3, create_data_loader, create_data_loader_without_attentions


def validate(model, criterion, epoch, epochs, iteration, iterations, data_loader_valid, save_path, train_loss,
             best_val_loss, best_model_path, punctuation_enc):
    val_losses = []
    val_accs = []
    val_f1s = []

    label_keys = list(punctuation_enc.keys())
    label_vals = list(punctuation_enc.values())

    for inputs, labels in tqdm(data_loader_valid, total=len(data_loader_valid)):
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            #output = model(inputs, attentions)
            output = model(inputs)
            val_loss = criterion(output, labels)
            val_losses.append(val_loss.cpu().data.numpy())

            y_pred = output.argmax(dim=1).cpu().data.numpy().flatten()
            y_true = labels.cpu().data.numpy().flatten()
            val_accs.append(metrics.accuracy_score(y_true, y_pred))
            val_f1s.append(metrics.f1_score(y_true, y_pred, average=None, labels=label_vals))

    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    val_f1 = np.array(val_f1s).mean(axis=0)

    improved = ''

    # model_path = '{}model_{:02d}{:02d}'.format(save_path, epoch, iteration)
    model_path = save_path+'model'
    torch.save(model.state_dict(), model_path)
    if val_loss < best_val_loss:
        improved = '*'
        best_val_loss = val_loss
        best_model_path = model_path

    f1_cols = ';'.join(['f1_'+key for key in label_keys])

    progress_path = save_path+'progress.csv'
    if not os.path.isfile(progress_path):
        with open(progress_path, 'w') as f:
            f.write('time;epoch;iteration;training loss;loss;accuracy;'+f1_cols+'\n')

    f1_vals = ';'.join(['{:.4f}'.format(val) for val in val_f1])

    with open(progress_path, 'a') as f:
        f.write('{};{};{};{:.4f};{:.4f};{:.4f};{}\n'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch+1,
            iteration,
            train_loss,
            val_loss,
            val_acc,
            f1_vals
            ))

    print("Epoch: {}/{}".format(epoch+1, epochs),
          "Iteration: {}/{}".format(iteration, iterations),
          "Loss: {:.4f}".format(train_loss),
          "Val Loss: {:.4f}".format(val_loss),
          "Acc: {:.4f}".format(val_acc),
          "F1: {}".format(f1_vals),
          improved)

    return best_val_loss, best_model_path

def train(model, optimizer, criterion, epochs, data_loader_train, data_loader_valid, save_path, punctuation_enc, iterations=3, best_val_loss=1e9):

    print_every = len(data_loader_train)//((iterations+1))
    clip = 5
    best_model_path = None
    model.train()
    pbar = tqdm(total=print_every)

    for e in range(epochs):

        counter = 1
        iteration = 1

        for inputs, labels in data_loader_train:

            inputs, labels = inputs.cuda(), labels.cuda()
            inputs.requires_grad = False
            #attentions.requires_grad = False
            labels.requires_grad = False
            #output = model(inputs, attentions)
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss = loss.cpu().data.numpy()

            pbar.update()

            if counter % print_every == 0:

                pbar.close()
                model.eval()
                best_val_loss, best_model_path = validate(model, criterion, e, epochs, iteration, iterations, data_loader_valid,
                    save_path, train_loss, best_val_loss, best_model_path, punctuation_enc)
                model.train()
                pbar = tqdm(total=print_every)
                iteration += 1
            counter += 1

        pbar.close()
        model.eval()
        best_val_loss, best_model_path = validate(model, criterion, e, epochs, iteration, iterations, data_loader_valid,
            save_path, train_loss, best_val_loss, best_model_path, punctuation_enc)
        model.train()
        if e < epochs-1:
            pbar = tqdm(total=print_every)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model, optimizer, best_val_loss

if __name__ == '__main__':

    #punctuation_enc = {
    #    'PAD': 0,
    #    'TOKEN': 1,
    #    ',': 2,
    #    '.': 3,
    #    '▁?': 4,
    #    '▁:': 5,
    #    '▁!': 6,
    #    '▁;': 7
    #}

    punctuation_enc = {
            'PAD': 0,
            'TOKEN': 1,
            ',': 2,
            '.': 3
            }

    #punctuation_enc_validation = {
    #    ',': 2,
    #    '.': 3,
    #    '▁?': 4,
    #    '▁:': 5,
    #    '▁!': 6,
    #    '▁;': 7
    #}

    punctuation_enc_validation = {
        ',': 2,
        '.': 3
        }

    #puncs = [
    #    'PAD', 'TOKEN', ',', '.', '▁?', '▁:', '▁!', '▁;']

    puncs = [
        'PAD', 'TOKEN', ',', '.']

    segment_word = 12
    segment_size = 36
    epochs_top = 1
    iterations_top = 2
    batch_size_top = 128
    learning_rate_top = 3e-4
    epochs_all = 4
    iterations_all = 3
    batch_size_all = 128
    learning_rate_all = 3e-4
    hyperparameters = {
        'segment_word': segment_word,
        'segment_size': segment_size,
        'epochs_top': epochs_top,
        'iterations_top': iterations_top,
        'batch_size_top': batch_size_top,
        'learning_rate_top': learning_rate_top,
        'epochs_all': epochs_all,
        'iterations_all': iterations_all,
        'batch_size_all': batch_size_all,
        'learning_rate_all': learning_rate_all,
    }
    train_data_path = "/media/nas/samir-data/punctuation/all_datasets/data_dir_punctuator_v2_wait"
    train_data_path2 = "/media/nas/samir-data/punctuation/all_datasets/data_europarl/training-monolingual-europarl"
    data_path = "/media/nas/samir-data/punctuation/all_datasets/data_dir_punctuator_v3"
    save_path = 'models/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(save_path)
    with open(save_path+'hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)

    print('LOADING DATA...')
    #data_train = load_file(os.path.join(train_data_path2, 'europarl-v7.fr_cleaned.txt'))
    #data_valid = load_file2(os.path.join(train_data_path, 'cleaned_leMonde_with_punct_v2_for_punctuator.dev.txt'), segment_word)
    data_train = load_file2(os.path.join(data_path,'subset_cleaned_leMonde_with_punct_v2_for_punctuator.train.txt'), segment_word)
    data_valid = load_file2(os.path.join(data_path,'subset_cleaned_leMonde_with_punct_v2_for_punctuator.test.txt'), segment_word)

    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

    print('PREPROCESSING DATA...')
    X_train, y_train = encode_data3(data_train, tokenizer, puncs, punctuation_enc, segment_size)
    X_valid, y_valid = encode_data3(data_valid, tokenizer, puncs, punctuation_enc, segment_size)

    print('INITIALIZING MODEL...')
    output_size = len(punctuation_enc)
    bert_punc = nn.DataParallel(BertPunc_ner(segment_size, output_size).cuda())

    print('TRAINING TOP LAYER...')
    #data_loader_train = create_data_loader(X_train, y_train, True, batch_size_top)
    #data_loader_valid = create_data_loader(X_valid, y_valid, False, batch_size_top)
    data_loader_train = create_data_loader_without_attentions(X_train, y_train, True, batch_size_top)
    data_loader_valid = create_data_loader_without_attentions(X_valid, y_valid, False, batch_size_top)
    for name, param in bert_punc.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False
        else:
            param.requires_grad = True
    optimizer = optim.AdamW(bert_punc.parameters(), lr=learning_rate_top)
    criterion = nn.CrossEntropyLoss()
    bert_punc, optimizer, best_val_loss = train(bert_punc, optimizer, criterion, epochs_top,
        data_loader_train, data_loader_valid, save_path, punctuation_enc, iterations_top, best_val_loss=1e9)


    print('TRAINING ALL LAYER...')
    #data_loader_train = create_data_loader(X_train, y_train, True, batch_size_all)
    #data_loader_valid = create_data_loader(X_valid, y_valid, False, batch_size_all)
    data_loader_train = create_data_loader_without_attentions(X_train, y_train, True, batch_size_all)
    data_loader_valid = create_data_loader_without_attentions(X_valid, y_valid, False, batch_size_all)
    for p in bert_punc.module.bert.parameters():
        p.requires_grad = True
    optimizer = optim.AdamW(bert_punc.parameters(), lr=learning_rate_all)
    criterion = nn.CrossEntropyLoss()
    bert_punc, optimizer, best_val_loss = train(bert_punc, optimizer, criterion, epochs_all,
        data_loader_train, data_loader_valid, save_path, punctuation_enc, iterations_all, best_val_loss=best_val_loss)

