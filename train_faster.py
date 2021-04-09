import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter
import os
import json
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import CamembertTokenizer

from sklearn import metrics  # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from model_faster import BertPunc_ner
from data_faster import PunctuationDataset


def validate(model, epoch, epochs, iteration, iterations, valid_loader, save_path, train_loss,
             best_val_loss, best_model_path, punctuation_enc):
    val_losses = []
    val_accs = []
    val_f1s = []

    label_keys = list(punctuation_enc.keys())
    label_vals = list(punctuation_enc.values())

    for x, y in tqdm(valid_loader, total=len(valid_loader)):
        with torch.no_grad():
            inputs, attentions, labels = x["input_ids"].cuda(), x["attention_mask"].cuda(), y.cuda()
            output = model(inputs, attentions, labels=labels)
            val_loss = output.loss
            val_losses.append(val_loss.cpu().data.numpy())

            y_pred = output.logits.view(-1, output_size, segment_size).argmax(dim=1).cpu().data.numpy().flatten()
            y_true = labels.cpu().data.numpy().flatten()
            val_accs.append(metrics.accuracy_score(y_true, y_pred))
            val_f1s.append(metrics.f1_score(y_true, y_pred, average=None, labels=label_vals))

    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    val_f1 = np.array(val_f1s).mean(axis=0)

    improved = ''

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

def train(model, optimizer, epochs, training_loader, valid_loader, save_path, punctuation_enc, iterations=3, best_val_loss=1e9):

    print_every = len(training_loader)//((iterations+1))
    best_model_path = None
    model.train()
    pbar = tqdm(total=print_every)

    for e in range(epochs):

        counter = 1
        iteration = 1

        for x, y in tqdm(training_loader, total=len(training_loader)):

            inputs, attentions, labels = x["input_ids"].cuda(), x["attention_mask"].cuda(), y.cuda()
            inputs.requires_grad = False
            attentions.requires_grad = False
            labels.requires_grad = False
            output = model(inputs, attentions, labels=labels)
            loss = output.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss = loss.cpu().data.numpy()

            pbar.update()

            if counter % print_every == 0:

                pbar.close()
                model.eval()
                best_val_loss, best_model_path = validate(model, e, epochs, iteration, iterations, valid_loader,
                    save_path, train_loss, best_val_loss, best_model_path, punctuation_enc)
                model.train()
                pbar = tqdm(total=print_every)
                iteration += 1
            counter += 1

        pbar.close()
        model.eval()
        best_val_loss, best_model_path = validate(model, e, epochs, iteration, iterations, valid_loader,
            save_path, train_loss, best_val_loss, best_model_path, punctuation_enc)
        model.train()
        if e < epochs-1:
            pbar = tqdm(total=print_every)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model, optimizer, best_val_loss

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    puncs = [
        'PAD', 'TOKEN', ',', '.', '▁?', '▁:', '▁!', '▁;']

    punctuation_enc = {k: i for i, k in enumerate(puncs)}

    segment_word = 20
    segment_size = 60
    batch_size = 64
    epochs_top = 1
    iterations_top = 2
    learning_rate_top = 3e-5
    epochs_all = 4
    iterations_all = 3
    learning_rate_all = 3e-5
    hyperparameters = {
        'segment_word': segment_word,
        'segment_size': segment_size,
        'batch_size': batch_size,
        'epochs_top': epochs_top,
        'iterations_top': iterations_top,
        'learning_rate_top': learning_rate_top,
        'epochs_all': epochs_all,
        'iterations_all': iterations_all,
        'learning_rate_all': learning_rate_all,
    }
    train_data_path = "/media/nas/samir-data/punctuation/all_datasets/data_dir_punctuator_v2_wait"
    valid_data_path = "/media/nas/samir-data/punctuation/all_datasets/data_dir_punctuator_v2_wait"
    save_path = 'models_faster/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(save_path)
    train_file = os.path.join(train_data_path, ...)
    valid_file = os.path.join(train_data_path, ...)
    with open(save_path+'hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)

    print('LOADING DATA...')
    training_set = PunctuationDataset(train_file, segment_word, segment_size, puncs)
    valid_set = PunctuationDataset(valid_file, segment_word, segment_size, puncs)

    training_loader = DataLoader(training_set, batch_size, shuffle=True, num_workers=5, collate_fn=None)
    valid_loader = DataLoader(valid_set, batch_size, shuffle=True, num_workers=0, collate_fn=None)

    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

    print('INITIALIZING MODEL...')
    output_size = len(puncs)
    bert_punc = BertPunc_ner(output_size).cuda()

    print('TRAINING TOP LAYER...')
    for name, param in bert_punc.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False
        else:
            param.requires_grad = True
    optimizer = optim.AdamW(bert_punc.parameters(), lr=learning_rate_top)
    bert_punc, optimizer, best_val_loss = train(bert_punc, optimizer, epochs_top,
        training_loader, valid_loader, save_path, punctuation_enc, iterations_top, best_val_loss=1e9)

    print('TRAINING ALL LAYER...')
    for p in bert_punc.module.bert.parameters():
        p.requires_grad = True
    optimizer = optim.AdamW(bert_punc.parameters(), lr=learning_rate_all)
    bert_punc, optimizer, best_val_loss = train(bert_punc, optimizer, epochs_all,
        training_loader, valid_loader, save_path, punctuation_enc, iterations_all, best_val_loss=best_val_loss)

