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
from transformers import get_linear_schedule_with_warmup

from sklearn import metrics  # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from model_faster1 import BertPunc_ner
from data_faster import PunctuationDataset


def validate(model, epoch, epochs, iteration, iterations, valid_loader, save_path, train_loss,
             best_val_loss, best_model_path, punctuation_enc):
    val_losses = []
    val_accs = []
    val_f1s = []

    label_keys = list(punctuation_enc.keys())
    label_vals = list(punctuation_enc.values())

    for data in tqdm(valid_loader, total=len(valid_loader)):
        with torch.no_grad():
            inputs, attentions, labels = data["inputs"].cuda(0), data["attentions"].cuda(0), data["labels"].cuda(0)
            x = {"input_ids":inputs, "attention_mask":attentions}
            #output = model(x, labels)
            output = model(x)
            val_loss = criterion(output.reshape(-1, output_size), labels.reshape(-1))
            val_loss = val_loss.mean()
            val_losses.append(val_loss.cpu().data.numpy())

            y_pred = output.view(-1, output_size, segment_size).argmax(dim=1).cpu().data.numpy().flatten()
            y_true = labels.cpu().data.numpy().flatten()
            val_accs.append(metrics.accuracy_score(y_true, y_pred))
            val_f1s.append(metrics.f1_score(y_true, y_pred, average=None, labels=label_vals))

            # get output :
            result_sentences = []
            inv_punctuation_enc = {0: "", 1: "", 2: ",", 3: ".", 4: "▁?", 5: "▁!"}
            for i, sentence in enumerate(inputs):
                sentence_input = tokenizer.convert_ids_to_tokens(sentence)
                sentence_output = [inv_punctuation_enc.get(item, item) for item in output.argmax(dim=2)[i].cpu().data.numpy()]
                result_sentence = [None]*(len(sentence_input)+len(sentence_output))
                result_sentence[::2] = sentence_input
                result_sentence[1::2] = sentence_output
                result_sentence = list(filter(lambda a: a != '', result_sentence))
                result_sentence = tokenizer.convert_tokens_to_ids(result_sentence)
                result_sentence = tokenizer.decode(result_sentence, skip_special_tokens=True)
                result_sentences.append(result_sentence)
            result_sentences = " ".join(result_sentences)
            print("epoch : ", str(epoch)+"\n")
            print(result_sentences)

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

def train(model, criterion, optimizer, scheduler, epochs, training_loader, valid_loader, save_path, punctuation_enc, iterations=3, best_val_loss=1e9):

    print_every = len(training_loader)//((iterations))
    best_model_path = None
    model.train()
    pbar = tqdm(total=print_every)

    for e in range(epochs):

        counter = 1
        iteration = 1

        for data in training_loader:
            optimizer.zero_grad()
            inputs, attentions, labels = data["inputs"].cuda(0), data["attentions"].cuda(0), data["labels"].cuda(0)
            x = {"input_ids":inputs, "attention_mask":attentions}
            inputs.requires_grad = False
            attentions.requires_grad = False
            labels.requires_grad = False
            output = model(x)
            #output = output.view(-1, output_size, segment_size)
            #output[attentions == 0] = -10000
            loss = criterion(output.reshape(-1, output_size), labels.reshape(-1))
            # need to mask the loss function for every subword which are not end of word
            loss = loss.mean()
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
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
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    puncs = [
        'PAD', 'TOKEN', ',', '.', '▁?', '▁!']
    #puncs = [
    #    'PAD', 'TOKEN', ',', '.', '?','!']

    punctuation_enc = {k: i for i, k in enumerate(puncs)}

    segment_word = 15
    segment_size = 40
    batch_size = 128
    epochs_top = 1
    iterations_top = 1
    learning_rate_top = 1e-4
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
    train_data_path = "/media/nas/samir-data/punctuation/all_datasets/data_europarl/training-monolingual-europarl"
    valid_data_path = "/media/nas/samir-data/punctuation/all_datasets/data_dir_punctuator_v3"
    save_path = 'models_faster/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(save_path)
    train_file = os.path.join(train_data_path, "europarl-v7.fr_cleaned_1000000.txt")
    valid_file = os.path.join(valid_data_path, "sub_subset_cleaned_leMonde_with_punct_v2_for_punctuator.dev.txt")
    with open(save_path+'hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)


    print('LOADING DATA...')
    training_set = PunctuationDataset(train_file, segment_word, segment_size, puncs)
    valid_set = PunctuationDataset(valid_file, segment_word, segment_size, puncs)

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')


    print("PREPARING TARGET WEIGHTS")
    # target_weights
    if os.path.isfile("target_weights.pt"):
        target_weights = torch.load("target_weights.pt")
    else:
        y_labels = [int(item) for sublist in [training_set[i]["labels"] for i in range(len(training_set))] for item in sublist]
        target_weights = torch.Tensor(compute_class_weight(class_weight="balanced",
            classes=list(set(y_labels)), y=y_labels)).clamp_max(10).cuda(0)
        torch.save(target_weights, "target_weights.pt")


    print('INITIALIZING MODEL...')
    output_size = len(puncs)
    bert_punc = BertPunc_ner(output_size).cuda(0)


    print('TRAINING TOP LAYER...')
    for name, param in bert_punc.named_parameters():
        #if 'classifier' not in name: # classifier layer
        if not name.startswith("bert"):
            param.requires_grad = True
        else:
            param.requires_grad = False
    criterion = nn.NLLLoss(weight=target_weights, reduction='none')
    optimizer = optim.AdamW(bert_punc.parameters(), lr=learning_rate_top)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=300, num_training_steps=len(training_loader))
    bert_punc, optimizer, best_val_loss = train(bert_punc, criterion, optimizer, scheduler, epochs_top,
        training_loader, valid_loader, save_path, punctuation_enc, iterations_top, best_val_loss=1e9)


    print('TRAINING ALL LAYER...')
    for p in bert_punc.parameters():
        p.requires_grad = True
    criterion = nn.NLLLoss(weight=target_weights, reduction='none')
    optimizer = optim.AdamW(bert_punc.parameters(), lr=learning_rate_all)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=300, num_training_steps=len(training_loader))
    bert_punc, optimizer, best_val_loss = train(bert_punc, criterion, optimizer, scheduler, epochs_all,
        training_loader, valid_loader, save_path, punctuation_enc, iterations_all, best_val_loss=best_val_loss)

