from torch import nn
from transformers import CamembertForMaskedLM, CamembertForTokenClassification


class BertPunc(nn.Module):

    def __init__(self, segment_size, output_size, dropout):
        super(BertPunc, self).__init__()
        self.bert = CamembertForMaskedLM.from_pretrained('camembert-base')
        self.bert_vocab_size = 32005
        self.output_size = output_size
        self.segment_size = segment_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.fc = nn.Linear(self.bert_vocab_size, output_size)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.bert(x)
        x = x[0]
        x = x.view(-1, self.bert_vocab_size)
        x = self.fc(x)
        x = x.view(-1, self.output_size, self.segment_size)
        return x

class BertPunc_ner(nn.Module):

    def __init__(self, segment_size, output_size):
        super(BertPunc_ner, self).__init__()
        self.segment_size = segment_size
        self.output_size = output_size
        self.bert = CamembertForTokenClassification.from_pretrained('camembert-base',
                                    num_labels = output_size,
                                    output_attentions = False, #True
                                    output_hidden_states = False)

    #def forward(self, x, attention_masks):
        #x = self.bert(x, attention_mask=attention_masks)
    def forward(self, x):
        x = self.bert(x)
        x = x[0]
        x = x.view(-1, self.output_size, self.segment_size)
        return x
