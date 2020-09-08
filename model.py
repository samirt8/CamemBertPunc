from torch import nn
from transformers import CamembertForMaskedLM


class BertPunc(nn.Module):

    def __init__(self, segment_size, output_size, dropout):
        super(BertPunc, self).__init__()
        self.bert = CamembertForMaskedLM.from_pretrained('camembert-base')
        self.bert_vocab_size = 32005
        self.output_size = output_size
        self.segment_size = segment_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.fc = nn.Linear(self.bert_vocab_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.bert(x)
        x = x[0]
        x = x.view(-1, self.bert_vocab_size)
        b_size = x.size(0)
        x = self.fc(x)
        x = x.view(-1, self.output_size, self.segment_size)
        return x
