from torch import nn
from transformers import CamembertModel

class BertPunc_ner(nn.Module):

    def __init__(self, output_size):
        super(BertPunc_ner, self).__init__()
        self.output_size = output_size
        self.bert = CamembertModel.from_pretrained('camembert-base',
                                    num_labels = output_size,
                                    output_attentions = False,
                                    output_hidden_states = False)
        self.dropout1 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(768, 1568)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(1568, self.output_size)

    def forward(self, x):
        output = self.bert(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        output = output[0]
        output = self.activation(self.linear1(output))
        output = self.dropout2(output)
        output = self.linear2(output)
        output = nn.functional.log_softmax(output, dim=-1)
        return output
