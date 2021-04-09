from torch import nn
from transformers import CamembertForMaskedLM, CamembertForTokenClassification

class BertPunc_ner(nn.Module):

    def __init__(self, output_size):
        super(BertPunc_ner, self).__init__()
        self.output_size = output_size
        self.bert = CamembertForTokenClassification.from_pretrained('camembert-base',
                                    num_labels = output_size,
                                    output_attentions = True,
                                    output_hidden_states = False)

    def forward(self, x):
        output = self.bert(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        return output
