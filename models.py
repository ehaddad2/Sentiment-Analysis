import torch
import random
from torch import nn
from transformers import BertForSequenceClassification

def layer_info(model):
    for name, param in model.named_parameters():
        status = "Trainable" if param.requires_grad else "Frozen"
        print(f"{name}: {status}")

class BertLP0(nn.Module):
    def __init__(self, out_dim=2):
        super(BertLP0, self).__init__()
        self.backbone = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=out_dim)

        for param in self.backbone.bert.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x, attention_mask, label):
        return self.backbone(input_ids=x, attention_mask=attention_mask, labels=label)