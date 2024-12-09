import torch
import random
from torch import nn
from transformers import BertForSequenceClassification, BertModel

"""
Helper Func/Classes
"""
def layer_info(model):
    for name, param in model.named_parameters():
        status = "Trainable" if param.requires_grad else "Frozen"
        print(f"{name}: {status}")

class Outputs:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits

"""
Model Architectures
"""
class BertLP0(nn.Module):
    def __init__(self, backbone_name, out_dim=2):
        super().__init__()
        self.backbone = BertForSequenceClassification.from_pretrained(backbone_name, num_labels=out_dim)
        for param in self.backbone.bert.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x, attention_mask, labels):
        return self.backbone(input_ids=x, attention_mask=attention_mask, labels=labels)

class BertLP1(nn.Module):
    """
    input -> BERT -> N extra layer MLP (post-pooling) -> softmax predictions
    """
    def __init__(self, backbone_name, hidden_dim=768, MLP_depth=1, out_dim=2):
        super().__init__()
        if backbone_name == "google/bert_uncased_L-8_H-512_A-8": hidden_dim=512
        self.backbone = BertModel.from_pretrained(backbone_name) 
        for param in self.backbone.parameters(): 
            param.requires_grad = False

        layers = []
        for _ in range(MLP_depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, attention_mask, labels=None):
        x = self.backbone(input_ids=x, attention_mask=attention_mask, return_dict=True)
        pooled_output = x.pooler_output  # Shape: (batch_size, hidden_size)
        logits = self.mlp(pooled_output)  # Shape: (batch_size, out_dim)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return Outputs(loss, logits)


class BertLPA0(nn.Module):
    def __init__(self, backbone_name, MLP_depth=0, hidden_dim=768, dropout_prob=0.1, out_dim=2):
        super().__init__()

        """
        Attn
        """
        self.backbone = BertModel.from_pretrained(backbone_name)
        if backbone_name == "google/bert_uncased_L-8_H-512_A-8": hidden_dim = 512
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            dropout=dropout_prob,
            batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

        """
        MLP
        """
        layers = []
        for _ in range(MLP_depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, attention_mask=None, labels=None):
        x = self.backbone(input_ids=x,attention_mask=attention_mask,return_dict=True)
        
        x = x.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size) = (64, 308, 768)
        attention_output, _ = self.attention(x,x,x,None)
        attention_output = self.layer_norm(attention_output + x)
        attention_output = self.dropout(attention_output)
        logits = attention_output[:, 0, :]
        logits = self.mlp(logits)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        return Outputs(loss, logits)