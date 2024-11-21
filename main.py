# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import wandb
import random
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import random
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import math
import torch
import tqdm
import re
import copy
from train import train_model, evaluate_model, compute_accuracy, generate_report
import kaggle

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'\nDevice being used: ', device, '\n')
SEED = 30
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 0.01
MAX_TOKEN_LENGTH = 64
DATASET_PATH = "./data/"
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
wandb.init(
    project="245 Final Project",
    entity="ehaddad2-university-of-rochester-org",    
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_token_length": MAX_TOKEN_LENGTH
    }
)

def preprocess_data(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    labels = df["type"].unique().tolist()
    label_dict = {label: i for i, label in enumerate(labels)}
    df['label'] = df["type"].replace(label_dict)
    input_ids = []
    attention_masks = []

    for text in df["text"]:
        encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=64, pad_to_max_length=True, return_attention_mask=True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(df['label'].values), label_dict

def create_model(label_dict):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_dict))
    return model

def split_dataset(input_ids, attention_masks, labels):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


    
def main():
    #print(IMDB_dataset.head())

    """
    Data Preprocessing/Prep
    """
    input_ids, attention_masks, labels, label_dict = preprocess_data(dataset)
    train_dataset, test_dataset = split_dataset(input_ids, attention_masks, labels)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=8)


    """
    Model Prep
    """
    model = create_model(label_dict)
    model.to(device)

    """
    Training
    """
    optimizer = AdamW(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * EPOCHS)
    train_model(model, train_dataloader, optimizer, scheduler, EPOCHS)
    predictions, true_labels = evaluate_model(model, test_dataloader)
    accuracy = compute_accuracy(predictions, true_labels)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(generate_report(predictions=predictions, true_labels=true_labels, label_dict=label_dict))
    torch.save(model.state_dict(), "bert_imdb_model.pth")
    wandb.save("bert_imdb_model.pth")
    wandb.finish()
if __name__ == '__main__':
    main()