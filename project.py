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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'\nDevice being used: ', device, '\n')

SEED = 30
EPOCHS = 10
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
IMDB_dataset = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

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

def clean(data):
    def strip_text(text):
        text = re.sub(r"(<[^>]+>)|(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)", "", text)
        return text
    stripped_text_dataset = copy.deepcopy(data)
    stripped_text_dataset['review'].apply(strip_text)
    stripped_text_dataset.head()

    
def main():
    #print(IMDB_dataset.head())

    """
    Data Preprocessing/Prep
    """
    dataset = clean(IMDB_dataset)
    input_ids, attention_masks, labels, label_dict = preprocess_data(dataset)
    train_dataset, test_dataset = split_dataset(input_ids, attention_masks, labels)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=8)


    """
    Model Prep
    """
    model = create_model(label_dict)
    optimizer = AdamW(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=0.01)
    epochs = 1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)

if __name__ == '__main__':
    main()