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
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset, DataLoader
import math
import torch
from tqdm import tqdm
from train import train_model, evaluate_model, compute_accuracy, generate_report, predict_df
import models
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'\nDevice being used: ', device, '\n')
SEED = 30
EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.01
MAX_TOKEN_LENGTH = 308 #mean token length
DATASET_PATH = "/home/elias/Deep Learning/Projects/NLP/Sentiment-Analysis/data/IMDB Dataset.csv"
#SAVE_DS_PATH = "/home/elias/Deep Learning/Projects/NLP/Sentiment-Analysis/data/Attn+MLP2+CleanDataset_pred.csv"
MODEL_PATH = "/home/elias/Deep Learning/Projects/NLP/Sentiment-Analysis/models/Attn+MLP1+UncleanedDataset.pth"
BACKBONE_NAME = "bert-base-uncased"
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

wandb.init(
    project="245 Final Project",
    entity="achen99-university-of-rochester",
    name="Attn+MLP1+UncleanedDataset_funrun",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_token_length": MAX_TOKEN_LENGTH
    }
)

def split_dataset(input_ids, attention_masks, labels):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED))

def preprocess_data(df, max_token_length, backbone_name):
    tokenizer = BertTokenizer.from_pretrained(backbone_name)
    labels = df["sentiment"].unique().tolist()
    label_to_num = {label: i for i, label in enumerate(labels)}
    num_to_label = {i:label for i, label in enumerate(labels)}
    df['label'] = df["sentiment"].replace(label_to_num)
    input_ids = []
    attention_masks = []
    for text in tqdm(df["review"], desc="Tokenizing reviews"):
        encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_token_length, pad_to_max_length=True, return_attention_mask=True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    labels = torch.tensor(df['label'].values)
    return torch.tensor(input_ids), torch.tensor(attention_masks), labels, label_to_num, num_to_label

def main():

    """
    Model Prep
    """
    saved_model = False
    model = models.BertLPA0(backbone_name=BACKBONE_NAME, MLP_depth=1)

    if os.path.exists(MODEL_PATH):
        print("\nFound saved model, using that")
        saved_model = True
        model.load_state_dict(torch.load(MODEL_PATH))
    else: print("\nNo saved model found, training from scratch")
    model.to(device)

    """
    Data Preprocessing/Prep
    """
    dataset = pd.read_csv(DATASET_PATH)
    inputs, attention_masks, labels, label_to_num, num_to_label = preprocess_data(dataset, MAX_TOKEN_LENGTH, BACKBONE_NAME)
    train_dataset, test_dataset = split_dataset(inputs, attention_masks, labels)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=os.cpu_count())
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=os.cpu_count())

    """
    Training
    """
    optimizer = AdamW(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * EPOCHS)
    if not saved_model: train_model(model, train_dataloader, test_dataloader, optimizer, scheduler, EPOCHS, device=device, wandb=wandb)
    
    """
    Eval
    """
    torch.cuda.empty_cache()
    predictions, true_labels,_,_ = evaluate_model(model, test_dataloader, device=device, wandb=wandb)
    accuracy = compute_accuracy(predictions, true_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(generate_report(predictions=predictions, true_labels=true_labels, label_dict=label_to_num))
    #predict_df(model, dataset, inputs, attention_masks, labels, num_to_label, device, batch_size=500).to_csv(SAVE_DS_PATH, index=False)
    if not saved_model:
        torch.save(model.state_dict(), MODEL_PATH)
        wandb.save(MODEL_PATH)
    wandb.finish()

if __name__ == '__main__':
    main()