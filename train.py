import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
import pandas as pd
import os
SEED = 30
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def compute_accuracy(predictions, true_labels):
    flat_predictions = [item for sublist in predictions for item in sublist]
    predicted_label_ids = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    return accuracy_score(flat_true_labels, predicted_label_ids)


def evaluate_model(model, test_dataloader, device, wandb):
    model.eval()
    predictions, true_labels = [], []
    total_test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input, mask, label = batch[0].to(device), batch[1].to(device),batch[2].to(device)
            
            outputs = model(input, attention_mask=mask, labels=label)
            logits = outputs.logits
            loss = outputs.loss
            total_test_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

            logits = logits.detach().cpu().numpy()
            label_ids = label.cpu().numpy()
            predictions.append(logits)
            true_labels.append(label_ids)

    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_accuracy = correct / total
    return predictions, true_labels,avg_test_loss, avg_test_accuracy

def train_model(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs, device, wandb):
    for epoch in tqdm(range(epochs)):
        total_train_loss = 0 
        correct = 0
        total = 0

        model.train()
        for step, batch in enumerate(train_dataloader):

            input, mask, label = batch[0].to(device), batch[1].to(device),batch[2].to(device)
            model.zero_grad()
            outputs = model(input, attention_mask=mask, labels=label)    
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = correct / total
        _,_,avg_test_loss, avg_test_accuracy = evaluate_model(model, test_dataloader, device, wandb)
        wandb.log({"avg_train_loss": avg_train_loss, "avg_train_acc":avg_train_accuracy}, step=epoch+1)
        wandb.log({"avg_test_loss": avg_test_loss, "avg_test_acc":avg_test_accuracy}, step=epoch+1)
        print(f'Average Training Loss: {avg_train_loss:.4f}')


def generate_report(predictions, true_labels, label_dict):
    flat_predictions = [item for sublist in predictions for item in sublist]
    predicted_label_ids = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    report = classification_report(flat_true_labels, predicted_label_ids, target_names=label_dict.keys())
    return report

def predict_df(model, df, inputs, attention_masks, labels, num_to_label, device, batch_size):
    """
    ### Annotates a dataframe with a new column of model's predictions for each relative sample
    """
    model.eval()
    
    # Prepare DataLoader
    dataset = TensorDataset(inputs, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count())
    predictions = []
    
    # Predict in batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting DF: "):
            input, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            outputs = model(x=input, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=1).cpu().tolist()
            predictions.extend(batch_predictions)
    
    # Add predictions to DataFrame
    test = [num_to_label[x] for x in predictions]
    df['prediction'] = test
    df = df.drop('label', axis=1)
    return df
    