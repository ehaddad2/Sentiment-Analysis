import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, classification_report
SEED = 30
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def compute_accuracy(predictions, true_labels):
    flat_predictions = [item for sublist in predictions for item in sublist]
    predicted_label_ids = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    return accuracy_score(flat_true_labels, predicted_label_ids)

def train_model(model, train_dataloader, optimizer, scheduler, epochs, device, wandb):
    for epoch in range(epochs):
        total_train_loss = 0
        for step, input, mask, label in enumerate(train_dataloader):
            input, mask, label = input.to(device), mask.to(device),label.to(device)
            model.zero_grad()
            outputs = model(input, attention_mask=mask, labels=label)    
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print('\r [epoch: %03d][iter: %04d][loss: %.6f]'%(epoch+1, step, loss.item()))
        avg_train_loss = total_train_loss / len(train_dataloader)
        wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_train_loss})
        print(f'Average Training Loss: {avg_train_loss:.4f}')

def evaluate_model(model, test_dataloader, device, wandb):
    model.eval()
    predictions, true_labels = [], []

    for batch in test_dataloader:
        input, mask, label = input.to(device), mask.to(device),label.to(device)

        with torch.no_grad():
            outputs = model(input, attention_mask=mask)
            logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = label.cpu().numpy()
        predictions.append(logits)
        true_labels.append(label_ids)

    return predictions, true_labels

def generate_report(predictions, true_labels, label_dict):
    flat_predictions = [item for sublist in predictions for item in sublist]
    predicted_label_ids = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    report = classification_report(flat_true_labels, predicted_label_ids, target_names=label_dict.keys())
    return report