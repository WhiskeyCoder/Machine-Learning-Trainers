#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERT Binary Classification Model

This script implements a binary classification model using BERT.
It handles data preprocessing, model training, evaluation, and inference.

Use Case Examples:
- Sentiment analysis (positive/negative)
- Spam detection
- Intent classification
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global configuration variables
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_NAME = "bert-base-uncased"  # Can be changed to other BERT variants


class TextClassificationDataset(Dataset):
    """Custom dataset for text classification tasks"""

    def __init__(self, texts, labels, tokenizer, max_len):
        """
        Initialize dataset with texts and labels

        Args:
            texts (list): List of text strings
            labels (list): List of labels (0 or 1 for binary classification)
            tokenizer: BERT tokenizer
            max_len (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Return dataset length"""
        return len(self.texts)

    def __getitem__(self, idx):
        """Get dataset item at index"""
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(data_path):
    """
    Load data from CSV file
    Expected format: CSV with 'text' and 'label' columns

    Args:
        data_path (str): Path to the CSV file

    Returns:
        tuple: (texts, labels)
    """
    df = pd.read_csv(data_path)
    texts = df['text'].values
    labels = df['label'].values
    return texts, labels


def prepare_dataloaders(texts, labels, tokenizer, test_size=0.2):
    """
    Split data and create train/validation dataloaders

    Args:
        texts (list): List of text strings
        labels (list): List of labels
        tokenizer: BERT tokenizer
        test_size (float): Proportion of data for validation

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )

    # Create datasets
    train_dataset = TextClassificationDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    val_dataset = TextClassificationDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return train_dataloader, val_dataloader


def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler):
    """
    Train the model and evaluate on validation set

    Args:
        model: BERT model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler

    Returns:
        model: Trained model
    """
    best_val_accuracy = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Training phase
        model.train()
        train_losses = []

        # Use tqdm for progress bar
        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            loss = outputs.loss
            train_losses.append(loss.item())

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Update progress bar
            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_losses = []
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                loss = outputs.loss
                val_losses.append(loss.item())

                # Get predictions
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                true_labs = labels.cpu().numpy()

                predictions.extend(preds)
                true_labels.extend(true_labs)

        # Calculate validation metrics
        val_accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved best model!")

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model


def predict(text, model, tokenizer):
    """
    Make prediction for a single text input

    Args:
        text (str): Input text
        model: BERT model
        tokenizer: BERT tokenizer

    Returns:
        tuple: (prediction, probability)
    """
    # Prepare input
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    # Set model to evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    # Get prediction and probability
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    prob = probs[0][pred].item()

    return pred, prob


def main():
    """Main function to run the script"""
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Example: Replace 'data.csv' with your actual data file
    # Expected format: CSV with 'text' and 'label' columns
    texts, labels = load_data('data.csv')

    # Prepare dataloaders
    train_dataloader, val_dataloader = prepare_dataloaders(texts, labels, tokenizer)

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # Binary classification
        output_attentions=False,
        output_hidden_states=False
    ).to(device)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Train model
    model = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler)

    # Save the trained model and tokenizer
    model_path = './saved_model'
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    # Example prediction
    sample_text = "This is a sample text for prediction"
    prediction, probability = predict(sample_text, model, tokenizer)
    print(f"Prediction: {prediction} (Confidence: {probability:.4f})")


if __name__ == "__main__":
    main()