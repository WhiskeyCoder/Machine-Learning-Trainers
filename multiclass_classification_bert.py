#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERT Multiclass Classification Model

This script implements a multiclass classification model using BERT.
It handles data preprocessing, model training, evaluation, and inference.

Use Case Examples:
- Topic classification
- Intent classification with multiple categories
- Product categorization
- News article categorization
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global configuration variables
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
MODEL_NAME = "bert-base-uncased"  # Can be changed to other BERT variants


class MulticlassDataset(Dataset):
    """Custom dataset for multiclass classification tasks"""

    def __init__(self, texts, labels, tokenizer, max_len):
        """
        Initialize dataset with texts and labels

        Args:
            texts (list): List of text strings
            labels (list): List of numeric labels (0, 1, 2, etc.)
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


def load_and_preprocess_data(data_path):
    """
    Load data from CSV file and preprocess for multiclass classification
    Expected format: CSV with 'text' and 'label' columns (where label is a string)

    Args:
        data_path (str): Path to the CSV file

    Returns:
        tuple: (texts, labels, label_encoder, num_classes)
    """
    df = pd.read_csv(data_path)
    texts = df['text'].values

    # Encode string labels to numeric values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'].values)

    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")

    # Check class distribution
    class_counts = pd.Series(labels).value_counts().sort_index()
    print("Class distribution:")
    for i, count in enumerate(class_counts):
        class_name = label_encoder.inverse_transform([i])[0]
        print(f"  {class_name} (Class {i}): {count}")

    return texts, labels, label_encoder, num_classes


def prepare_dataloaders(texts, labels, tokenizer, test_size=0.2, val_size=0.1):
    """
    Split data and create train/validation/test dataloaders

    Args:
        texts (list): List of text strings
        labels (list): List of labels
        tokenizer: BERT tokenizer
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # First split into train and temporary test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Then split temporary test set into validation and final test sets
    # Adjust val_size to account for the previous split
    val_size_adjusted = val_size / (1 - test_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        test_texts, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )

    # Create datasets
    train_dataset = MulticlassDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    val_dataset = MulticlassDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    test_dataset = MulticlassDataset(
        texts=test_texts,
        labels=test_labels,
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

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return train_dataloader, val_dataloader, test_dataloader


def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, num_classes):
    """
    Train the model and evaluate on validation set

    Args:
        model: BERT model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_classes: Number of classes

    Returns:
        model: Trained model
        history: Training history
    """
    best_val_accuracy = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }

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
        history['train_loss'].append(avg_train_loss)
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
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_accuracy = accuracy_score(true_labels, predictions)
        val_f1 = f1_score(true_labels, predictions, average='weighted')

        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score (weighted): {val_f1:.4f}")

        # Print detailed classification report for last epoch
        if epoch == EPOCHS - 1:
            print("\nDetailed Classification Report:")
            print(classification_report(true_labels, predictions, digits=4))

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_multiclass_model.pt')
            print("Saved best model!")

    # Load best model
    model.load_state_dict(torch.load('best_multiclass_model.pt'))
    return model, history


def evaluate_on_test(model, test_dataloader, label_encoder):
    """
    Evaluate the model on the test set

    Args:
        model: BERT model
        test_dataloader: Test data loader
        label_encoder: Label encoder used to convert between string and numeric labels

    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Get predictions
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            true_labs = labels.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(true_labs)

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')

    print("\nTest Results:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score (weighted): {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predictions,
                                target_names=label_encoder.classes_, digits=4))

    # Create confusion matrix
    plt.figure(figsize=(10, 8))
    cm = pd.crosstab(
        pd.Series(true_labels, name='Actual'),
        pd.Series(predictions, name='Predicted'),
        rownames=['Actual'],
        colnames=['Predicted'],
        normalize='index'
    )

    # Map numeric labels back to string labels
    cm.index = [label_encoder.inverse_transform([i])[0] for i in cm.index]
    cm.columns = [label_encoder.inverse_transform([i])[0] for i in cm.columns]

    sns.heatmap(cm, annot=True, fmt='.2f', cmap="Blues")
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': predictions,
        'true_labels': true_labels
    }


def plot_training_history(history):
    """
    Plot training and validation metrics

    Args:
        history (dict): Training history
    """
    plt.figure(figsize=(12, 4))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Accuracy')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def predict(text, model, tokenizer, label_encoder):
    """
    Make prediction for a single text input

    Args:
        text (str): Input text
        model: BERT model
        tokenizer: BERT tokenizer
        label_encoder: Label encoder to convert numeric predictions to string labels

    Returns:
        tuple: (prediction_label, prediction_class_index, probability)
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
    pred_class = torch.argmax(probs, dim=1).item()
    pred_prob = probs[0][pred_class].item()

    # Convert numeric prediction to string label
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    # Get probabilities for all classes
    all_probs = probs[0].cpu().numpy()
    class_probs = {label_encoder.inverse_transform([i])[0]: prob.item()
                   for i, prob in enumerate(probs[0])}

    print(f"Prediction: {pred_label} (Class {pred_class})")
    print(f"Confidence: {pred_prob:.4f}")
    print("Probabilities for all classes:")
    for label, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {prob:.4f}")

    return pred_label, pred_class, pred_prob


def save_model_artifacts(model, tokenizer, label_encoder, model_path='./saved_multiclass_model'):
    """
    Save model artifacts for later use

    Args:
        model: BERT model
        tokenizer: BERT tokenizer
        label_encoder: Label encoder
        model_path (str): Path to save the model
    """
    import pickle

    os.makedirs(model_path, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Save label encoder
    with open(os.path.join(model_path, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save configuration
    config = {
        'max_len': MAX_LEN,
        'model_name': MODEL_NAME,
        'num_classes': len(label_encoder.classes_)
    }

    with open(os.path.join(model_path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print(f"Model artifacts saved to {model_path}")


def load_model_artifacts(model_path='./saved_multiclass_model'):
    """
    Load model artifacts for inference

    Args:
        model_path (str): Path to the saved model

    Returns:
        tuple: (model, tokenizer, label_encoder)
    """
    import pickle

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Load label encoder
    with open(os.path.join(model_path, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)

    # Load configuration
    with open(os.path.join(model_path, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=config['num_classes']
    ).to(device)

    return model, tokenizer, label_encoder


def main():
    """Main function to run the script"""
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Example: Replace 'data.csv' with your actual data file
    # Expected format: CSV with 'text' and 'label' columns
    texts, labels, label_encoder, num_classes = load_and_preprocess_data('data.csv')

    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(texts, labels, tokenizer)

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
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
    model, history = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, num_classes)

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    evaluate_on_test(model, test_dataloader, label_encoder)

    # Save the trained model and artifacts
    save_model_artifacts(model, tokenizer, label_encoder)

    # Example prediction
    sample_text = "This is a sample text for multiclass prediction"
    pred_label, pred_class, probability = predict(sample_text, model, tokenizer, label_encoder)


if __name__ == "__main__":
    main()