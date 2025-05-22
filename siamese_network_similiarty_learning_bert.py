#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERT Siamese Network for Similarity Learning

This script implements a Siamese network using BERT for learning text similarity.
It can be used for tasks like semantic similarity, duplicate detection, and more.

Use Case Examples:
- Semantic similarity between sentences
- Duplicate question detection
- Document similarity
- Information retrieval
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global configuration variables
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
MODEL_NAME = "bert-base-uncased"  # Can be changed to other BERT variants


class SiameseBERT(nn.Module):
    """Siamese network with BERT encoders for similarity learning"""

    def __init__(self, bert_model_name=MODEL_NAME, dropout_prob=0.1):
        """
        Initialize Siamese BERT model

        Args:
            bert_model_name (str): Name of the pre-trained BERT model
            dropout_prob (float): Dropout probability
        """
        super(SiameseBERT, self).__init__()

        # BERT encoder (shared weights for both inputs)
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze BERT parameters to save memory and speed up training
        # Uncomment the following lines if you want to freeze BERT
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # Output dimensionality of BERT
        self.hidden_size = self.bert.config.hidden_size

        # Similarity scoring module
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.hidden_size * 3, 1)  # 3x features: text1, text2, |text1-text2|

    def forward_one(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass for one text input

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs

        Returns:
            torch.Tensor: Text embedding (CLS token representation)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Use the [CLS] token representation as the text embedding
        cls_output = outputs.pooler_output
        return cls_output

    def forward(self, input_ids_1, attention_mask_1, token_type_ids_1,
                input_ids_2, attention_mask_2, token_type_ids_2):
        """
        Forward pass for the Siamese network

        Args:
            input_ids_1, input_ids_2: Input token IDs for text pairs
            attention_mask_1, attention_mask_2: Attention masks for text pairs
            token_type_ids_1, token_type_ids_2: Token type IDs for text pairs

        Returns:
            tuple: (similarity_score, text1_embedding, text2_embedding)
        """
        # Get embeddings for both texts
        embedding_1 = self.forward_one(input_ids_1, attention_mask_1, token_type_ids_1)
        embedding_2 = self.forward_one(input_ids_2, attention_mask_2, token_type_ids_2)

        # Combine features: embedding_1, embedding_2, |embedding_1 - embedding_2|
        abs_diff = torch.abs(embedding_1 - embedding_2)
        combined_features = torch.cat([embedding_1, embedding_2, abs_diff], dim=1)

        # Apply dropout and get similarity score
        combined_features = self.dropout(combined_features)
        similarity_score = self.classifier(combined_features)

        return similarity_score, embedding_1, embedding_2


class SiameseDataset(Dataset):
    """Custom dataset for Siamese BERT network"""

    def __init__(self, text_pairs, labels, tokenizer, max_len):
        """
        Initialize dataset

        Args:
            text_pairs (list): List of text pairs (text1, text2)
            labels (list): List of similarity labels (0 or 1)
            tokenizer: BERT tokenizer
            max_len (int): Maximum sequence length
        """
        self.text_pairs = text_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Return the dataset size"""
        return len(self.labels)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        text1, text2 = self.text_pairs[idx]
        label = self.labels[idx]

        # Tokenize the first text
        encoding1 = self.tokenizer(
            text1,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize the second text
        encoding2 = self.tokenizer(
            text2,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension added by tokenizer
        encoding1 = {k: v.squeeze(0) for k, v in encoding1.items()}
        encoding2 = {k: v.squeeze(0) for k, v in encoding2.items()}

        return {
            'input_ids_1': encoding1['input_ids'],
            'attention_mask_1': encoding1['attention_mask'],
            'token_type_ids_1': encoding1['token_type_ids'],
            'input_ids_2': encoding2['input_ids'],
            'attention_mask_2': encoding2['attention_mask'],
            'token_type_ids_2': encoding2['token_type_ids'],
            'labels': torch.tensor(label, dtype=torch.float)
        }


def load_similarity_data(data_path):
    """
    Load data for similarity learning

    Expected format:
    CSV: columns 'text1', 'text2', 'is_similar'
    OR JSON: list of objects with 'text1', 'text2', 'is_similar' keys

    Args:
        data_path (str): Path to the data file

    Returns:
        tuple: (text_pairs, labels)
    """
    text_pairs = []
    labels = []

    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        required_columns = ['text1', 'text2', 'is_similar']

        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")

        text_pairs = list(zip(df['text1'].tolist(), df['text2'].tolist()))
        labels = df['is_similar'].tolist()

    elif data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            if 'text1' in item and 'text2' in item and 'is_similar' in item:
                text_pairs.append((item['text1'], item['text2']))
                labels.append(item['is_similar'])

    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    # Print data statistics
    print(f"Total number of examples: {len(text_pairs)}")
    print(f"Positive examples (similar): {sum(labels)}")
    print(f"Negative examples (not similar): {len(labels) - sum(labels)}")

    if text_pairs:
        # Print length distribution
        text1_lengths = [len(t[0].split()) for t in text_pairs]
        text2_lengths = [len(t[1].split()) for t in text_pairs]

        print("\nText1 length distribution:")
        print(f"  Min: {min(text1_lengths)}")
        print(f"  Max: {max(text1_lengths)}")
        print(f"  Median: {np.median(text1_lengths)}")

        print("\nText2 length distribution:")
        print(f"  Min: {min(text2_lengths)}")
        print(f"  Max: {max(text2_lengths)}")
        print(f"  Median: {np.median(text2_lengths)}")

    return text_pairs, labels


def prepare_similarity_dataloaders(text_pairs, labels, tokenizer, test_size=0.2):
    """
    Split data and create train/validation/test dataloaders for similarity learning

    Args:
        text_pairs (list): List of text pairs
        labels (list): List of similarity labels
        tokenizer: BERT tokenizer
        test_size (float): Proportion of data for testing

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # Split data into train and test sets
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        text_pairs, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Split test set into validation and final test sets
    val_pairs, test_pairs, val_labels, test_labels = train_test_split(
        test_pairs, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )

    print(f"Train examples: {len(train_pairs)}")
    print(f"Validation examples: {len(val_pairs)}")
    print(f"Test examples: {len(test_pairs)}")

    # Create datasets
    train_dataset = SiameseDataset(train_pairs, train_labels, tokenizer, MAX_LEN)
    val_dataset = SiameseDataset(val_pairs, val_labels, tokenizer, MAX_LEN)
    test_dataset = SiameseDataset(test_pairs, test_labels, tokenizer, MAX_LEN)

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


def train_similarity_model(model, train_dataloader, val_dataloader, optimizer, scheduler):
    """
    Train the similarity model and evaluate on validation set

    Args:
        model: Siamese BERT model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler

    Returns:
        model: Trained model
        history: Training history
    """
    best_val_auc = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': []
    }

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Training phase
        model.train()
        train_losses = []

        # Use tqdm for progress bar
        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            similarity_scores, _, _ = model(
                batch['input_ids_1'],
                batch['attention_mask_1'],
                batch['token_type_ids_1'],
                batch['input_ids_2'],
                batch['attention_mask_2'],
                batch['token_type_ids_2']
            )

            # Calculate loss
            loss = criterion(similarity_scores.view(-1), batch['labels'])
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
        val_predictions = []
        val_true_labels = []
        val_scores = []

        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                similarity_scores, _, _ = model(
                    batch['input_ids_1'],
                    batch['attention_mask_1'],
                    batch['token_type_ids_1'],
                    batch['input_ids_2'],
                    batch['attention_mask_2'],
                    batch['token_type_ids_2']
                )

                # Calculate loss
                loss = criterion(similarity_scores.view(-1), batch['labels'])
                val_losses.append(loss.item())

                # Get predictions
                probs = torch.sigmoid(similarity_scores).view(-1).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                true_labs = batch['labels'].cpu().numpy()

                val_predictions.extend(preds)
                val_true_labels.extend(true_labs)
                val_scores.extend(probs)

        # Calculate validation metrics
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        val_auc = roc_auc_score(val_true_labels, val_scores)

        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_auc'].append(val_auc)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")

        # Print detailed classification report
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_true_labels, val_predictions, average='binary'
        )
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_similarity_model.pt')
            print("Saved best model!")

    # Load best model
    model.load_state_dict(torch.load('best_similarity_model.pt'))
    return model, history


def evaluate_similarity(model, test_dataloader):
    """
    Evaluate the similarity model on the test set

    Args:
        model: Siamese BERT model
        test_dataloader: Test data loader

    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    test_predictions = []
    test_true_labels = []
    test_scores = []

    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    test_losses = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            similarity_scores, _, _ = model(
                batch['input_ids_1'],
                batch['attention_mask_1'],
                batch['token_type_ids_1'],
                batch['input_ids_2'],
                batch['attention_mask_2'],
                batch['token_type_ids_2']
            )

            # Calculate loss
            loss = criterion(similarity_scores.view(-1), batch['labels'])
            test_losses.append(loss.item())

            # Get predictions
            probs = torch.sigmoid(similarity_scores).view(-1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            true_labs = batch['labels'].cpu().numpy()

            test_predictions.extend(preds)
            test_true_labels.extend(true_labs)
            test_scores.extend(probs)

    # Calculate test metrics
    avg_test_loss = sum(test_losses) / len(test_losses)
    test_accuracy = accuracy_score(test_true_labels, test_predictions)
    test_auc = roc_auc_score(test_true_labels, test_scores)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_true_labels, test_predictions, average='binary'
    )

    print("\nTest Results:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Plot ROC curve
    plot_roc_curve(test_true_labels, test_scores)

    # Plot confusion matrix
    plot_confusion_matrix(test_true_labels, test_predictions)

    return {
        'accuracy': test_accuracy,
        'auc': test_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': avg_test_loss
    }


def plot_roc_curve(y_true, y_scores):
    """
    Plot ROC curve

    Args:
        y_true: True labels
        y_scores: Predicted scores
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()


def plot_similarity_training_history(history):
    """
    Plot training and validation metrics for similarity learning

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
    plt.plot(history['val_auc'], label='AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()

    plt.tight_layout()
    plt.savefig('similarity_training_history.png')
    plt.close()


def predict_similarity(text1, text2, model, tokenizer):
    """
    Predict similarity between two texts

    Args:
        text1 (str): First text
        text2 (str): Second text
        model: Siamese BERT model
        tokenizer: BERT tokenizer

    Returns:
        tuple: (similarity_score, similarity_label)
    """
    # Tokenize the first text
    encoding1 = tokenizer(
        text1,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Tokenize the second text
    encoding2 = tokenizer(
        text2,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move to device
    encoding1 = {k: v.to(device) for k, v in encoding1.items()}
    encoding2 = {k: v.to(device) for k, v in encoding2.items()}

    # Set model to evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        similarity_score, emb1, emb2 = model(
            encoding1['input_ids'],
            encoding1['attention_mask'],
            encoding1['token_type_ids'],
            encoding2['input_ids'],
            encoding2['attention_mask'],
            encoding2['token_type_ids']
        )

    # Get probability and label
    probability = torch.sigmoid(similarity_score).item()
    label = 1 if probability >= 0.5 else 0

    print("\nSimilarity Prediction:")
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity Score: {probability:.4f}")
    print(f"Similarity Label: {'Similar' if label == 1 else 'Not Similar'}")

    return probability, label


def save_similarity_model_artifacts(model, tokenizer, model_path='./saved_similarity_model'):
    """
    Save similarity model artifacts for later use

    Args:
        model: Siamese BERT model
        tokenizer: BERT tokenizer
        model_path (str): Path to save the model
    """
    import pickle

    os.makedirs(model_path, exist_ok=True)

    # Save model state dict
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))

    # Save tokenizer
    tokenizer.save_pretrained(model_path)

    # Save configuration
    config = {
        'max_len': MAX_LEN,
        'model_name': MODEL_NAME
    }

    with open(os.path.join(model_path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print(f"Similarity model artifacts saved to {model_path}")


def load_similarity_model_artifacts(model_path='./saved_similarity_model'):
    """
    Load similarity model artifacts for inference

    Args:
        model_path (str): Path to the saved model

    Returns:
        tuple: (model, tokenizer)
    """
    import pickle

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Load configuration
    with open(os.path.join(model_path, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    # Update global variables if needed
    global MAX_LEN, MODEL_NAME
    MAX_LEN = config['max_len']
    MODEL_NAME = config['model_name']

    # Initialize model
    model = SiameseBERT(MODEL_NAME).to(device)

    # Load model weights
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))

    return model, tokenizer


def main():
    """Main function to run the script"""
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Example: Replace 'similarity_data.json' with your actual data file
    text_pairs, labels = load_similarity_data('similarity_data.json')

    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_similarity_dataloaders(
        text_pairs, labels, tokenizer
    )

    # Initialize model
    model = SiameseBERT().to(device)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Train model
    model, history = train_similarity_model(model, train_dataloader, val_dataloader, optimizer, scheduler)

    # Plot training history
    plot_similarity_training_history(history)

    # Evaluate on test set
    evaluate_similarity(model, test_dataloader)

    # Save the trained model and artifacts
    save_similarity_model_artifacts(model, tokenizer)

    # Example prediction
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A fast fox leaps over a sleeping canine."
    similarity_score, similarity_label = predict_similarity(text1, text2, model, tokenizer)


if __name__ == "__main__":
    main()