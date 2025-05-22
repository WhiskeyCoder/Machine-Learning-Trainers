#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERT Multi-Task Learning Model

This script implements a multi-task learning model using BERT where a single model
is trained to perform multiple classification tasks simultaneously.

Use Case Examples:
- Joint intent classification and entity recognition
- Toxic comment classification with multiple toxicity subtypes
- Product reviews with sentiment and aspect category classification
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global configuration variables
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
MODEL_NAME = "bert-base-uncased"  # Can be changed to other BERT variants


class BertForMultiTaskClassification(nn.Module):
    """Custom BERT model for multi-task classification"""

    def __init__(self, num_labels_dict, bert_model_name=MODEL_NAME):
        """
        Initialize multi-task BERT model

        Args:
            num_labels_dict (dict): Dictionary mapping task names to number of labels
            bert_model_name (str): Name of the pre-trained BERT model
        """
        super(BertForMultiTaskClassification, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Hidden size of BERT outputs
        self.hidden_size = self.bert.config.hidden_size

        # Create classification heads for each task
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(self.hidden_size, num_labels)
            for task, num_labels in num_labels_dict.items()
        })

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, token_type_ids, task=None, labels=None):
        """
        Forward pass

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            task (str, optional): Task name for single task forward pass
            labels (torch.Tensor, optional): Labels for loss calculation

        Returns:
            dict: Dictionary containing logits (and loss if labels provided) for specified tasks
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get the [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        result = {}

        # If task is specified, only compute logits for that task
        if task:
            logits = self.classifiers[task](pooled_output)
            result['logits'] = logits

            # Calculate loss if labels are provided
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.classifiers[task].out_features), labels.view(-1))
                result['loss'] = loss

            return result

        # Otherwise, compute logits for all tasks
        for task_name, classifier in self.classifiers.items():
            result[task_name] = {'logits': classifier(pooled_output)}

        # Calculate losses if labels are provided
        if labels is not None:
            for task_name in self.classifiers.keys():
                if task_name in labels:
                    loss_fct = nn.CrossEntropyLoss()
                    task_loss = loss_fct(
                        result[task_name]['logits'].view(-1, self.classifiers[task_name].out_features),
                        labels[task_name].view(-1)
                    )
                    result[task_name]['loss'] = task_loss

        return result


class MultiTaskDataset(Dataset):
    """Custom dataset for multi-task learning"""

    def __init__(self, texts, task_labels_dict, tokenizer, max_len):
        """
        Initialize dataset with texts and labels for multiple tasks

        Args:
            texts (list): List of text strings
            task_labels_dict (dict): Dictionary mapping task names to labels
            tokenizer: BERT tokenizer
            max_len (int): Maximum sequence length
        """
        self.texts = texts
        self.task_labels_dict = task_labels_dict
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_names = list(task_labels_dict.keys())

    def __len__(self):
        """Return dataset length"""
        return len(self.texts)

    def __getitem__(self, idx):
        """Get dataset item at index"""
        text = str(self.texts[idx])

        # Collect labels for all tasks
        labels = {task: self.task_labels_dict[task][idx] for task in self.task_names}

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

        # Prepare item with text features and labels for all tasks
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }

        # Add labels for each task
        for task in self.task_names:
            item[f'{task}_labels'] = torch.tensor(labels[task], dtype=torch.long)

        return item


def load_and_preprocess_multitask_data(data_path):
    """
    Load data from JSON or CSV file for multi-task learning

    Args:
        data_path (str): Path to the data file

    Returns:
        tuple: (texts, task_labels_dict, label_encoders_dict, num_labels_dict)
    """
    from sklearn.preprocessing import LabelEncoder

    # Check file extension
    if data_path.endswith('.json'):
        # JSON format expected:
        # [{"text": "...", "task1_label": "...", "task2_label": "..."}, ...]
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Extract texts and identify tasks
        texts = [item['text'] for item in data]

        # Identify task column names (any column that ends with _label)
        task_columns = [col for col in data[0].keys() if col.endswith('_label')]
        task_names = [col.replace('_label', '') for col in task_columns]

        # Create a dictionary to hold labels for each task
        raw_task_labels = {task: [item[f'{task}_label'] for item in data] for task in task_names}

    elif data_path.endswith('.csv'):
        # CSV format with 'text' column and one column per task
        df = pd.read_csv(data_path)
        texts = df['text'].values

        # Identify task columns (any column except 'text')
        task_columns = [col for col in df.columns if col != 'text']
        task_names = task_columns

        # Create a dictionary to hold labels for each task
        raw_task_labels = {task: df[task].values for task in task_names}
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    # Initialize dictionaries
    label_encoders_dict = {}
    task_labels_dict = {}
    num_labels_dict = {}

    # Encode labels for each task
    for task in task_names:
        # Create label encoder for this task
        label_encoder = LabelEncoder()
        task_labels_dict[task] = label_encoder.fit_transform(raw_task_labels[task])
        label_encoders_dict[task] = label_encoder
        num_labels_dict[task] = len(label_encoder.classes_)

        # Print task information
        print(f"Task: {task}")
        print(f"  Number of classes: {num_labels_dict[task]}")
        print(f"  Classes: {label_encoder.classes_}")

        # Check class distribution
        class_counts = pd.Series(task_labels_dict[task]).value_counts().sort_index()
        print("  Class distribution:")
        for i, count in enumerate(class_counts):
            class_name = label_encoder.inverse_transform([i])[0]
            print(f"    {class_name} (Class {i}): {count}")

    return texts, task_labels_dict, label_encoders_dict, num_labels_dict


def prepare_multitask_dataloaders(texts, task_labels_dict, tokenizer, test_size=0.2):
    """
    Split data and create train/validation/test dataloaders for multi-task learning

    Args:
        texts (list): List of text strings
        task_labels_dict (dict): Dictionary mapping task names to labels
        tokenizer: BERT tokenizer
        test_size (float): Proportion of data for testing

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # Get task names
    task_names = list(task_labels_dict.keys())

    # Pick the first task to stratify on (could be more sophisticated)
    stratify_task = task_names[0]
    stratify_labels = task_labels_dict[stratify_task]

    # Split into train and temporary test sets
    train_indices, test_indices = train_test_split(
        range(len(texts)),
        test_size=test_size,
        random_state=42,
        stratify=stratify_labels
    )

    # Split temporary test set into validation and final test sets
    test_stratify_labels = [stratify_labels[i] for i in test_indices]
    val_indices, test_indices_final = train_test_split(
        test_indices,
        test_size=0.5,
        random_state=42,
        stratify=test_stratify_labels
    )

    # Extract data for each split
    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    test_texts = [texts[i] for i in test_indices_final]

    # Extract labels for each task and split
    train_task_labels = {task: [labels[i] for i in train_indices] for task, labels in task_labels_dict.items()}
    val_task_labels = {task: [labels[i] for i in val_indices] for task, labels in task_labels_dict.items()}
    test_task_labels = {task: [labels[i] for i in test_indices_final] for task, labels in task_labels_dict.items()}

    # Create datasets
    train_dataset = MultiTaskDataset(
        texts=train_texts,
        task_labels_dict=train_task_labels,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    val_dataset = MultiTaskDataset(
        texts=val_texts,
        task_labels_dict=val_task_labels,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    test_dataset = MultiTaskDataset(
        texts=test_texts,
        task_labels_dict=test_task_labels,
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


def train_multitask_model(model, train_dataloader, val_dataloader, optimizer, scheduler, task_names):
    """
    Train the multi-task model and evaluate on validation set

    Args:
        model: Multi-task BERT model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        task_names: List of task names

    Returns:
        model: Trained model
        history: Training history
    """
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
    }

    # Initialize task-specific metrics
    for task in task_names:
        history[f'{task}_val_accuracy'] = []
        history[f'{task}_val_f1'] = []

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

            # Prepare labels dictionary
            labels = {task: batch[f'{task}_labels'].to(device) for task in task_names}

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            # Calculate total loss as sum of all task losses
            total_loss = sum(outputs[task]['loss'] for task in task_names)
            train_losses.append(total_loss.item())

            # Backward pass and optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Update progress bar
            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=total_loss.item())

        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_losses = []
        task_predictions = {task: [] for task in task_names}
        task_true_labels = {task: [] for task in task_names}

        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)

                # Prepare labels dictionary
                labels = {task: batch[f'{task}_labels'].to(device) for task in task_names}

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                # Calculate total loss
                total_loss = sum(outputs[task]['loss'] for task in task_names)
                val_losses.append(total_loss.item())

                # Get predictions for each task
                for task in task_names:
                    preds = torch.argmax(outputs[task]['logits'], dim=1).cpu().numpy()
                    true_labs = labels[task].cpu().numpy()

                    task_predictions[task].extend(preds)
                    task_true_labels[task].extend(true_labs)

        # Calculate validation metrics
        avg_val_loss = sum(val_losses) / len(val_losses)
        history['val_loss'].append(avg_val_loss)

        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Calculate and print metrics for each task
        for task in task_names:
            task_accuracy = accuracy_score(task_true_labels[task], task_predictions[task])
            task_f1 = f1_score(task_true_labels[task], task_predictions[task], average='weighted')

            history[f'{task}_val_accuracy'].append(task_accuracy)
            history[f'{task}_val_f1'].append(task_f1)

            print(f"Task: {task}")
            print(f"  Validation Accuracy: {task_accuracy:.4f}")
            print(f"  Validation F1 Score (weighted): {task_f1:.4f}")

        # Print detailed classification report for last epoch
        if epoch == EPOCHS - 1:
            print("\nDetailed Classification Reports:")
            for task in task_names:
                print(f"\nTask: {task}")
                print(classification_report(
                    task_true_labels[task],
                    task_predictions[task],
                    digits=4
                ))

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_multitask_model.pt')
            print("Saved best model!")

    # Load best model
    model.load_state_dict(torch.load('best_multitask_model.pt'))
    return model, history


def evaluate_multitask_on_test(model, test_dataloader, label_encoders_dict, task_names):
    """
    Evaluate the multi-task model on the test set

    Args:
        model: Multi-task BERT model
        test_dataloader: Test data loader
        label_encoders_dict: Dictionary mapping task names to label encoders
        task_names: List of task names

    Returns:
        dict: Evaluation metrics for each task
    """
    model.eval()
    task_predictions = {task: [] for task in task_names}
    task_true_labels = {task: [] for task in task_names}

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Get predictions for each task
            for task in task_names:
                preds = torch.argmax(outputs[task]['logits'], dim=1).cpu().numpy()
                true_labs = batch[f'{task}_labels'].numpy()

                task_predictions[task].extend(preds)
                task_true_labels[task].extend(true_labs)

    # Calculate evaluation metrics for each task
    results = {}

    print("\nTest Results:")
    for task in task_names:
        accuracy = accuracy_score(task_true_labels[task], task_predictions[task])
        f1 = f1_score(task_true_labels[task], task_predictions[task], average='weighted')

        print(f"\nTask: {task}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test F1 Score (weighted): {f1:.4f}")
        print("\n  Detailed Classification Report:")
        print(classification_report(
            task_true_labels[task],
            task_predictions[task],
            target_names=label_encoders_dict[task].classes_,
            digits=4
        ))

        # Create confusion matrix
        plt.figure(figsize=(10, 8))
        cm = pd.crosstab(
            pd.Series(task_true_labels[task], name='Actual'),
            pd.Series(task_predictions[task], name='Predicted'),
            rownames=['Actual'],
            colnames=['Predicted'],
            normalize='index'
        )

        # Map numeric labels back to string labels
        cm.index = [label_encoders_dict[task].inverse_transform([i])[0] for i in cm.index]
        cm.columns = [label_encoders_dict[task].inverse_transform([i])[0] for i in cm.columns]

        sns.heatmap(cm, annot=True, fmt='.2f', cmap="Blues")
        plt.title(f'Confusion Matrix - {task}')
        plt.savefig(f'confusion_matrix_{task}.png')
        plt.close()

        results[task] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': task_predictions[task],
            'true_labels': task_true_labels[task]
        }

    return results


def plot_multitask_training_history(history, task_names):
    """
    Plot training and validation metrics for multi-task learning

    Args:
        history (dict): Training history
        task_names (list): List of task names
    """
    num_tasks = len(task_names)

    plt.figure(figsize=(15, 5 + num_tasks * 3))

    # Plot losses
    plt.subplot(num_tasks + 1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot task-specific metrics
    for i, task in enumerate(task_names):
        # Plot accuracy
        plt.subplot(num_tasks + 1, 2, i * 2 + 3)
        plt.plot(history[f'{task}_val_accuracy'], label=f'{task} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Task: {task} - Validation Accuracy')
        plt.legend()

        # Plot F1 score
        plt.subplot(num_tasks + 1, 2, i * 2 + 4)
        plt.plot(history[f'{task}_val_f1'], label=f'{task} F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title(f'Task: {task} - Validation F1 Score')
        plt.legend()

    plt.tight_layout()
    plt.savefig('multitask_training_history.png')
    plt.close()


def predict_multitask(text, model, tokenizer, label_encoders_dict, task_names):
    """
    Make multi-task predictions for a single text input

    Args:
        text (str): Input text
        model: Multi-task BERT model
        tokenizer: BERT tokenizer
        label_encoders_dict: Dictionary mapping task names to label encoders
        task_names: List of task names

    Returns:
        dict: Dictionary mapping task names to predictions
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

    # Get predictions for each task
    results = {}
    print("\nPredictions:")
    for task in task_names:
        logits = outputs[task]['logits']
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0][pred_class].item()

        # Convert numeric prediction to string label
        pred_label = label_encoders_dict[task].inverse_transform([pred_class])[0]

        # Get probabilities for all classes
        all_probs = probs[0].cpu().numpy()
        class_probs = {
            label_encoders_dict[task].inverse_transform([i])[0]: prob.item()
            for i, prob in enumerate(probs[0])
        }

        results[task] = {
            'label': pred_label,
            'class_index': pred_class,
            'confidence': pred_prob,
            'class_probabilities': class_probs
        }

        print(f"Task: {task}")
        print(f"  Prediction: {pred_label} (Class {pred_class})")
        print(f"  Confidence: {pred_prob:.4f}")
        print("  Probabilities for all classes:")
        for label, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"    {label}: {prob:.4f}")

    return results


def save_multitask_model_artifacts(model, tokenizer, label_encoders_dict, task_names,
                                   model_path='./saved_multitask_model'):
    """
    Save model artifacts for later use

    Args:
        model: Multi-task BERT model
        tokenizer: BERT tokenizer
        label_encoders_dict: Dictionary mapping task names to label encoders
        task_names: List of task names
        model_path (str): Path to save the model
    """
    import pickle

    os.makedirs(model_path, exist_ok=True)

    # Save model state dict
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))

    # Save tokenizer
    tokenizer.save_pretrained(model_path)

    # Save label encoders
    with open(os.path.join(model_path, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders_dict, f)

    # Save configuration
    config = {
        'max_len': MAX_LEN,
        'model_name': MODEL_NAME,
        'task_names': task_names,
        'num_labels_dict': {task: len(label_encoders_dict[task].classes_) for task in task_names}
    }

    with open(os.path.join(model_path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print(f"Multi-task model artifacts saved to {model_path}")


def load_multitask_model_artifacts(model_path='./saved_multitask_model'):
    """
    Load multi-task model artifacts for inference

    Args:
        model_path (str): Path to the saved model

    Returns:
        tuple: (model, tokenizer, label_encoders_dict, task_names)
    """
    import pickle

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Load label encoders
    with open(os.path.join(model_path, 'label_encoders.pkl'), 'rb') as f:
        label_encoders_dict = pickle.load(f)

    # Load configuration
    with open(os.path.join(model_path, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    # Get task names and number of labels
    task_names = config['task_names']
    num_labels_dict = config['num_labels_dict']

    # Initialize model
    model = BertForMultiTaskClassification(num_labels_dict).to(device)

    # Load model weights
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))

    return model, tokenizer, label_encoders_dict, task_names


def main():
    """Main function to run the script"""
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Example: Replace 'data.json' with your actual data file
    texts, task_labels_dict, label_encoders_dict, num_labels_dict = load_and_preprocess_multitask_data('data.json')

    # Get task names
    task_names = list(task_labels_dict.keys())

    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_multitask_dataloaders(
        texts, task_labels_dict, tokenizer
    )

    # Load model
    model = BertForMultiTaskClassification(num_labels_dict).to(device)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Train model
    model, history = train_multitask_model(model, train_dataloader, val_dataloader, optimizer, scheduler, task_names)

    # Plot training history
    plot_multitask_training_history(history, task_names)

    # Evaluate on test set
    evaluate_multitask_on_test(model, test_dataloader, label_encoders_dict, task_names)

    # Save the trained model and artifacts
    save_multitask_model_artifacts(model, tokenizer, label_encoders_dict, task_names)

    # Example prediction
    sample_text = "This is a sample text for multi-task prediction"
    predictions = predict_multitask(sample_text, model, tokenizer, label_encoders_dict, task_names)


if __name__ == "__main__":
    main()