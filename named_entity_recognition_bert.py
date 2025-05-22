#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERT for Named Entity Recognition (NER)

This script implements a token classification model using BERT for sequence labeling tasks
such as Named Entity Recognition (NER).

Use Case Examples:
- Named Entity Recognition
- Part-of-Speech (POS) tagging
- Chunk extraction
- Custom entity extraction
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
    BertForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1_score
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
EPOCHS = 4
LEARNING_RATE = 3e-5
MODEL_NAME = "bert-base-uncased"  # Can be changed to other BERT variants


class NERDataset(Dataset):
    """Custom dataset for Named Entity Recognition (NER) tasks"""

    def __init__(self, texts, tags, tokenizer, max_len, tag2idx):
        """
        Initialize dataset with texts and tags

        Args:
            texts (list): List of text strings
            tags (list): List of tag sequences
            tokenizer: BERT tokenizer
            max_len (int): Maximum sequence length
            tag2idx (dict): Mapping from tag to index
        """
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2idx = tag2idx

    def __len__(self):
        """Return dataset length"""
        return len(self.texts)

    def __getitem__(self, idx):
        """Get dataset item at index"""
        text = str(self.texts[idx])
        tags = self.tags[idx]

        # Tokenize the text and map the tokens to their word IDs
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            is_split_into_words=False  # We're not providing pre-tokenized text
        )

        # Remove batch dimension added by the tokenizer
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        special_tokens_mask = inputs["special_tokens_mask"].squeeze(0).bool()
        offset_mapping = inputs["offset_mapping"].squeeze(0)

        # Create the label sequence with the same length as input_ids
        labels = torch.ones(self.max_len, dtype=torch.long) * -100  # -100 is ignored by the loss function

        # Create a mapping from token positions to word positions
        token_to_word_map = {}
        current_word_idx = -1
        prev_word_end = -1

        for i, (start, end) in enumerate(offset_mapping):
            # Skip special tokens
            if special_tokens_mask[i]:
                continue

            # If this token starts a new word
            if start.item() != prev_word_end:
                current_word_idx += 1

            if current_word_idx < len(tags):
                labels[i] = self.tag2idx.get(tags[current_word_idx], self.tag2idx["O"])

            prev_word_end = end.item()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def load_ner_data(data_path):
    """
    Load data for NER from JSON or CoNLL format

    Expected JSON format:
    [
        {"text": "John Smith works at Microsoft.", "entities": [[0, 10, "PERSON"], [19, 28, "ORG"]]},
        ...
    ]
    OR CoNLL format:
    John B-PER
    Smith I-PER
    works O
    at O
    Microsoft B-ORG
    . O

    Args:
        data_path (str): Path to the data file

    Returns:
        tuple: (texts, tags_list, tag2idx, idx2tag)
    """
    texts = []
    tags_list = []

    # Check if JSON or CoNLL format based on file extension
    if data_path.endswith('.json'):
        # JSON format
        with open(data_path, 'r') as f:
            data = json.load(f)

        for item in data:
            text = item['text']
            entities = item['entities']

            # Create character-level tag mapping
            tag_sequence = ['O'] * len(text)

            for start, end, tag in entities:
                for i in range(start, end):
                    if i == start:
                        tag_sequence[i] = f'B-{tag}'
                    else:
                        tag_sequence[i] = f'I-{tag}'

            # Now convert character-level tags to token-level tags
            # This is a simplification and would need to be adapted to your tokenizer
            words = text.split()
            word_tags = []
            char_idx = 0

            for word in words:
                word_len = len(word)
                # Use the tag of the first character of the word
                word_tags.append(tag_sequence[char_idx])
                char_idx += word_len + 1  # +1 for the space

            texts.append(text)
            tags_list.append(word_tags)

    elif data_path.endswith('.txt') or data_path.endswith('.conll'):
        # CoNLL format
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Split by double newlines to get sentences
        sentences = content.split('\n\n')

        for sentence in sentences:
            if not sentence.strip():
                continue

            lines = sentence.strip().split('\n')
            words = []
            tags = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    tag = parts[-1]  # Last column is the NER tag
                    words.append(word)
                    tags.append(tag)

            if words and tags:
                texts.append(' '.join(words))
                tags_list.append(tags)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    # Create tag mappings
    unique_tags = set()
    for tags in tags_list:
        unique_tags.update(tags)

    print(f"Unique tags: {unique_tags}")

    # Create tag2idx and idx2tag dictionaries
    # Make sure 'O' is always present
    if 'O' not in unique_tags:
        unique_tags.add('O')

    # Sort tags to ensure consistent mapping
    unique_tags = sorted(list(unique_tags))

    tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    print(f"Number of tags: {len(tag2idx)}")
    print(f"Tag mapping: {tag2idx}")

    # Print data statistics
    print(f"Total number of examples: {len(texts)}")

    # Print tag distribution
    all_tags = []
    for tags in tags_list:
        all_tags.extend(tags)

    tag_counts = pd.Series(all_tags).value_counts()
    print("\nTag distribution:")
    for tag, count in tag_counts.items():
        print(f"  {tag}: {count}")

    return texts, tags_list, tag2idx, idx2tag


def prepare_ner_dataloaders(texts, tags_list, tokenizer, tag2idx, test_size=0.2):
    """
    Split data and create train/validation/test dataloaders for NER

    Args:
        texts (list): List of text strings
        tags_list (list): List of tag sequences
        tokenizer: BERT tokenizer
        tag2idx (dict): Mapping from tag to index
        test_size (float): Proportion of data for testing

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # Split into train and test sets
    train_texts, test_texts, train_tags, test_tags = train_test_split(
        texts, tags_list, test_size=test_size, random_state=42
    )

    # Split test set into validation and final test sets
    val_texts, test_texts, val_tags, test_tags = train_test_split(
        test_texts, test_tags, test_size=0.5, random_state=42
    )

    # Create datasets
    train_dataset = NERDataset(
        texts=train_texts,
        tags=train_tags,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        tag2idx=tag2idx
    )

    val_dataset = NERDataset(
        texts=val_texts,
        tags=val_tags,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        tag2idx=tag2idx
    )

    test_dataset = NERDataset(
        texts=test_texts,
        tags=test_tags,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        tag2idx=tag2idx
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

    return (
        train_dataloader, val_dataloader, test_dataloader,
        train_texts, val_texts, test_texts,
        train_tags, val_tags, test_tags
    )


def train_ner_model(model, train_dataloader, val_dataloader, optimizer, scheduler, idx2tag):
    """
    Train the NER model and evaluate on validation set

    Args:
        model: BERT model for token classification
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        idx2tag: Mapping from index to tag

    Returns:
        model: Trained model
        history: Training history
    """
    best_val_f1 = 0
    history = {
        'train_loss': [],
        'val_loss': [],
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
            labels = batch['labels'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                val_losses.append(loss.item())

                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2)

                # Convert predictions and labels to tag sequences
                for i in range(predictions.shape[0]):
                    pred_tags = []
                    true_tags = []

                    for j in range(predictions.shape[1]):
                        if labels[i, j] != -100:  # Ignore padding tokens
                            pred_idx = predictions[i, j].item()
                            true_idx = labels[i, j].item()

                            pred_tags.append(idx2tag[pred_idx])
                            true_tags.append(idx2tag[true_idx])

                    val_predictions.append(pred_tags)
                    val_true_labels.append(true_tags)

        # Calculate validation metrics
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_f1 = seqeval_f1_score(val_true_labels, val_predictions)

        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")

        # Print detailed classification report
        if epoch == EPOCHS - 1:
            print("\nDetailed Classification Report:")
            print(seqeval_report(val_true_labels, val_predictions))

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_ner_model.pt')
            print("Saved best model!")

    # Load best model
    model.load_state_dict(torch.load('best_ner_model.pt'))
    return model, history


def evaluate_ner_on_test(model, test_dataloader, idx2tag, test_texts, test_tags):
    """
    Evaluate the NER model on the test set

    Args:
        model: BERT model for token classification
        test_dataloader: Test data loader
        idx2tag: Mapping from index to tag
        test_texts: List of test texts
        test_tags: List of test tag sequences

    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    test_predictions = []
    test_true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)

            # Convert predictions and labels to tag sequences
            for i in range(predictions.shape[0]):
                pred_tags = []
                true_tags = []

                for j in range(predictions.shape[1]):
                    if labels[i, j] != -100:  # Ignore padding tokens
                        pred_idx = predictions[i, j].item()
                        true_idx = labels[i, j].item()

                        pred_tags.append(idx2tag[pred_idx])
                        true_tags.append(idx2tag[true_idx])

                test_predictions.append(pred_tags)
                test_true_labels.append(true_tags)

    # Calculate test metrics
    test_f1 = seqeval_f1_score(test_true_labels, test_predictions)

    print("\nTest Results:")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(seqeval_report(test_true_labels, test_predictions))

    # Show some examples with predictions
    print("\nExample predictions:")
    num_examples = min(5, len(test_texts))
    for i in range(num_examples):
        words = test_texts[i].split()
        true_tags = test_tags[i]

        # Get corresponding prediction (they might have different lengths due to tokenization)
        # Find the prediction with closest length
        closest_idx = -1
        closest_diff = float('inf')

        for j, pred in enumerate(test_predictions):
            diff = abs(len(pred) - len(true_tags))
            if diff < closest_diff:
                closest_diff = diff
                closest_idx = j
                if diff == 0:
                    break

        if closest_idx >= 0:
            pred_tags = test_predictions[closest_idx]

            # Truncate or pad to match lengths
            min_len = min(len(words), len(true_tags), len(pred_tags))
            words = words[:min_len]
            true_tags = true_tags[:min_len]
            pred_tags = pred_tags[:min_len]

            print(f"\nExample {i + 1}:")
            print("Text:", test_texts[i])
            print("Word\tTrue\tPred")
            print("-" * 30)

            for w, t, p in zip(words, true_tags, pred_tags):
                print(f"{w}\t{t}\t{p}")

    return {
        'f1_score': test_f1,
        'predictions': test_predictions,
        'true_labels': test_true_labels
    }


def plot_ner_training_history(history):
    """
    Plot training and validation metrics for NER

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

    # Plot F1 score
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('ner_training_history.png')
    plt.close()


def predict_ner(text, model, tokenizer, idx2tag):
    """
    Make NER predictions for a single text input

    Args:
        text (str): Input text
        model: BERT model for token classification
        tokenizer: BERT tokenizer
        idx2tag: Mapping from index to tag

    Returns:
        list: List of (word, tag) tuples
    """
    # Tokenize the text
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_offsets_mapping=True
    )

    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    offset_mapping = inputs["offset_mapping"][0].numpy()

    # Set model to evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # Get predictions
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    predictions = predictions[0].cpu().numpy()

    # Map predictions to tokens and then to words
    predicted_tags = []
    words = []
    current_word = ""
    current_tag = ""

    # Skip special tokens ([CLS], [SEP], [PAD])
    for idx, (offset, pred_idx) in enumerate(zip(offset_mapping, predictions)):
        # Skip if special token
        if offset[0] == offset[1]:
            continue

        # Get the predicted tag
        pred_tag = idx2tag[pred_idx]

        # Get the corresponding token
        token = tokenizer.decode([input_ids[0][idx]])

        # If this is a continuation of a word
        if offset[0] > 0 and offset_mapping[idx - 1][1] == offset[0]:
            current_word += token.replace("##", "")
            # Keep the tag from the first token of the word
        else:
            # If we already had a word, add it to results
            if current_word:
                words.append(current_word)
                predicted_tags.append(current_tag)

            # Start a new word
            current_word = token
            current_tag = pred_tag

    # Add the last word if it exists
    if current_word:
        words.append(current_word)
        predicted_tags.append(current_tag)

    # Clean up the words (remove ## and special chars)
    clean_words = []
    for word in words:
        clean_word = word.replace("##", "").strip()
        if clean_word:
            clean_words.append(clean_word)

    # Make sure we have the same number of words and tags
    predicted_tags = predicted_tags[:len(clean_words)]

    # Create list of (word, tag) pairs
    result = list(zip(clean_words, predicted_tags))

    print("\nNER Prediction:")
    print(f"Text: {text}")
    print("Word\tPredicted Tag")
    print("-" * 30)
    for word, tag in result:
        print(f"{word}\t{tag}")

    return result


def save_ner_model_artifacts(model, tokenizer, tag2idx, idx2tag, model_path='./saved_ner_model'):
    """
    Save NER model artifacts for later use

    Args:
        model: BERT model for token classification
        tokenizer: BERT tokenizer
        tag2idx: Mapping from tag to index
        idx2tag: Mapping from index to tag
        model_path (str): Path to save the model
    """
    import pickle

    os.makedirs(model_path, exist_ok=True)

    # Save model
    model.save_pretrained(model_path)

    # Save tokenizer
    tokenizer.save_pretrained(model_path)

    # Save tag mappings
    with open(os.path.join(model_path, 'tag_mappings.pkl'), 'wb') as f:
        pickle.dump({'tag2idx': tag2idx, 'idx2tag': idx2tag}, f)

    # Save configuration
    config = {
        'max_len': MAX_LEN,
        'model_name': MODEL_NAME,
        'num_labels': len(tag2idx)
    }

    with open(os.path.join(model_path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print(f"NER model artifacts saved to {model_path}")


def load_ner_model_artifacts(model_path='./saved_ner_model'):
    """
    Load NER model artifacts for inference

    Args:
        model_path (str): Path to the saved model

    Returns:
        tuple: (model, tokenizer, tag2idx, idx2tag)
    """
    import pickle

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Example: Replace 'ner_data.json' with your actual data file
    texts, tags_list, tag2idx, idx2tag = load_ner_data('ner_data.json')

    # Prepare dataloaders
    (
        train_dataloader, val_dataloader, test_dataloader,
        train_texts, val_texts, test_texts,
        train_tags, val_tags, test_tags
    ) = prepare_ner_dataloaders(texts, tags_list, tokenizer, tag2idx)

    # Load model
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(tag2idx)
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
    model, history = train_ner_model(model, train_dataloader, val_dataloader, optimizer, scheduler, idx2tag)

    # Plot training history
    plot_ner_training_history(history)

    # Evaluate on test set
    evaluate_ner_on_test(model, test_dataloader, idx2tag, test_texts, test_tags)

    # Save the trained model and artifacts
    save_ner_model_artifacts(model, tokenizer, tag2idx, idx2tag)

    # Example prediction
    sample_text = "John Smith works at Google in London."
    entities = predict_ner(sample_text, model, tokenizer, idx2tag)


if __name__ == "__main__":
    main()
    model_path)

    # Load tag mappings
    with open(os.path.join(model_path, 'tag_mappings.pkl'), 'rb') as f:
        mappings = pickle.load(f)
    tag2idx = mappings['tag2idx']
    idx2tag = mappings['idx2tag']

    # Load configuration
    with open(os.path.join(model_path, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    # Load model
    model = BertForTokenClassification.from_pretrained(
    model_path,
    num_labels = config['num_labels']
).to(device)

return model, tokenizer, tag2idx, idx2tag


def main():
    """Main function to run the script"""
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(