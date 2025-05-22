#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERT for Abstractive Text Summarization

This script implements a text summarization model using BERT for encoder and
a decoder for generating summaries.

Use Case Examples:
- News article summarization
- Document summarization
- Meeting notes generation
- Report summarization
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    BertModel,
    BertConfig,
    BertGenerationDecoder,
    AdamW,
    get_linear_schedule_with_warmup
)
from rouge_score import rouge_scorer
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global configuration variables
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 128
BATCH_SIZE = 4  # Smaller batch size due to model size
EPOCHS = 3
LEARNING_RATE = 3e-5
MODEL_NAME = "bert-base-uncased"  # For the encoder
DECODER_NAME = "bert-base-uncased"  # For the decoder


class SummarizationDataset(Dataset):
    """Custom dataset for text summarization tasks"""

    def __init__(self, text_encodings, summary_encodings):
        """
        Initialize dataset with encoded texts and summaries

        Args:
            text_encodings: Encoded input texts
            summary_encodings: Encoded output summaries
        """
        self.text_encodings = text_encodings
        self.summary_encodings = summary_encodings

    def __len__(self):
        """Return dataset length"""
        return len(self.text_encodings['input_ids'])

    def __getitem__(self, idx):
        """Get dataset item at index"""
        item = {key: val[idx] for key, val in self.text_encodings.items()}

        # Set up decoder inputs and labels
        item['decoder_input_ids'] = self.summary_encodings['input_ids'][idx]
        item['decoder_attention_mask'] = self.summary_encodings['attention_mask'][idx]
        item['labels'] = self.summary_encodings['input_ids'][idx].clone()

        # Replace padding token id with -100 so it's ignored in loss calculation
        item['labels'][item['labels'] == 0] = -100

        return item


def load_summarization_data(data_path):
    """
    Load data for text summarization

    Expected format:
    CSV: columns 'text' and 'summary'
    OR JSON: list of objects with 'text' and 'summary' keys

    Args:
        data_path (str): Path to the data file

    Returns:
        tuple: (texts, summaries)
    """
    texts = []
    summaries = []

    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        required_columns = ['text', 'summary']

        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")

        texts = df['text'].tolist()
        summaries = df['summary'].tolist()

    elif data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            if 'text' in item and 'summary' in item:
                texts.append(item['text'])
                summaries.append(item['summary'])

    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    # Filter out empty texts or summaries
    filtered_data = [(t, s) for t, s in zip(texts, summaries) if t and s]
    if len(filtered_data) < len(texts):
        print(f"Filtered out {len(texts) - len(filtered_data)} examples with empty text or summary")

    texts, summaries = zip(*filtered_data) if filtered_data else ([], [])

    # Print data statistics
    print(f"Total number of examples: {len(texts)}")

    if texts:
        print(f"Average text length: {sum(len(t.split()) for t in texts) / len(texts):.1f} words")
        print(f"Average summary length: {sum(len(s.split()) for s in summaries) / len(summaries):.1f} words")

        # Print length distribution
        text_lengths = [len(t.split()) for t in texts]
        summary_lengths = [len(s.split()) for s in summaries]

        print("\nText length distribution:")
        print(f"  Min: {min(text_lengths)}")
        print(f"  Max: {max(text_lengths)}")
        print(f"  Median: {np.median(text_lengths)}")

        print("\nSummary length distribution:")
        print(f"  Min: {min(summary_lengths)}")
        print(f"  Max: {max(summary_lengths)}")
        print(f"  Median: {np.median(summary_lengths)}")

    return texts, summaries


def prepare_summarization_data(texts, summaries, tokenizer, test_size=0.2):
    """
    Split and prepare data for text summarization

    Args:
        texts (list): List of input texts
        summaries (list): List of summaries
        tokenizer: BERT tokenizer
        test_size (float): Proportion of data for testing

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Split data into train and test sets
    train_texts, test_texts, train_summaries, test_summaries = train_test_split(
        texts, summaries, test_size=test_size, random_state=42
    )

    # Split test set into validation and final test sets
    val_texts, test_texts, val_summaries, test_summaries = train_test_split(
        test_texts, test_summaries, test_size=0.5, random_state=42
    )

    print(f"Train examples: {len(train_texts)}")
    print(f"Validation examples: {len(val_texts)}")
    print(f"Test examples: {len(test_texts)}")

    # Tokenize inputs
    train_text_encodings = tokenizer(
        train_texts,
        max_length=MAX_INPUT_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    val_text_encodings = tokenizer(
        val_texts,
        max_length=MAX_INPUT_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    test_text_encodings = tokenizer(
        test_texts,
        max_length=MAX_INPUT_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Tokenize summaries (outputs)
    train_summary_encodings = tokenizer(
        train_summaries,
        max_length=MAX_OUTPUT_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    val_summary_encodings = tokenizer(
        val_summaries,
        max_length=MAX_OUTPUT_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    test_summary_encodings = tokenizer(
        test_summaries,
        max_length=MAX_OUTPUT_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Create datasets
    train_dataset = SummarizationDataset(train_text_encodings, train_summary_encodings)
    val_dataset = SummarizationDataset(val_text_encodings, val_summary_encodings)
    test_dataset = SummarizationDataset(test_text_encodings, test_summary_encodings)

    return (
        train_dataset, val_dataset, test_dataset,
        train_texts, val_texts, test_texts,
        train_summaries, val_summaries, test_summaries
    )


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """
    Create dataloaders for training and evaluation

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
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


def initialize_model():
    """
    Initialize a BERT encoder-decoder model for summarization

    Returns:
        model: Initialized model
    """
    # Option 1: Use the built-in encoder-decoder model (BertEncoderDecoder)
    # model = EncoderDecoderModel.from_encoder_decoder_pretrained(MODEL_NAME, DECODER_NAME)

    # Option 2: Initialize from scratch with custom configuration
    # This gives more control over the configuration
    encoder = BertModel.from_pretrained(MODEL_NAME)

    # Initialize a decoder with the same configuration as the encoder
    decoder_config = BertConfig.from_pretrained(DECODER_NAME)
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder = BertGenerationDecoder.from_pretrained(DECODER_NAME, config=decoder_config)

    # Create the encoder-decoder model
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    # Set special tokens
    model.config.decoder_start_token_id = 101  # [CLS] token ID for BERT
    model.config.eos_token_id = 102  # [SEP] token ID for BERT
    model.config.pad_token_id = 0  # [PAD] token ID for BERT

    # Set generation parameters
    model.config.max_length = MAX_OUTPUT_LEN
    model.config.min_length = 10
    model.config.no_repeat_ngram_size = 2
    model.config.early_stopping = True
    model.config.length_penalty = 1.0
    model.config.num_beams = 4

    return model


def train_summarization_model(model, train_dataloader, val_dataloader, optimizer, scheduler):
    """
    Train the summarization model and evaluate on validation set

    Args:
        model: Encoder-decoder model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler

    Returns:
        model: Trained model
        history: Training history
    """
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': []
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
            batch = {k: v.to(device) for k, v in batch.items()}

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                decoder_input_ids=batch['decoder_input_ids'],
                decoder_attention_mask=batch['decoder_attention_mask'],
                labels=batch['labels']
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

        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=batch['decoder_input_ids'],
                    decoder_attention_mask=batch['decoder_attention_mask'],
                    labels=batch['labels']
                )

                loss = outputs.loss
                val_losses.append(loss.item())

        # Calculate average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        history['val_loss'].append(avg_val_loss)

        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Generate a few examples for validation during training
        generate_validation_examples(model, val_dataloader, tokenizer, epoch)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_summarization_model.pt')
            print("Saved best model!")

    # Load best model
    model.load_state_dict(torch.load('best_summarization_model.pt'))
    return model, history


def generate_validation_examples(model, val_dataloader, tokenizer, epoch):
    """
    Generate and print example summaries during validation

    Args:
        model: Encoder-decoder model
        val_dataloader: Validation data loader
        tokenizer: BERT tokenizer
        epoch: Current epoch number
    """
    model.eval()

    # Get one batch
    batch = next(iter(val_dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}

    # Generate summaries
    generated_ids = model.generate(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        max_length=MAX_OUTPUT_LEN,
        num_beams=4,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    # Decode generated summaries
    generated_summaries = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in generated_ids
    ]

    # Decode actual summaries (labels)
    actual_summaries = [
        tokenizer.decode(
            [token_id for token_id in ids if token_id != -100],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        for ids in batch['labels']
    ]

    # Print a few examples
    print("\nExample summaries:")
    for i in range(min(2, len(generated_summaries))):
        print(f"\nInput: {tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)[:100]}...")
        print(f"Actual: {actual_summaries[i]}")
        print(f"Generated: {generated_summaries[i]}")


def evaluate_summarization(model, test_dataloader, tokenizer, test_texts, test_summaries):
    """
    Evaluate the summarization model on the test set

    Args:
        model: Encoder-decoder model
        test_dataloader: Test data loader
        tokenizer: BERT tokenizer
        test_texts: List of test texts
        test_summaries: List of test summaries

    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    generated_summaries = []

    # Set up ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Generate summaries
            generated_ids = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=MAX_OUTPUT_LEN,
                num_beams=4,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )

            # Decode generated summaries
            batch_summaries = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in generated_ids
            ]

            generated_summaries.extend(batch_summaries)

    # Calculate ROUGE scores
    rouge_scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }

    for gen_summary, ref_summary in zip(generated_summaries, test_summaries):
        # Skip empty summaries
        if not gen_summary or not ref_summary:
            continue

        scores = scorer.score(ref_summary, gen_summary)

        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)

    # Calculate average scores
    avg_rouge1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
    avg_rouge2 = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
    avg_rougeL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])

    print("\nTest Results:")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")

    # Show some examples
    print("\nExample summaries:")
    num_examples = min(5, len(test_texts))
    for i in range(num_examples):
        print(f"\nExample {i + 1}:")
        print(f"Input: {test_texts[i][:100]}...")
        print(f"Reference: {test_summaries[i]}")
        print(f"Generated: {generated_summaries[i]}")

    return {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'generated_summaries': generated_summaries
    }


def plot_summarization_training_history(history):
    """
    Plot training and validation losses for summarization

    Args:
        history (dict): Training history
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Summarization Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('summarization_training_history.png')
    plt.close()


def generate_summary(text, model, tokenizer):
    """
    Generate a summary for a single text input

    Args:
        text (str): Input text
        model: Encoder-decoder model
        tokenizer: BERT tokenizer

    Returns:
        str: Generated summary
    """
    # Tokenize the input
    inputs = tokenizer(
        [text],
        max_length=MAX_INPUT_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate summary
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=MAX_OUTPUT_LEN,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

    # Decode generated summary
    summary = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    print("\nSummarization:")
    print(f"Input: {text[:100]}...")
    print(f"Generated Summary: {summary}")

    return summary


def save_summarization_model_artifacts(model, tokenizer, model_path='./saved_summarization_model'):
    """
    Save summarization model artifacts for later use

    Args:
        model: Encoder-decoder model
        tokenizer: BERT tokenizer
        model_path (str): Path to save the model
    """
    import pickle

    os.makedirs(model_path, exist_ok=True)

    # Save model
    model.save_pretrained(model_path)

    # Save tokenizer
    tokenizer.save_pretrained(model_path)

    # Save configuration
    config = {
        'max_input_len': MAX_INPUT_LEN,
        'max_output_len': MAX_OUTPUT_LEN
    }

    with open(os.path.join(model_path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print(f"Summarization model artifacts saved to {model_path}")


def load_summarization_model_artifacts(model_path='./saved_summarization_model'):
    """
    Load summarization model artifacts for inference

    Args:
        model_path (str): Path to the saved model

    Returns:
        tuple: (model, tokenizer)
    """
    import pickle

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Load model
    model = EncoderDecoderModel.from_pretrained(model_path).to(device)

    # Load configuration
    with open(os.path.join(model_path, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    # Update global variables if needed
    global MAX_INPUT_LEN, MAX_OUTPUT_LEN
    MAX_INPUT_LEN = config['max_input_len']
    MAX_OUTPUT_LEN = config['max_output_len']

    return model, tokenizer


def main():
    """Main function to run the script"""
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Example: Replace 'summarization_data.json' with your actual data file
    texts, summaries = load_summarization_data('summarization_data.json')

    # Prepare data
    (
        train_dataset, val_dataset, test_dataset,
        train_texts, val_texts, test_texts,
        train_summaries, val_summaries, test_summaries
    ) = prepare_summarization_data(texts, summaries, tokenizer)

    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    # Initialize model
    model = initialize_model().to(device)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Train model
    model, history = train_summarization_model(model, train_dataloader, val_dataloader, optimizer, scheduler)

    # Plot training history
    plot_summarization_training_history(history)

    # Evaluate on test set
    evaluate_summarization(model, test_dataloader, tokenizer, test_texts, test_summaries)

    # Save the trained model and artifacts
    save_summarization_model_artifacts(model, tokenizer)

    # Example summary generation
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes 
    actions that maximize its chance of achieving its goals. The term "artificial intelligence" had previously been used to describe machines that 
    mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving." This definition 
    has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how 
    intelligence can be articulated.
    """
    generated_summary = generate_summary(sample_text, model, tokenizer)


if __name__ == "__main__":
    main()