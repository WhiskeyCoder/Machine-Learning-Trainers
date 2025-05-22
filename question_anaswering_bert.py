#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERT for Question Answering

This script implements a question answering model using BERT.
The model extracts answers to questions from provided contexts (extractive QA).

Use Case Examples:
- Building a FAQ system
- Document search with answer extraction
- Information extraction from structured documents
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
    BertForQuestionAnswering,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global configuration variables
MAX_LEN = 384  # Many QA datasets use 384 as max length
BATCH_SIZE = 8  # QA models often need smaller batches due to sequence length
EPOCHS = 3
LEARNING_RATE = 3e-5
MODEL_NAME = "bert-base-uncased"  # Can be changed to other BERT variants
DOC_STRIDE = 128  # Stride size when splitting long contexts


class QADataset(Dataset):
    """Custom dataset for Question Answering tasks"""

    def __init__(self, encodings):
        """
        Initialize dataset with encoded examples

        Args:
            encodings (dict): Dictionary of tensors for input_ids, attention_mask, etc.
        """
        self.encodings = encodings

    def __len__(self):
        """Return dataset length"""
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        """Get dataset item at index"""
        return {key: tensor[idx] for key, tensor in self.encodings.items()}


def load_qa_data(data_path):
    """
    Load data for Question Answering

    Expected format (similar to SQuAD):
    {
        "data": [
            {
                "title": "Title",
                "paragraphs": [
                    {
                        "context": "Context text...",
                        "qas": [
                            {
                                "id": "id",
                                "question": "Question text?",
                                "answers": [
                                    {
                                        "text": "Answer text",
                                        "answer_start": 42
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }

    Args:
        data_path (str): Path to the data file

    Returns:
        tuple: (contexts, questions, answers)
    """
    contexts = []
    questions = []
    answer_texts = []
    answer_starts = []

    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle SQuAD-like format
        if 'data' in data:
            for article in data['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']

                    for qa in paragraph['qas']:
                        question = qa['question']

                        # Handle cases with multiple answers
                        if 'answers' in qa and qa['answers']:
                            for answer in qa['answers']:
                                contexts.append(context)
                                questions.append(question)
                                answer_texts.append(answer['text'])
                                answer_starts.append(answer['answer_start'])
                        # Handle cases with no answers (e.g., SQuAD 2.0)
                        else:
                            contexts.append(context)
                            questions.append(question)
                            answer_texts.append("")
                            answer_starts.append(-1)
        # Handle simplified format
        else:
            for item in data:
                contexts.append(item['context'])
                questions.append(item['question'])
                if 'answer' in item and item['answer']:
                    answer_texts.append(item['answer']['text'])
                    answer_starts.append(item['answer']['answer_start'])
                else:
                    answer_texts.append("")
                    answer_starts.append(-1)

    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        required_columns = ['context', 'question', 'answer_text', 'answer_start']

        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")

        contexts = df['context'].tolist()
        questions = df['question'].tolist()
        answer_texts = df['answer_text'].tolist()
        answer_starts = df['answer_start'].tolist()

    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    # Print data statistics
    print(f"Total number of examples: {len(contexts)}")
    print(f"Average context length: {sum(len(ctx) for ctx in contexts) / len(contexts):.1f} chars")
    print(f"Average question length: {sum(len(q) for q in questions) / len(questions):.1f} chars")
    print(f"Average answer length: {sum(len(a) for a in answer_texts) / len(answer_texts):.1f} chars")

    return contexts, questions, answer_texts, answer_starts


def encode_qa_examples(contexts, questions, answer_texts, answer_starts, tokenizer):
    """
    Encode examples for Question Answering

    Args:
        contexts (list): List of context strings
        questions (list): List of question strings
        answer_texts (list): List of answer text strings
        answer_starts (list): List of answer start positions
        tokenizer: BERT tokenizer

    Returns:
        dict: Dictionary of encoded examples for train, validation, and test
    """
    # Split data into train, validation, and test sets
    train_contexts, test_contexts, train_questions, test_questions, train_answers, test_answers, train_starts, test_starts = train_test_split(
        contexts, questions, answer_texts, answer_starts, test_size=0.2, random_state=42
    )

    val_contexts, test_contexts, val_questions, test_questions, val_answers, test_answers, val_starts, test_starts = train_test_split(
        test_contexts, test_questions, test_answers, test_starts, test_size=0.5, random_state=42
    )

    print(f"Train examples: {len(train_contexts)}")
    print(f"Validation examples: {len(val_contexts)}")
    print(f"Test examples: {len(test_contexts)}")

    # Encode each set
    train_encodings = encode_qa_dataset(train_contexts, train_questions, train_answers, train_starts, tokenizer)
    val_encodings = encode_qa_dataset(val_contexts, val_questions, val_answers, val_starts, tokenizer)
    test_encodings = encode_qa_dataset(test_contexts, test_questions, test_answers, test_starts, tokenizer)

    return {
        'train': train_encodings,
        'validation': val_encodings,
        'test': test_encodings
    }


def encode_qa_dataset(contexts, questions, answers, answer_starts, tokenizer):
    """
    Encode a single dataset for Question Answering

    Args:
        contexts (list): List of context strings
        questions (list): List of question strings
        answers (list): List of answer text strings
        answer_starts (list): List of answer start positions
        tokenizer: BERT tokenizer

    Returns:
        encodings: Encoded examples with start and end positions
    """
    # Tokenize questions and contexts
    encodings = tokenizer(
        questions,
        contexts,
        max_length=MAX_LEN,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    # Map from original example idx to feature idx (for chunked examples)
    example_to_feature = encodings.pop("overflow_to_sample_mapping")
    offset_mapping = encodings.pop("offset_mapping").cpu().numpy()

    # Initialize start and end positions for each example
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        # Get the index of the original example this feature came from
        sample_idx = example_to_feature[i]

        # Extract answer information for this example
        answer_text = answers[sample_idx]
        start_char = answer_starts[sample_idx]

        # Skip unanswerable questions
        if start_char == -1 or not answer_text:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Find the token positions that correspond to the answer
        token_start_idx = None
        token_end_idx = None

        # Find tokens that encompass the answer
        for j, (start, end) in enumerate(offset):
            # Skip if it's a special token
            if start == end == 0:
                continue

            # If the token's start char is within or before the answer's start
            if start <= start_char < end:
                token_start_idx = j

            # If the token's end char is within or after (answer_start + len(answer_text))
            end_char = start_char + len(answer_text) - 1  # -1 because end char is inclusive
            if start <= end_char < end:
                token_end_idx = j
                break

        # If the answer is not in this chunk, point to the [CLS] token
        if token_start_idx is None or token_end_idx is None:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(token_start_idx)
            end_positions.append(token_end_idx)

    # Add start and end positions to encodings
    encodings["start_positions"] = torch.tensor(start_positions, dtype=torch.long)
    encodings["end_positions"] = torch.tensor(end_positions, dtype=torch.long)

    return encodings


def prepare_qa_dataloaders(encodings):
    """
    Create train/validation/test dataloaders for Question Answering

    Args:
        encodings (dict): Dictionary of encoded examples

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # Create datasets
    train_dataset = QADataset(encodings['train'])
    val_dataset = QADataset(encodings['validation'])
    test_dataset = QADataset(encodings['test'])

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


def train_qa_model(model, train_dataloader, val_dataloader, optimizer, scheduler):
    """
    Train the Question Answering model and evaluate on validation set

    Args:
        model: BERT model for question answering
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
            outputs = model(**batch)

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
                outputs = model(**batch)

                loss = outputs.loss
                val_losses.append(loss.item())

        # Calculate average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        history['val_loss'].append(avg_val_loss)

        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_qa_model.pt')
            print("Saved best model!")

    # Load best model
    model.load_state_dict(torch.load('best_qa_model.pt'))
    return model, history


def evaluate_qa_exact_match(predictions, references):
    """
    Calculate exact match score for QA evaluation

    Args:
        predictions (list): List of predicted answers
        references (list): List of ground truth answers

    Returns:
        float: Exact match score
    """

    # Normalize answers for comparison
    def normalize_text(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        import re
        import string

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    # Calculate exact match score
    exact_match = 0
    for pred, ref in zip(predictions, references):
        if normalize_text(pred) == normalize_text(ref):
            exact_match += 1

    return exact_match / len(predictions) if predictions else 0


def evaluate_qa_f1(predictions, references):
    """
    Calculate F1 score for QA evaluation

    Args:
        predictions (list): List of predicted answers
        references (list): List of ground truth answers

    Returns:
        float: F1 score
    """
    f1_scores = []

    # Normalize answers for comparison
    def normalize_text(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        import re
        import string

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    # Calculate F1 score
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_text(pred).split()
        ref_tokens = normalize_text(ref).split()

        # If either is empty, score is 0 or 1 depending on if both are empty
        if not pred_tokens or not ref_tokens:
            if not pred_tokens and not ref_tokens:
                f1_scores.append(1.0)
            else:
                f1_scores.append(0.0)
            continue

        # Count common tokens
        common_tokens = set(pred_tokens) & set(ref_tokens)
        if not common_tokens:
            f1_scores.append(0.0)
            continue

        # Calculate precision, recall, and F1
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)

        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0


def evaluate_qa_on_test(model, test_dataloader, encodings, tokenizer, contexts, questions, answer_texts):
    """
    Evaluate the Question Answering model on the test set

    Args:
        model: BERT model for question answering
        test_dataloader: Test data loader
        encodings: Encoded examples
        tokenizer: BERT tokenizer
        contexts: List of context strings
        questions: List of question strings
        answer_texts: List of answer text strings

    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_start_logits = []
    all_end_logits = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k != 'offset_mapping'}

            # Forward pass
            outputs = model(**batch)

            # Get logits
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()

            all_start_logits.extend(start_logits)
            all_end_logits.extend(end_logits)

    # Get the most likely answer for each example
    example_to_features = encodings['test'].pop('overflow_to_sample_mapping', None)
    if example_to_features is None:
        # If overflow_to_sample_mapping is not available (e.g., for simple datasets),
        # assume each feature corresponds to one example
        example_to_features = list(range(len(all_start_logits)))

    offset_mapping = encodings['test'].pop('offset_mapping', None).cpu().numpy()

    # Initialize predictions
    predictions = [""] * len(contexts)  # Initialize with empty strings

    # Process each feature
    for i, (start_logits, end_logits) in enumerate(zip(all_start_logits, all_end_logits)):
        # Get the example index this feature comes from
        example_idx = example_to_features[i] if example_to_features is not None else i

        # Get the offsets mapping for this feature
        offsets = offset_mapping[i] if offset_mapping is not None else None

        # Find the tokens with the highest start and end scores
        start_idx = np.argmax(start_logits)
        end_idx = np.argmax(end_logits)

        # Skip if the answer points to the [CLS] token or if end is before start
        if start_idx == 0 or end_idx < start_idx:
            continue

        # Get the answer text from the context
        if offsets is not None:
            # Convert token indices to char indices using offset mapping
            answer_start = offsets[start_idx][0]
            answer_end = offsets[end_idx][1]

            # Extract the answer from the context
            predicted_answer = contexts[example_idx][answer_start:answer_end]
        else:
            # If offset mapping is not available, use token indices directly
            # This is less accurate but can be a fallback
            tokens = tokenizer.convert_ids_to_tokens(encodings['test']['input_ids'][i])
            predicted_answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1])

        # Update predictions for this example
        # If multiple features predict for the same example, keep the one with higher confidence
        current_score = np.max(start_logits) + np.max(end_logits)
        previous_score = 0  # Placeholder for comparison

        if predictions[example_idx]:
            # We already have a prediction for this example, compare scores
            # In a real implementation, you would store the scores with the predictions
            previous_score = -1  # Simplified; in reality, would store and compare actual scores

        if not predictions[example_idx] or current_score > previous_score:
            predictions[example_idx] = predicted_answer

    # Calculate evaluation metrics
    exact_match = evaluate_qa_exact_match(predictions, answer_texts)
    f1_score = evaluate_qa_f1(predictions, answer_texts)

    print("\nTest Results:")
    print(f"Exact Match: {exact_match:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # Show some examples with predictions
    print("\nExample predictions:")
    num_examples = min(5, len(contexts))
    for i in range(num_examples):
        print(f"\nExample {i + 1}:")
        print(f"Context: {contexts[i][:100]}...")
        print(f"Question: {questions[i]}")
        print(f"True Answer: {answer_texts[i]}")
        print(f"Predicted Answer: {predictions[i]}")

    return {
        'exact_match': exact_match,
        'f1_score': f1_score,
        'predictions': predictions
    }


def plot_qa_training_history(history):
    """
    Plot training and validation losses for QA

    Args:
        history (dict): Training history
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Question Answering Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('qa_training_history.png')
    plt.close()


def predict_qa(question, context, model, tokenizer):
    """
    Make Question Answering predictions for a single question and context

    Args:
        question (str): Question text
        context (str): Context text
        model: BERT model for question answering
        tokenizer: BERT tokenizer

    Returns:
        dict: Answer prediction including text, score, and positions
    """
    # Tokenize the input
    inputs = tokenizer(
        question,
        context,
        max_length=MAX_LEN,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    # Get the features and offset mapping
    features = {k: v.to(device) for k, v in inputs.items() if k != 'offset_mapping'}
    offset_mapping = inputs.pop('offset_mapping').cpu().numpy()

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**features)

    # Get the start and end logits
    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()

    # Get the most likely answer
    best_answer = ""
    best_score = -float('inf')

    for feature_idx, (start_logit, end_logit) in enumerate(zip(start_logits, end_logits)):
        # Find the tokens with the highest start and end scores
        start_idx = np.argmax(start_logit)
        end_idx = np.argmax(end_logit)

        # Skip if the answer points to the [CLS] token or if end is before start
        if start_idx == 0 or end_idx < start_idx:
            continue

        # Calculate score for this answer
        score = start_logit[start_idx] + end_logit[end_idx]

        # Convert token indices to char indices using offset mapping
        offsets = offset_mapping[feature_idx]
        answer_start = offsets[start_idx][0]
        answer_end = offsets[end_idx][1]

        # Extract the answer from the context
        answer = context[answer_start:answer_end]

        # Update best answer if this one has a higher score
        if score > best_score:
            best_score = score
            best_answer = answer
            best_positions = (answer_start, answer_end)

    print("\nQuestion Answering:")
    print(f"Question: {question}")
    print(f"Context: {context[:100]}...")
    print(f"Predicted Answer: {best_answer}")
    print(f"Confidence Score: {best_score:.4f}")

    return {
        'answer': best_answer,
        'score': best_score,
        'start': best_positions[0] if best_answer else -1,
        'end': best_positions[1] if best_answer else -1
    }


def save_qa_model_artifacts(model, tokenizer, model_path='./saved_qa_model'):
    """
    Save Question Answering model artifacts for later use

    Args:
        model: BERT model for question answering
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
        'max_len': MAX_LEN,
        'model_name': MODEL_NAME,
        'doc_stride': DOC_STRIDE
    }

    with open(os.path.join(model_path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print(f"Question Answering model artifacts saved to {model_path}")


def load_qa_model_artifacts(model_path='./saved_qa_model'):
    """
    Load Question Answering model artifacts for inference

    Args:
        model_path (str): Path to the saved model

    Returns:
        tuple: (model, tokenizer)
    """
    import pickle

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Load model
    model = BertForQuestionAnswering.from_pretrained(model_path).to(device)

    # Load configuration
    with open(os.path.join(model_path, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    # Update global variables if needed
    global MAX_LEN, DOC_STRIDE
    MAX_LEN = config['max_len']
    DOC_STRIDE = config['doc_stride']

    return model, tokenizer


def main():
    """Main function to run the script"""
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Example: Replace 'qa_data.json' with your actual data file
    contexts, questions, answer_texts, answer_starts = load_qa_data('qa_data.json')

    # Encode examples
    encodings = encode_qa_examples(contexts, questions, answer_texts, answer_starts, tokenizer)

    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_qa_dataloaders(encodings)

    # Load model
    model = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Train model
    model, history = train_qa_model(model, train_dataloader, val_dataloader, optimizer, scheduler)

    # Plot training history
    plot_qa_training_history(history)

    # Evaluate on test set
    evaluate_qa_on_test(model, test_dataloader, encodings, tokenizer, contexts, questions, answer_texts)

    # Save the trained model and artifacts
    save_qa_model_artifacts(model, tokenizer)

    # Example prediction
    sample_question = "Who is the CEO of OpenAI?"
    sample_context = "OpenAI was founded in December 2015 by Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever, John Schulman, and Wojciech Zaremba. In 2023, Sam Altman serves as the CEO of OpenAI."
    prediction = predict_qa(sample_question, sample_context, model, tokenizer)


if __name__ == "__main__":
    main()