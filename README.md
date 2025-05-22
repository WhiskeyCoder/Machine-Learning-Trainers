# BERT-based Machine Learning Models Suite

This repository contains a suite of well-documented Python scripts for building different types of machine learning models using BERT for various natural language processing tasks. Each script is extensively commented to help you understand the process and easily adapt it for your specific use cases.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Models Included](#models-included)
4. [When to Use Each Model](#when-to-use-each-model)
5. [Usage Instructions](#usage-instructions)
6. [Adapting for Your Data](#adapting-for-your-data)
7. [GPU Considerations](#gpu-considerations)

## Overview

This suite provides a collection of ready-to-use scripts for training BERT-based models on various NLP tasks. Each script follows a similar structure to make it easy to understand and modify:

- Data loading and preprocessing
- Dataset and dataloader creation
- Model definition and initialization
- Training and evaluation functions
- Utility functions for prediction and model saving/loading

## Requirements

```
torch>=1.8.0
transformers>=4.5.0
sklearn>=0.24.0
numpy>=1.19.0
pandas>=1.2.0
tqdm>=4.60.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

For specific models:
- Question Answering: No additional requirements
- NER: `seqeval` for entity-based evaluation
- Summarization: `rouge-score` for evaluation

## Models Included

1. **Binary Classification** (`binary_classification_bert.py`)
   - For tasks with two output classes (yes/no, positive/negative)

2. **Multiclass Classification** (`multiclass_classification_bert.py`)
   - For tasks with multiple output classes (topic classification, intent detection)

3. **Multi-Task Learning** (`multi-task_learning_bert.py`)
   - For training a single model on multiple classification tasks simultaneously

4. **Named Entity Recognition** (`named_entity_recognition_bert.py`)
   - For identifying and classifying entities in text (people, organizations, locations)

5. **Question Answering** (`question_anaswering_bert.py`)
   - For extracting answers to questions from a given context

6. **Text Summarization** (`abstractive_text_summarization_bert.py`)
   - For generating concise summaries of longer texts

7. **Similarity Learning** (`siamese_network_similiarty_learning_bert.py`)
   - For measuring semantic similarity between text pairs

## When to Use Each Model

### Binary Classification
- **Use Cases**: Sentiment analysis, spam detection, intent classification
- **Input**: Single text
- **Output**: Binary label (0 or 1)
- **When to Use**: When your task involves categorizing text into one of two classes

### Multiclass Classification
- **Use Cases**: Topic classification, intent classification with multiple categories, product categorization
- **Input**: Single text
- **Output**: One of several class labels
- **When to Use**: When your task involves categorizing text into one of several (3+) classes

### Multi-Task Learning
- **Use Cases**: Joint intent and slot detection, multi-aspect sentiment analysis
- **Input**: Single text
- **Output**: Multiple labels for different aspects of the text
- **When to Use**: When you want to train a single model to perform multiple related classification tasks

### Named Entity Recognition (NER)
- **Use Cases**: Extracting entities like people, organizations, locations from text
- **Input**: Text sequence
- **Output**: Token-level entity tags (B-PER, I-ORG, etc.)
- **When to Use**: When you need to identify specific entities within text

### Question Answering
- **Use Cases**: Building a FAQ system, information extraction
- **Input**: Question and context passage
- **Output**: Answer span from the context
- **When to Use**: When you need to extract specific information from longer texts

### Text Summarization
- **Use Cases**: Article summarization, report generation
- **Input**: Long text
- **Output**: Concise summary
- **When to Use**: When you need to create shorter versions of longer documents

### Similarity Learning
- **Use Cases**: Duplicate detection, semantic search, document similarity
- **Input**: Pair of texts
- **Output**: Similarity score (0-1)
- **When to Use**: When you need to measure how similar two texts are

## Usage Instructions

Each script follows a similar pattern for usage:

1. **Data Preparation**: Prepare your data in CSV or JSON format (specific format depends on the task)
2. **Configuration**: Adjust the global configuration variables at the top of each script
3. **Training**: Run the script to train the model
4. **Evaluation**: Automatic evaluation on validation and test sets is performed
5. **Inference**: Use the provided prediction functions for inference on new data

### Example Usage (Binary Classification)

```python
# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Load data
texts, labels = load_data('your_data.csv')

# Prepare dataloaders
train_dataloader, val_dataloader = prepare_dataloaders(texts, labels, tokenizer)

# Initialize model
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(device)

# Prepare optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Train and evaluate
model = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler)

# Make prediction on new data
text = "This is a sample text for prediction"
prediction, probability = predict(text, model, tokenizer)
```

## Adapting for Your Data

### Data Format
The scripts accept different input formats depending on the task:

1. **Classification Tasks**: CSV with columns 'text' and 'label', or JSON with 'text' and 'label' keys
2. **NER**: Either JSON format with texts and entity spans, or CoNLL format with token/tag pairs
3. **Question Answering**: SQuAD-like format with contexts, questions, and answer spans
4. **Summarization**: CSV or JSON with 'text' and 'summary' pairs
5. **Similarity Learning**: CSV or JSON with 'text1', 'text2', and 'is_similar' fields

### Adjusting Model Parameters

At the top of each script, you'll find global configuration variables you can modify:

```python
# Global configuration variables
MAX_LEN = 128           # Maximum sequence length
BATCH_SIZE = 16         # Batch size for training
EPOCHS = 3              # Number of training epochs
LEARNING_RATE = 2e-5    # Learning rate
MODEL_NAME = "bert-base-uncased"  # Base model
```

For more complex changes:

1. **Model Architecture**: Modify the model definition classes
2. **Training Loop**: Modify the training functions
3. **Evaluation Metrics**: Add or modify metrics in evaluation functions

## GPU Considerations

### Memory Management
BERT models can be memory-intensive. Here are tips for managing GPU memory:

1. **Reduce Batch Size**: Lower BATCH_SIZE if you encounter OOM errors
2. **Reduce Sequence Length**: Lower MAX_LEN if your texts permit
3. **Gradient Accumulation**: Add gradient accumulation to simulate larger batches
4. **Model Pruning**: Use smaller BERT variants (BERT-small, DistilBERT)
5. **Freeze Layers**: Freeze certain layers of BERT to reduce memory during training

### Mixed Precision Training
To enable mixed precision training for faster computation:

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

# In training loop
with autocast():
    # Forward pass with mixed precision
    outputs = model(...)
    loss = outputs.loss

# Scale loss and backward
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Advanced Usage Examples

### Fine-tuning Strategies

1. **Layer Freezing**: Freeze BERT layers and only train the classification head

```python
# Freeze all BERT layers
for param in model.bert.parameters():
    param.requires_grad = False
    
# Only train classification layer parameters
```

2. **Gradual Unfreezing**: Start with frozen layers and gradually unfreeze

```python
# Initially freeze all layers
for param in model.bert.parameters():
    param.requires_grad = False
    
# In later epochs, unfreeze more layers
def unfreeze_layers(epoch):
    if epoch == 1:
        # Unfreeze last 2 layers
        for layer in model.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
```

3. **Different Learning Rates**: Use lower learning rates for BERT layers

```python
from torch.optim import AdamW

# Group parameters with different learning rates
bert_params = list(model.bert.parameters())
classifier_params = list(model.classifier.parameters())

optimizer = AdamW([
    {'params': bert_params, 'lr': 1e-5},
    {'params': classifier_params, 'lr': 3e-5}
])
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Solution: Reduce batch size, sequence length, or use gradient accumulation

2. **Slow Training**
   - Solution: Use mixed precision training, smaller model variants, or more GPUs

3. **Poor Performance**
   - Solution: Increase model size, adjust learning rate, increase training data, or use data augmentation

4. **Overfitting**
   - Solution: Add dropout, weight decay, use early stopping, or collect more training data

## Citation and References

If you use these models in your research, please cite the relevant papers:

```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
