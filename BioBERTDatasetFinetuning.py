import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import DatasetDict, Dataset
from seqeval.metrics import classification_report
import numpy as np
import json

# Dataset name variable
DATASET_NAME = "DatasetFolder/DatasetName"

# Define dataset paths based on the dataset name
train_dataset_path = f"./Datasets/{DATASET_NAME}/train.txt"
valid_dataset_path = f"./Datasets/{DATASET_NAME}/valid.txt"
test_dataset_path = f"./Datasets/{DATASET_NAME}/test.txt"

# Load and preprocess dataset
def load_custom_dataset(file_path):
    tokens, labels = [], []
    current_tokens, current_labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Non-empty line
                parts = line.split()
                if len(parts) >= 2:  # Expecting at least 2 columns: token and label
                    token = parts[0]  # First column is the token
                    label = parts[-1]  # Last column is the label
                    current_tokens.append(token)
                    current_labels.append(label)
            else:  # Empty line indicates end of a sentence
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens, current_labels = [], []
    # In case there's no trailing newline
    if current_tokens:
        tokens.append(current_tokens)
        labels.append(current_labels)
    return tokens, labels

def convert_to_hf_format(tokens, labels, label2id):
    """Convert token and label lists into HuggingFace Dataset format."""
    hf_data = []
    for token_list, label_list in zip(tokens, labels):
        label_ids = [label2id[label] for label in label_list]
        hf_data.append({"tokens": token_list, "labels": label_ids})
    return hf_data

# Load your train, valid, and test datasets
tokens_train, labels_train = load_custom_dataset(train_dataset_path)
tokens_valid, labels_valid = load_custom_dataset(valid_dataset_path)
tokens_test, labels_test = load_custom_dataset(test_dataset_path)

# Create label mappings
unique_labels = set([label for label_list in labels_train + labels_valid + labels_test for label in label_list])
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)

# Convert datasets to HuggingFace format
hf_data_train = convert_to_hf_format(tokens_train, labels_train, label2id)
hf_data_valid = convert_to_hf_format(tokens_valid, labels_valid, label2id)
hf_data_test = convert_to_hf_format(tokens_test, labels_test, label2id)

# Create DatasetDict for train, validation, and test
train_dataset = Dataset.from_list(hf_data_train)
valid_dataset = Dataset.from_list(hf_data_valid)
test_dataset = Dataset.from_list(hf_data_test)

# Combine into DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset,
    "test": test_dataset
})

# Load BioBERT tokenizer and model
# Use AutoTokenizer and AutoModelForTokenClassification for flexibility
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=num_labels)

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding='max_length', max_length=128)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply the tokenization using multiple CPU cores
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, num_proc=12, remove_columns=["tokens", "labels"])  # Use 12 cores for parallel processing

# Define output directories based on dataset name
finetuned_model_dir = f"./BioBERTFinetuned_{DATASET_NAME}"
evaluation_results_dir = f"./EvaluationResults_{DATASET_NAME}"
if not os.path.exists(evaluation_results_dir):
    os.makedirs(evaluation_results_dir)

# Training arguments
training_args = TrainingArguments(
    output_dir=finetuned_model_dir,  # output directory
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    warmup_steps=500,
    remove_unused_columns=False,  # Keep all columns for compatibility
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(finetuned_model_dir)
print(f"Fine-tuned BioBERT model has been saved to {finetuned_model_dir}")

# Save additional model components
# Save label list to JSON
label_list_file = os.path.join(finetuned_model_dir, "label_list.json")
with open(label_list_file, "w") as f:
    json.dump(list(label2id.keys()), f)
print(f"Label list has been saved to {label_list_file}")

# Save tokenizer
tokenizer.save_pretrained(finetuned_model_dir)
print(f"Tokenizer has been saved to {finetuned_model_dir}")

# Save training arguments
training_args_file = os.path.join(finetuned_model_dir, "training_args.bin")
torch.save(training_args, training_args_file)
print(f"Training arguments have been saved to {training_args_file}")

# Evaluate the model on the test dataset
test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print("Test results:", test_results)

# Save the evaluation results to a file
evaluation_file_path = os.path.join(evaluation_results_dir, "test_results.txt")
with open(evaluation_file_path, "w") as eval_file:
    for key, value in test_results.items():
        eval_file.write(f"{key}: {value}\n")
print(f"Test evaluation results have been saved to {evaluation_file_path}")

# Calculate F1, Precision, and Recall using seqeval
def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id2label[label] for label in label_set if label != -100] for label_set in labels]
    true_predictions = [[id2label[pred] for (pred, label) in zip(pred_set, label_set) if label != -100] for pred_set, label_set in zip(predictions, labels)]

    report = classification_report(true_labels, true_predictions, output_dict=True)
    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
    }

# Update trainer with the compute_metrics function for calculating F1, precision, and recall
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# Evaluate again to get detailed metrics
test_results_with_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print("Test results with metrics:", test_results_with_metrics)

# Save the evaluation results with metrics to a file
evaluation_metrics_file_path = os.path.join(evaluation_results_dir, "test_results_with_metrics.txt")
with open(evaluation_metrics_file_path, "w") as eval_file:
    for key, value in test_results_with_metrics.items():
        eval_file.write(f"{key}: {value}\n")
print(f"Test evaluation results with metrics have been saved to {evaluation_metrics_file_path}")
