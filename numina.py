### Code for washing the numina dataset

import random
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from datasets import load_from_disk, concatenate_datasets
from sft_utils import extract_all_boxed_answers, extract_boxed_content

# -------------------------------
# Configuration
# -------------------------------
MODEL_NAME = "./model/Llama-3.2-1B"         # Update with your actual model identifier/path.
DATASET_PATH = "./data/NuminaMath-CoT"   # Directory where the original dataset is stored.
PROCESSED_TRAIN_PATH = "./data/NuminaMath-CoT/train"  # Directory to save processed training data.
PROCESSED_EVAL_PATH = "./data/NuminaMath-CoT/test"    # Directory to save processed validation data.
MAX_TOKENS = 2048                           # Maximum tokens per example.
MAX_TRAIN_EXAMPLES = 100000                 # Maximum training examples.
MAX_EVAL_EXAMPLES = 100                     # Maximum evaluation examples.

# -------------------------------
# Data Loading Function
# -------------------------------
def load_datasets(dataset_name_or_path):
    """
    Loads the dataset from disk using the Hugging Face datasets library.
    Shards the training dataset into 10 parts and returns the shard at index 0.
    Expects that the dataset on disk has splits "train" and "test".
    """
    raw_datasets = load_from_disk(dataset_name_or_path)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    return train_dataset, eval_dataset

# -------------------------------
# Filtering: Remove Proving Questions
# -------------------------------
def filter_proving_questions(dataset):
    """
    Filters out any examples where the "problem" (or associated text fields)
    contain keywords that indicate a proving question.
    The check is performed in a case-insensitive manner.
    """
    keywords = ["prove", "proving"]

    def no_proving(example):
        # Check within the 'problem' field.
        text = example["problem"].lower()
        # Optionally, you can also check within the answer if needed.
        answer_text = ""
        messages = example["messages"]
        if messages:
            answer_text = messages[-1]["content"].lower()
        combined = text + " " + answer_text
        return not any(kw in combined for kw in keywords)

    return dataset.filter(no_proving)

# -------------------------------
# Filtering: Remove Examples With Long Token Counts
# -------------------------------
def filter_by_length(example, tokenizer, max_tokens=MAX_TOKENS):
    """
    Constructs the example text by concatenating:
        problem = example['problem']
        answer  = example['messages'][-1]['content']
    Then tokenizes the combined text using the provided tokenizer.
    Returns True if the token count is strictly less than max_tokens.
    Otherwise, the example is filtered out.
    """
    # Extract the problem.
    problem = example.get("problem", "")
    # Extract the answer from the last message.
    messages = example["messages"]
    answer = messages[-1]["content"]
    combined_text = problem + "\n" + answer
    tokens = tokenizer.encode(combined_text)
    return len(tokens) < max_tokens

def filter_multiple_answers(dataset):
    """
    Filters out examples that have multiple answers.
    This is done by checking the number of messages in the example.
    If there are more than 2 messages, we consider it to have multiple answers.
    """
    def no_multiple_answers(example):
        answer = example["messages"][-1]["content"]
        len_answers = len(extract_boxed_content(answer))
        return len_answers == 1
    return dataset.filter(no_multiple_answers)


# -------------------------------
# Main Processing Pipeline
# -------------------------------
def main():
    # 1. Load the original datasets.
    train_dataset, eval_dataset = load_datasets(DATASET_PATH)
    print(f"Loaded training dataset with {len(train_dataset)} examples.")
    print(f"Loaded evaluation dataset with {len(eval_dataset)} examples.")
    
    # 2. Filter out proving questions from both splits.
    train_dataset = filter_proving_questions(train_dataset)
    eval_dataset = filter_proving_questions(eval_dataset)
    print(f"After filtering proving questions, training dataset has {len(train_dataset)} examples.")
    print(f"After filtering proving questions, evaluation dataset has {len(eval_dataset)} examples.")
    
    # FIlter multiple-answer examples
    train_dataset = filter_multiple_answers(train_dataset)
    eval_dataset = filter_multiple_answers(eval_dataset)
    print(f"After filtering multiple answers, training dataset has {len(train_dataset)} examples.")
    print(f"After filtering multiple answers, evaluation dataset has {len(eval_dataset)} examples.")

    # 3. Initialize the tokenizer for your base model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    # 4. Filter out examples that are too long (>= 2048 tokens).
    train_dataset = train_dataset.filter(
        filter_by_length,
        fn_kwargs={"tokenizer": tokenizer, "max_tokens": MAX_TOKENS}
    )
    eval_dataset = eval_dataset.filter(
        filter_by_length,
        fn_kwargs={"tokenizer": tokenizer, "max_tokens": MAX_TOKENS}
    )
    
    print(f"After length filtering, training dataset has {len(train_dataset)} examples.")
    print(f"After length filtering, evaluation dataset has {len(eval_dataset)} examples.")
    
    # If the length of eval_dataset is less than 100, use a few examples from train_dataset
    if len(eval_dataset) < 100:
        print("Evaluation dataset is too small, using some examples from training dataset.")
        num_additional_examples = 100 - len(eval_dataset)
        
        # Randomly sample indices from train dataset
        additional_ids = random.sample(range(len(train_dataset)), num_additional_examples)
        
        # Select the examples
        additional_examples = train_dataset.select(additional_ids)
        
        # Concatenate using Hugging Face utility
        eval_dataset = concatenate_datasets([eval_dataset, additional_examples])
        
        print(f"After adding examples, evaluation dataset has {len(eval_dataset)} examples.")

    # 5. Limit the number of examples to the specified maximum (shuffling for randomness).
    train_dataset = train_dataset.shuffle(seed=42).select(range(min(MAX_TRAIN_EXAMPLES, len(train_dataset))))
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(min(MAX_EVAL_EXAMPLES, len(eval_dataset))))
    print(f"Final training examples: {len(train_dataset)}")
    print(f"Final evaluation examples: {len(eval_dataset)}")
    
    # 6. Save the processed datasets in native Hugging Face format
    train_dataset.save_to_disk(PROCESSED_TRAIN_PATH, num_shards=5)
    eval_dataset.save_to_disk(PROCESSED_EVAL_PATH)
    print("Processed datasets have been saved to disk.")

if __name__ == "__main__":
    main()