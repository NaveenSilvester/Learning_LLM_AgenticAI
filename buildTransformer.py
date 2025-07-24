# The following libraries are required for running the code
# pip install transformers datasets accelerate tokenizers torch

#######################################################################################################
# Step 1: Prepare Your Dataset
#######################################################################################################
from datasets import load_dataset
# Load a small dataset for demonstration
# Wikitext-2 is a good choice for language modeling
# load_dataset: This is the core function from the datasets library. It's designed to easily load datasets from various sources:

"""
dataset =: This assigns the result of the load_dataset function call to a variable named dataset. This dataset variable will hold a Dataset object, which is a specialized data structure provided by the Hugging Face datasets library. This object is highly optimized for machine learning tasks, particularly for large datasets, as it often uses Apache Arrow internally for efficient memory management and I/O operations.
load_dataset(...): This is the core function from the datasets library. It's designed to easily load datasets from various sources:
Hugging Face Hub: The primary use case is to load datasets that are available on the Hugging Face Hub (huggingface.co/datasets), which hosts thousands of publicly available datasets for NLP, computer vision, audio, and more.
Local files: It can also load datasets from local files (like CSV, JSON, text files, Parquet, etc.) on your machine.
Custom loading scripts: For more complex or custom data formats, you can even provide your own Python script to define how the dataset should be loaded.
"wikitext": This is the first argument to load_dataset and specifies the name of the dataset you want to load. In this case, it's the "WikiText" dataset. WikiText is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia, commonly used for language modeling tasks.
"wikitext-2-raw-v1": This is the second argument, often referred to as the configuration name or subset name. Many datasets on the Hugging Face Hub have different "configurations" or "versions" that represent different pre-processing steps or subsets of the original data.
"""
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]") # Using 10% of the training set
#dataset = load_dataset("wikitext_train_10_percent.txt", split="train[:10%]") # Using 10% of the training set

print(f"Dataset size: {len(dataset)}")
print(dataset[0]['text']) # Example of text



#######################################################################################################
# Step 2: Train a Custom Tokenizer
#######################################################################################################
"""
A tokenizer converts text into numerical tokens that the model can understand. 
For GPT-like models, we typically use a Byte-Level Byte Pair Encoding (BPE) tokenizer, 
similar to GPT-2. Training a custom tokenizer on your specific data helps the model 
learn a more efficient representation of your text.
"""
import os
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, Trainer, pipeline
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import torch

# Save dataset text to a temporary file for tokenizer training
text_data_file = "temp_text_for_tokenizer.txt"
with open(text_data_file, "w", encoding="utf-8") as f:
    for text in dataset["text"]:
        if text.strip(): # Only write non-empty lines
            f.write(text + "\n")

# Initialize a ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()

# Define paths to your text data
paths = [text_data_file]

# Train the tokenizer
# vocab_size can be small for a mini-GPT, e.g., 5000-10000
# min_frequency: ignore tokens that appear less than this many times
tokenizer.train(
    files=paths,
    vocab_size=8000, # A small vocabulary size for "mini"
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]
)

# Save the tokenizer files
tokenizer_dir = "./mini_gpt_tokenizer"
# Create the directory if it doesn't exist
os.makedirs(tokenizer_dir, exist_ok=True)
# The exist_ok=True argument prevents an error if the directory already exists.
# Wrap your trained tokenizer in a PreTrainedTokenizerFast
# This step creates a tokenizer object that can be seamlessly saved
# and loaded by Hugging Face's AutoTokenizer.

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>" # If you use mask_token
)

# Now, save using the save_pretrained method of the PreTrainedTokenizerFast wrapper
hf_tokenizer.save_pretrained(tokenizer_dir)
tokenizer.save_model(tokenizer_dir)
print(f"Tokenizer (wrapped for HuggingFace) saved to {tokenizer_dir}")
# Save the tokenizer files
print(f"Tokenizer saved to {tokenizer_dir}")

# Load the trained tokenizer using HuggingFace's AutoTokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
tokenizer.pad_token = tokenizer.eos_token # For causal language modeling, usually pad with EOS token

# Test the tokenizer
example_text = "Hello, this is a test for our mini GPT tokenizer."
encoded_input = tokenizer(example_text, return_tensors="pt")
print(f"Original text: {example_text}")
print(f"Encoded input IDs: {encoded_input.input_ids}")
print(f"Decoded tokens: {tokenizer.decode(encoded_input.input_ids[0])}")



#######################################################################################################
# Step 3: Configure Your Mini GPT Model
#######################################################################################################
"""
We'll define the architecture for our mini GPT. We'll use GPT2Config as a base, as it's a standard 
causal language model configuration. We'll significantly reduce the parameters to make it "mini".

These are the core parameters to shrink for a "mini" model.
vocab_size: This must match the vocabulary size of your tokenizer.
n_positions: The maximum sequence length the model can handle.
n_embd: The dimensionality of the token embeddings and the hidden states of the transformer.
n_layer: The number of transformer blocks (layers).
n_head: The number of attention heads in each multi-head attention mechanism.
"""
from transformers import GPT2Config, GPT2LMHeadModel
import torch

print ("STEP-3 STARTING NOW #################################################")
# Define a mini GPT-like configuration
# These parameters are much smaller than a full GPT-2
config = GPT2Config(
    vocab_size=len(tokenizer),       # Vocabulary size from our tokenizer
    n_positions=512,                 # Max sequence length
    n_embd=128,                      # Embedding dimension
    n_layer=4,                       # Number of transformer layers
    n_head=4,                        # Number of attention heads
    resid_pdrop=0.1,                 # Dropout probability for residual connections
    embd_pdrop=0.1,                  # Dropout probability for embeddings
    attn_pdrop=0.1,                  # Dropout probability for attention
)

# Initialize the model from the configuration
# GPT2LMHeadModel is a GPT-2 model with a language modeling head on top
model = GPT2LMHeadModel(config)

print(f"Number of parameters in the mini GPT model: {model.num_parameters()}")
print ("Step-3 is completed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


#######################################################################################################
# Step 4: Preprocess the Dataset for Training
#######################################################################################################
"""
For causal language modeling, the task is to predict the next token in a sequence. This means the labels
 for a given input sequence [t1, t2, t3] will be [t2, t3, <EOS_TOKEN>] (or simply the input shifted by 
 one, with padding for the last token). Hugging Face's Trainer handles this automatically when using 
 GPT2LMHeadModel and provides labels as input_ids.
We'll need to tokenize the dataset and then group the texts into blocks of a fixed size.
"""
print ("STEP-4 STARTING NOW #################################################")
from transformers import DataCollatorForLanguageModeling
import torch

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4, # Use multiple processes for faster tokenization if available
    remove_columns=["text"] # Remove the original text column
)

# Group texts into blocks of n_positions (e.g., 512)
block_size = config.n_positions

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this line and the next.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=4, # Use multiple processes for faster processing
)

print(f"Number of tokenized and grouped samples: {len(lm_datasets)}")
print(f"First sample input_ids length: {len(lm_datasets[0]['input_ids'])}")

# Create a data collator. This will handle padding dynamically during training.
# For causal language modeling, we typically use MLM=False.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

print ("Step-4 is completed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

#######################################################################################################
# Step 5: Set up Training Arguments and Trainer
#######################################################################################################
"""
The Trainer class from Hugging Face simplifies the training loop significantly.
"""
from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./mini_gpt_model",       # Directory to save checkpoints
    overwrite_output_dir=True,
    num_train_epochs=5,                  # Number of training epochs (adjust as needed)
    per_device_train_batch_size=8,       # Batch size per GPU/CPU
    save_steps=500,                      # Save model checkpoint every 500 steps
    save_total_limit=2,                  # Only keep the last 2 checkpoints
    logging_dir="./logs",                # Directory for logs
    logging_steps=100,                   # Log every 100 steps
    learning_rate=2e-4,                  # Learning rate
    weight_decay=0.01,                   # Weight decay for regularization
    gradient_accumulation_steps=2,       # Accumulate gradients over 2 steps (to effectively increase batch size)
    fp16=torch.cuda.is_available(),      # Use mixed precision training if CUDA is available
    report_to="none"                     # Disable reporting to W&B, MLflow etc. for simplicity
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    data_collator=data_collator,
)

print("Starting training...")
# Start training
trainer.train()

print("Training complete!")

# Save the final model and tokenizer
trainer.save_model("./final_mini_gpt_model")
tokenizer.save_pretrained("./final_mini_gpt_model")
print("Model and tokenizer saved to ./final_mini_gpt_model")

#######################################################################################################
# Step 6: Generate Text with Your Mini GPT
#######################################################################################################
"""
After training, you can use your model to generate text.
"""
from transformers import pipeline

# Load the trained model and tokenizer
model_path = "./final_mini_gpt_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Create a text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1 # Use GPU if available
)

# Generate text
prompt = "The quick brown fox"
generated_text = generator(
    prompt,
    max_length=50,
    num_return_sequences=1,
    do_sample=True,      # Enable sampling for more diverse output
    top_k=50,            # Sample from top 50 likely tokens
    temperature=0.7,     # Control randomness
    pad_token_id=tokenizer.eos_token_id # Important for generation
)

print(f"\n--- Generated Text ---")
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text[0]['generated_text']}")

prompt = "In the heart of the ancient forest,"
generated_text = generator(
    prompt,
    max_length=70,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id
)
print(f"\n--- Generated Text ---")
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text[0]['generated_text']}")