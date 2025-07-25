print("Program Begins>>>>>>>>>>>>>>>>>>>")
print ("Importing OS Library\n")
import os
print ("Importing tokenizers Library\n")
from tokenizers import ByteLevelBPETokenizer
print ("Importing pathlib Library\n")
from pathlib import Path
print ("Importing transformers Library\n")
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, Trainer, pipeline, PreTrainedTokenizerFast
print ("Importing datasets Library\n")
from datasets import load_dataset
print ("Importing torch Library\n")
import torch
print("Program Library Imporging Completed >>>>>>>>>>>>>>>>>>>")

print("#####################################################################")
print("            STEP-6: Generate Text with your Mini GPT")
print("#####################################################################")
print("\n--- Testing text generation ---")
model_path = "./final_mini_gpt_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

prompt = "The quick brown fox"
generated_text = generator(
    prompt,
    max_length=50,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
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