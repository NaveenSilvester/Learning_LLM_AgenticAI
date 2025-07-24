from datasets import load_dataset
import os

# 1. Load the dataset
# This step automatically downloads and caches the dataset.
#dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")

# Define the local file path to save the text
output_text_file = "wikitext_train_10_percent.txt"

# 2. Iterate through the dataset and write the text content to the file
print(f"Saving dataset content to {output_text_file}...")
with open(output_text_file, "w", encoding="utf-8") as f:
    for entry in dataset:
        text = entry['text'].strip() # Get the 'text' field and remove leading/trailing whitespace
        if text: # Only write non-empty lines
            f.write(text + "\n")

print(f"Dataset content successfully saved to {output_text_file}")

# Verify by checking file size or reading a few lines
print(f"File size: {os.path.getsize(output_text_file) / (1024*1024):.2f} MB")