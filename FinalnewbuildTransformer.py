import os
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, Trainer, pipeline, PreTrainedTokenizerFast
from datasets import load_dataset
import torch

# --- All your imports should be outside the if __name__ == '__main__': block ---

# ==============================================================================
# All your main execution logic goes inside this block
# ==============================================================================
if __name__ == '__main__':
    # ----------------------------------------------------------------------
    # Step 1: Prepare Your Dataset
    # ----------------------------------------------------------------------
    # Create a dummy text file for tokenizer training if it doesn't exist
    print("#####################################################################")
    print("               STEP-1: To Prepare Your Dataset")
    print("#####################################################################")

    text_data_file = "temp_text_for_tokenizer.txt"
    if not os.path.exists(text_data_file):
        print(f"Creating dummy dataset file: {text_data_file}")
        with open(text_data_file, "w", encoding="utf-8") as f:
            for i in range(100): # Write enough lines for a decent tokenizer
                f.write(f"This is a sample line {i} of text for our mini GPT tokenizer training. It should contain diverse vocabulary.\n")
                f.write(f"Another sentence to enrich the training data for the tokenizer. The quick brown fox jumps over the lazy dog.\n")
            f.write("And this is the final sentence for the tokenizer.\n")
    else:
        print(f"Using existing dataset file: {text_data_file}")


    # Load a small dataset for demonstration
    # Wikitext-2 is a good choice for language modeling
    # Using 'text' builder for local file
    print("Loading dataset for model training...")
    dataset = load_dataset("text", data_files={"train": text_data_file}, split="train[:100%]")
    # Or if you still want wikitext-2, ensure it's loaded within this block
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")

    print(f"Dataset size: {len(dataset)}")
    print(dataset[0]['text'])


    # ----------------------------------------------------------------------
    # Step 2: Train a Custom Tokenizer
    # ----------------------------------------------------------------------
    print("#####################################################################")
    print("               STEP-2: Train a Custom Tokenizer")
    print("#####################################################################")
    tokenizer_dir = "./mini_gpt_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)

    # Check if tokenizer is already trained and saved
    if not (os.path.exists(os.path.join(tokenizer_dir, 'vocab.json')) and \
            os.path.exists(os.path.join(tokenizer_dir, 'merges.txt'))):
        print("Training new tokenizer...")
        tokenizer_obj = ByteLevelBPETokenizer()
        tokenizer_obj.train(
            files=[text_data_file],
            vocab_size=8000, # A small vocabulary size for "mini"
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )

        # Wrap and save the tokenizer for Hugging Face compatibility
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>"
        )
        hf_tokenizer.save_pretrained(tokenizer_dir)
        print(f"Tokenizer (wrapped for HuggingFace) saved to {tokenizer_dir}")
    else:
        print(f"Tokenizer already found at {tokenizer_dir}, loading existing.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token # For causal language modeling, usually pad with EOS token
    print("Tokenizer loaded successfully with AutoTokenizer!")

    # Test the tokenizer
    example_text = "Hello, this is a test for our mini GPT tokenizer."
    encoded_input = tokenizer(example_text, return_tensors="pt")
    print(f"Original text: {example_text}")
    print(f"Encoded input IDs: {encoded_input.input_ids}")
    print(f"Decoded tokens: {tokenizer.decode(encoded_input.input_ids[0])}")


    # ----------------------------------------------------------------------
    # Step 3: Configure Your Mini GPT Model
    # ----------------------------------------------------------------------
    print("#####################################################################")
    print("               STEP-3: Configuring mini GPT model..")
    print("#####################################################################")    
    print("Configuring mini GPT model...")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=512,
        n_embd=128,
        n_layer=4,
        n_head=4,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config)
    print(f"Number of parameters in the mini GPT model: {model.num_parameters()}")


    # ----------------------------------------------------------------------
    # Step 4: Preprocess the Dataset for Training
    # ----------------------------------------------------------------------
    print("#####################################################################")
    print("           STEP-4: Preprocessing dataset for training...")
    print("#####################################################################")    
    print("Preprocessing dataset for training...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    # Note: num_proc > 1 often causes the multiprocessing issue if not in __name__ == '__main__'
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1, # Setting to 1 to avoid multiprocessing issues if the fix above doesn't fully resolve, or if you're still debugging. Can increase later.
        remove_columns=["text"]
    )

    block_size = config.n_positions

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1, # Same as above
    )

    print(f"Number of tokenized and grouped samples: {len(lm_datasets)}")
    print(f"First sample input_ids length: {len(lm_datasets[0]['input_ids'])}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )


    # ----------------------------------------------------------------------
    # Step 5: Set up Training Arguments and Trainer
    # ----------------------------------------------------------------------
    print("#####################################################################")
    print("          STEP-5: Set up Training Arguments and Trainer")
    print("#####################################################################")
    training_args = TrainingArguments(
        output_dir="./mini_gpt_model",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=2e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    trainer.save_model("./final_mini_gpt_model")
    tokenizer.save_pretrained("./final_mini_gpt_model")
    print("Model and tokenizer saved to ./final_mini_gpt_model")


    # ----------------------------------------------------------------------
    # Step 6: Generate Text with Your Mini GPT
    # ----------------------------------------------------------------------
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