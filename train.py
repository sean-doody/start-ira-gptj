import os
import boto3
import json
import logging
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split


def main():
    # load data:
    logging.info("Loading data")
    AWS_KEY = os.environ.get("AWS_KEY")
    AWS_SECRET = os.environ.get("AWS_SECRET")
    BUCKET = os.environ.get("AWS_BUCKET")
    AWS_FILE = "data/processed/training_prompts.json"
    LOCAL_FILE = "training_prompts.json"
    MODEL_NAME = "gpt-neo-2.7b-eval"

    boto3_session = boto3.Session(aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)
    s3 = boto3_session.client("s3")
    s3.download_file(BUCKET, AWS_FILE, LOCAL_FILE)
    
    with open(LOCAL_FILE, "r") as f:
        data = json.load(f)["texts"][:500]
    
    # load model:
    logging.info("Downloading tokenizer & model")
    hf_model = "gpt-neo-2.7b"
    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{hf_model}",
                                              bos_token="<|startoftext|>", 
                                              eos_token="<|endoftext|>", 
                                              pad_token="<|pad|>")

    model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{hf_model}")
    
    model.resize_token_embeddings(len(tokenizer))
    
    # pytorch dataset:
    class GPTDataset(Dataset):
        def __init__(self, tweets, tokenizer, max_length):
            self.input_ids = []
            self.attn_masks = []
            self.labels = []
            for tweet in tqdm(tweets):
                encodings_dict = tokenizer(tweet, truncation=True, max_length=max_length, padding="max_length")
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx]
    
    
    logging.info("Preparing datasets")
    logging.info("Estimating max length")
    max_len = max([len(tokenizer.encode(tweet)) for tweet in tqdm(data)])
    
    logging.info("Creating train and test sets")
    train, test = train_test_split(data, test_size=.10, random_state=42)
    train_tokens = GPTDataset(train, tokenizer, max_len)
    test_tokens = GPTDataset(test, tokenizer, max_len)
    
    # hyperparameters:
    LEARNING_RATE = 1.372e-4
    N_EPOCHS = 3
    
    training_args = TrainingArguments(
        output_dir=MODEL_NAME,
        num_train_epochs=N_EPOCHS,
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=int(0.10*len(train_tokens)),
        weight_decay=0.01,
        learning_rate=LEARNING_RATE
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=lambda data: {"input_ids": torch.stack([f[0] for f in data]),
                                    "attention_mask": torch.stack([f[1] for f in data]),
                                    "labels": torch.stack([f[0] for f in data])},
        train_dataset=train_tokens,
        eval_dataset=test_tokens
    )
    
    # train model:
    logging.info("Training")
    trainer.train()
    
    # save the model:
    logging.info("Saving Model")
    trainer.save_model()
    
    logging.info("Uploading to AWS")
    for root,dirs,files in os.walk(MODEL_NAME):
        for file in tqdm(files):
            s3.upload_file(os.path.join(root, file), BUCKET, os.path.join("models", root, file))


if __name__ == "__main__":
    main()