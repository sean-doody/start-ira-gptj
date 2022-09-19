import os
import boto3
import json
import logging

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
    MODEL_NAME = os.environ.get("MODEL_NAME")
    os.mkdir(f"models/{MODEL_NAME}_debug/{MODEL_NAME}")

    boto3_session = boto3.Session(aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)
    s3 = boto3_session.client("s3")
    s3.download_file(BUCKET, AWS_FILE, LOCAL_FILE)
    
    with open(LOCAL_FILE, "r") as f:
        data = json.load(f)["texts"][:500]
    
    # load model:
    logging.info("Downloading tokenizer & model")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",
                                              bos_token="<|startoftext|>", 
                                              eos_token="<|endoftext|>", 
                                              pad_token="<|pad|>")

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    
    model.resize_token_embeddings(len(tokenizer))
    
    # prepare dataset:
    class GPTDataset(Dataset):

        def __init__(self, encodings):

            self.encodings = encodings
            self.labels = None
            
        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key,val in self.encodings.items()}
            return item
    
    
    validation_set = False

    logging.info("Preparing datasets")
    if validation_set:
        train, test = train_test_split(data, test_size=.20, random_state=42)
        train_tokens = tokenizer(train, padding=True, truncation=True, max_length=512)
        test_tokens = tokenizer(test, padding=True, truncation=True, max_length=512)
        train_data = GPTDataset(train_tokens)
        test_data = GPTDataset(test_tokens)
    else:
        train_tokens = tokenizer(data, padding=True, truncation=True, max_length=512)
        train_data = GPTDataset(train_tokens)
        
    # prepare trainer:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)    
    
    training_args = TrainingArguments(
        output_dir=f"models/{MODEL_NAME}_debug/{MODEL_NAME}",
        do_train=True,
        num_train_epochs=1,
        logging_steps=25,
        save_strategy="epoch",
        per_device_train_batch_size=15,
        warmup_steps=50,
        weight_decay=0.01,
        fp16=True,
        deepspeed="gpt-j-deepspeed.json"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data
    )
    
    # train model:
    logging.info("Training")
    trainer.train()
    
    # save the model:
    logging.info("Saving Model")
    trainer.save_model(f"models/{MODEL_NAME}_debug/{MODEL_NAME}")
    
    # port to AWS:
    for root,dirs,files in os.walk("models"):
        for file in files:
            s3.upload_file(os.path.join(root, file), BUCKET, root + file)
        
    

if __name__ == "__main__":
    main()