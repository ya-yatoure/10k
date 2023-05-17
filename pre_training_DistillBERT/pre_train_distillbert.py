import os
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import yaml
from datasets import Dataset
import torch
import time

print(f"GPU: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

data_path = "../data/"
hand_path = "./"

output_path = "10k/pre_training_DistillBERT/"
checkpoint_path = "10k/pre_training_DistillBERT/checkpoints/"
final_model_path = "10k/pre_training_DistillBERT/final/"

# check if the output directories exist and, if not, create them
for dir_path in [output_path, checkpoint_path, final_model_path]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory {dir_path} created successfully")
    else:
        print(f"Directory {dir_path} already exists")

# open YAML file with parameters for pretraining
stream = open("10k/pre_training_DistillBERT/distilbert_params.yaml", 'r')
params = yaml.load(stream, Loader=yaml.Loader)

# Load the data
df = pd.read_csv("10k/pre_training_DistillBERT/2019_10kdata_with_covars_sample.csv")
df = df[['text']]

# Transform into Dataset class
raw_datasets = Dataset.from_pandas(df)

# Tokenize text
model_name = params["model_name"]
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForMaskedLM.from_pretrained(model_name)

# Tokenization
max_sent_size = params["max_sent_size"] 
tokenized_datasets = raw_datasets.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=max_sent_size, padding='max_length'), batched=True)

# Prepare Masking
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=params["mlm_probability"], # probability of replacing a token by [MASK]
    pad_to_multiple_of=8 if params["pad_to_multiple_of_8"] else None,
)

# Training
learning_rate = float(params["learning_rate"])
batch_size = int(params["batch_size"])

training_args = TrainingArguments(checkpoint_path,
                                learning_rate=learning_rate,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                max_steps=params["steps"],
                                warmup_ratio=params["warmup_ratio"],
                                evaluation_strategy="steps",
                                eval_steps=params["eval_steps"],
                                save_strategy="no",
                                logging_dir=checkpoint_path,
                                logging_strategy="steps",
                                logging_steps=params["logging_steps"]
                                )

# Define a progress callback function
def progress_callback(info, state):
    if state.is_local_process_zero:
        print(f"Progress: {info.epoch}/{info.num_epochs} | {info.step}/{info.num_steps}")

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[progress_callback],  # Add the progress callback
)

# TRAIN!
train_result = trainer.train()
train_metrics = train_result.metrics

# save final model
trainer.save_model(final_model_path)
