import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import yaml
from datasets import Dataset
import torch

# You are already in the "10k" directory
os.chdir("pre_training_DistillBERT")

print(f"GPU: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

output_path = "pre_training_DistillBERT/"
checkpoint_path = "pre_training_DistillBERT/checkpoints/"
final_model_path = "pre_training_DistillBERT/final/"

# check if the output directories exist and, if not, create them
for dir_path in [output_path, checkpoint_path, final_model_path]:
    full_path = os.path.join(os.getcwd(), dir_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Directory {full_path} created successfully")
    else:
        print(f"Directory {full_path} already exists")

# open YAML file with parameters for pretraining
stream = open("distilbert_params.yaml", 'r')
params = yaml.load(stream, Loader=yaml.Loader)

# Load the data
df = pd.read_csv("2019_10kdata_with_covars_sample.csv")
df = df[['text']]

# Split the data into training and eval set
train_df, eval_df = train_test_split(df, test_size=0.2)

# Transform into Dataset class
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Tokenize text
model_name = params["model_name"]
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForMaskedLM.from_pretrained(model_name)

# Tokenization
max_sent_size = params["max_sent_size"]
tokenized_train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=max_sent_size, padding='max_length'), batched=True)
tokenized_eval_dataset = eval_dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, max_length=max_sent_size, padding='max_length'), batched=True)

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


# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# TRAIN!
train_result = trainer.train()
train_metrics = train_result
