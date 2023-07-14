# Import necessary libraries
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW, DistilBertConfig
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# Set hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.2
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 5e-2
DATASET_FRACTION = 0.4

# Load data
df = pd.read_csv("../Data/merged_20200224_headers.csv")
df = df.sample(frac=DATASET_FRACTION)

# Target
target = 'ER_1'

# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Group by companies when test/train splitting
unique_companies = df['cik'].unique()
train_companies, test_companies = train_test_split(unique_companies, test_size=TRAIN_TEST_SPLIT_RATIO)

# Split the train_companies into train and validation
train_companies, val_companies = train_test_split(train_companies, test_size=0.2)

# Create dataframes for each set
train_df = df[df['cik'].isin(train_companies)]
val_df = df[df['cik'].isin(val_companies)]
test_df = df[df['cik'].isin(test_companies)]

# For Train, Validation and Test set
datasets = [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]
dataloaders = {}

for dataset, name in datasets:
    encodings = tokenizer(list(dataset['text']), truncation=True, padding=True)
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    targets = torch.tensor(dataset[target].values, dtype=torch.float)  # Change from 'residuals' to 'ER_1'

    data = TensorDataset(input_ids, attention_mask, targets)
    dataloaders[name] = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Define the custom model
class DistilBertForSequenceRegression(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Sequential(
            nn.Linear(config.dim, config.dim),  # First Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(config.dim, config.dim // 2),  # Second Linear layer
            nn.ReLU()  # ReLU activation
        )
        self.regressor = nn.Linear(config.dim//2, 1)  # Output layer
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        predictions = self.regressor(pooled_output)  # Predictions instead of logits

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions.view(-1), labels.view(-1))

        return predictions if loss is None else (loss, predictions)  # Return predictions instead of logits

# Load pre-trained model
config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
config.seq_classif_dropout = 0.2
model = DistilBertForSequenceRegression(config)

# Freeze the distilbert parameters
for param in model.distilbert.parameters():
    param.requires_grad = False
    
# Define the optimizer
optimizer = optim.Adam([
    {'params': model.distilbert.parameters(), 'lr': 1e-5},  # Lower learning rate for pre-trained parameters
    {'params': model.regressor.parameters()}], 
    lr=LEARNING_RATE)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()

    for batch in dataloaders['train']:
        input_ids, attention_mask, targets = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Train loss {total_loss/len(dataloaders['train'])}")

    model.eval()
    total_val_loss = 0

    for batch in dataloaders['val']:
        input_ids, attention_mask, targets = [b.to(device) for b in batch]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs[0]
        total_val_loss += loss.item()

    print(f"Val loss {total_val_loss/len(dataloaders['val'])}")

# Generate predictions on test set
model.eval()
preds = []
actuals = []

with torch.no_grad():
    for batch in dataloaders['test']:
        input_ids, attention_mask, targets = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        preds.extend(outputs[1].detach().cpu().numpy())
        actuals.extend(targets.detach().cpu().numpy())

# Compute R-squared
from sklearn.metrics import r2_score
r_squared = r2_score(actuals, preds)

print(f"R-Squared: {r_squared}")
