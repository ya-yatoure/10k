from transformers import GPT2Model, GPT2TokenizerFast, DistilBertModel, DistilBertTokenizerFast
from torch import nn
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import DataParallel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Initialize DistilBERT tokenizer
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

scaler = StandardScaler()

# load data
df = pd.read_csv("../Data/text_covars_to512_2019_sample_90mb.csv")

dataset_fraction = 0.3  # use frac of data

# Sample dataset_fraction of the data
df = df.sample(frac=dataset_fraction)


# Fit initial regression model
initial_regressor = LinearRegression()
initial_regressor.fit(df[['lev', 'logEMV', 'naics2']], df['ER_1'])

# Get residuals
df['ER_1_residuals'] = df['ER_1'] - initial_regressor.predict(df[['lev', 'logEMV', 'naics2']])


# Group by companies when test/train splitting so we don't have companies that appear in test and trainset
unique_companies = df['cik'].unique()
train_companies, test_companies = train_test_split(unique_companies, test_size=0.2)

train_df = df[df['cik'].isin(train_companies)]
test_df = df[df['cik'].isin(test_companies)]



# For Train and Test set:
# input IDs and attention masks as PyTorch tensors
# structured data to a PyTorch tensor
# abnormal returns variable as a PyTorch tensor
train_encodings = distilbert_tokenizer(list(train_df['text']), truncation=True, padding=True)
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])
train_target = torch.tensor(train_df['ER_1_residuals'].values, dtype=torch.float)

test_encodings = distilbert_tokenizer(list(test_df['text']), truncation=True, padding=True)
test_input_ids = torch.tensor(test_encodings['input_ids'])
test_attention_mask = torch.tensor(test_encodings['attention_mask'])
test_target = torch.tensor(test_df['ER_1_residuals'].values, dtype=torch.float)

train_data = TensorDataset(train_input_ids, train_attention_mask, train_target)
test_data = TensorDataset(test_input_ids, test_attention_mask, test_target)


# Create data loaders
batch_size = 16

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class DistilBERTRegressor(nn.Module):
    def __init__(self):
        super(DistilBERTRegressor, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.2)
        self.ffnet = nn.Linear(self.distilbert.config.dim, 1)  # ffnet

    def forward(self, input_ids, attention_mask):
        # get embeddings from DistilBERT model
        embeddings = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embeddings = self.dropout(embeddings)

        # pool over sequence dimension
        embeddings = embeddings.mean(dim=1)

        # pass pooled embeddings through the ffnet
        output = self.ffnet(embeddings)
        return output.squeeze(-1)


model = DistilBERTRegressor()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
n_epochs = 25  # adjust as needed

for epoch in range(n_epochs):
    model.train()  # turn on training mode
    total_loss = 0
    print("Starting epoch", epoch)

    for i, batch in enumerate(train_loader):
        # Get data
        input_ids, attention_mask, targets = batch

        # Move data to the GPU if available
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = model(input_ids, attention_mask)

        # Compute loss
        loss = criterion(outputs, targets)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print intermediate loss
        if i % 100 == 0:  # print every 100 batches
            print("Batch: {}, Loss: {}".format(i, loss.item()))

        total_loss += loss.item()

    print("Epoch: {}, Loss: {}".format(epoch, total_loss))

# switch to evaluation mode
model.eval()

with torch.no_grad():
    predictions = []
    actuals = []

    for batch in test_loader:
        # Get data
        input_ids, attention_mask, targets = batch

        # Move data to the GPU if available
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        # Make predictions
        outputs = model(input_ids, attention_mask)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(targets.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate R-squared score
    r2 = r2_score(actuals, predictions)

print('Out of sample R-squared:', r2)
