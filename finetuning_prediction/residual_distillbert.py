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

# Here we split the train_companies into train and validation
train_companies, val_companies = train_test_split(train_companies, test_size=0.2)

train_df = df[df['cik'].isin(train_companies)]
val_df = df[df['cik'].isin(val_companies)]  # New validation data frame
test_df = df[df['cik'].isin(test_companies)]

# For Train and Test set:
# input IDs and attention masks as PyTorch tensors
# structured data to a PyTorch tensor
# abnormal returns variable as a PyTorch tensor
# Create data loaders
batch_size = 16

# Encode the validation data
val_encodings = distilbert_tokenizer(list(val_df['text']), truncation=True, padding=True)
val_input_ids = torch.tensor(val_encodings['input_ids'])
val_attention_mask = torch.tensor(val_encodings['attention_mask'])
val_target = torch.tensor(val_df['ER_1_residuals'].values, dtype=torch.float)

val_data = TensorDataset(val_input_ids, val_attention_mask, val_target)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)  # New validation data loader

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

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define Attention class
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


# Define DistilBERTRegressor class
class DistilBERTRegressor(nn.Module):
    def __init__(self):
        super(DistilBERTRegressor, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # Introduce attention and dropout layers
        self.attention = Attention(self.distilbert.config.dim, self.distilbert.config.max_position_embeddings)
        self.dropout = nn.Dropout(0.5)
        
        self.linear = nn.Linear(self.distilbert.config.dim, 1)

    def forward(self, input_ids, attention_mask):
        # get embeddings from DistilBERT model
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # apply attention
        attention_output = self.attention(distilbert_output)
        
        # apply dropout
        dropout_output = self.dropout(attention_output)

        # pass through linear layer
        output = self.linear(dropout_output)
        
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

# Initialize early stopping variables
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

# Training loop
for epoch in range(n_epochs):
    model.train()
    total_loss = 0

    # Actual training part
    for i, batch in enumerate(train_loader):
        # Get data
        input_ids, attention_mask, targets = batch
        # Move data to the GPU if available
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Make predictions
        outputs = model(input_ids, attention_mask)
        # Compute loss
        loss = criterion(outputs, targets)
        # Perform backpropagation
        loss.backward()
        # Update the weights
        optimizer.step()

        total_loss += loss.item()

    print("Epoch: {}, Training Loss: {}".format(epoch, total_loss / len(train_loader)))

    # After each epoch of training, evaluate on the validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Get data
            input_ids, attention_mask, targets = batch
            # Move data to the GPU if available
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            # Make predictions
            outputs = model(input_ids, attention_mask)
            # Compute loss
            loss = criterion(outputs, targets)
            val_loss += loss.item()

        val_loss = val_loss / len(val_loader)  # average loss

    print("Epoch: {}, Validation Loss: {}".format(epoch, val_loss))

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping!")
            break  # break the training loop

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
