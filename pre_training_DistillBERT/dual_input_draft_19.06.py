
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertTokenizerFast
from torch import nn

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

df = pd.read_csv("2019_10kdata_with_covars_sample.csv")

# random sub sample of  rows
df_sample = df.sample(n=6400)

# Initialize  tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the text data
encodings = tokenizer(list(df_sample['text']), truncation=True, padding=True)

# Get the input IDs and attention masks as PyTorch tensors
input_ids = torch.tensor(encodings['input_ids'])
attention_mask = torch.tensor(encodings['attention_mask'])

scaler = StandardScaler()
structured_data = scaler.fit_transform(df_sample[['lev', 'logEMV', 'naics2']])

# Convert the structured data to a PyTorch tensor
structured_data = torch.tensor(structured_data, dtype=torch.float)

# Get the abnormal returns variable as a PyTorch tensor
target = torch.tensor(df_sample['ER_1'].values, dtype=torch.float)

train_inputs, val_inputs, train_masks, val_masks, train_structured, val_structured, train_target, val_target = train_test_split(
    input_ids, attention_mask, structured_data, target, test_size=0.2)

# Create TensorDatasets for the training and validation sets
train_data = TensorDataset(train_inputs, train_masks, train_structured, train_target)
val_data = TensorDataset(val_inputs, val_masks, val_structured, val_target)

# Create DataLoaders for the training and validation sets
train_dataloader = DataLoader(train_data, batch_size=16)
val_dataloader = DataLoader(val_data, batch_size=16)
class DualInputModel(nn.Module):
    def __init__(self, num_structured_features, text_embedding_dim):
        super(DualInputModel, self).__init__()


        #FREEZE DISTILBERT
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        for param in self.distilbert.parameters():
            param.requires_grad = False

        # Unfrozen # The DistilBERT model for the text data
        #self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # A feed-forward neural network for the structured data
        self.ffnn = nn.Sequential(
            nn.Linear(num_structured_features, 6),
            nn.ReLU(),
            nn.Linear(6, 6)
        )

        combined_dim = text_embedding_dim + 6

        # Linear combination layer
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, structured_data):
        # Pass the text data through DistilBERT
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = distilbert_output.last_hidden_state.mean(dim=1)  # Average the sequence dimension

        # Pass the structured data through the feed-forward neural network
        structured_embeddings = self.ffnn(structured_data)

        # Concatenate the text embeddings and structured embeddings
        combined = torch.cat((text_embeddings, structured_embeddings), dim=1)

        # Pass the combined embeddings through the combined layer
        output = self.combined_layer(combined)

        return output



# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DualInputModel(num_structured_features=3, text_embedding_dim=768).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

epochs = 50
batch_size = 16
train_losses, val_losses = [], []

for epoch in range(epochs):
    for dataloader, is_training in [(train_dataloader, True), (val_dataloader, False)]:
        total_loss = total_samples = 0

        model.train(is_training)
        for batch in dataloader:
            *inputs, targets = (t.to(device) for t in batch)
            targets = targets.unsqueeze(1)

            with torch.set_grad_enabled(is_training):
                outputs = model(*inputs)
                loss = loss_fn(outputs, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_samples += 1

        avg_loss = total_loss / total_samples
        print(f"{'Train' if is_training else 'Validation'} loss per observation {avg_loss}")
        (train_losses if is_training else val_losses).append(avg_loss)


# Generate predictions after all epochs
model.eval()
with torch.no_grad():
    pred = model(val_inputs.to(device), val_masks.to(device), val_structured.to(device))
pred = pred.cpu().numpy().flatten()

# Calculate R-squared
r2 = r2_score(val_target.numpy(), pred)
print('R-squared: ', r2)

plt.plot(train_losses, label='Training Loss per observation')
plt.plot(val_losses, label='Validation Loss per observation')
plt.xlabel('Epoch')
plt.ylabel('Loss per observation')
plt.legend()
plt.show()
