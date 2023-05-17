from transformers import DistilBertModel, DistilBertTokenizerFast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
df = pd.read_csv('2019_10kdata_with_covars_sample.csv')
df_sample = df.sample(n=6400)

# Tokenize the text data
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(list(df_sample['text']), truncation=True, padding=True)

# Get the input IDs and attention masks as PyTorch tensors
input_ids = torch.tensor(encodings['input_ids'])
attention_mask = torch.tensor(encodings['attention_mask'])

# Normalize Features
scaler = StandardScaler()
structured_data = scaler.fit_transform(df_sample[['lev', 'logEMV', 'naics2']])
structured_data = torch.tensor(structured_data, dtype=torch.float)

# Get the abnormal returns variable as a PyTorch tensor
target = torch.tensor(df_sample['ER_1'].values, dtype=torch.float)

# Split data into training and validation sets
train_inputs, val_inputs, train_masks, val_masks, train_structured, val_structured, train_target, val_target = train_test_split(
    input_ids, attention_mask, structured_data, target, test_size=0.2)

# Create TensorDatasets for the training and validation sets
train_data = TensorDataset(train_inputs, train_masks, train_structured, train_target)
val_data = TensorDataset(val_inputs, val_masks, val_structured, val_target)

# Create DataLoaders for the training and validation sets
train_dataloader = DataLoader(train_data, batch_size=32)
val_dataloader = DataLoader(val_data, batch_size=32)


class DualInputModel(nn.Module):
    def __init__(self, num_structured_features, text_embedding_dim):
        super(DualInputModel, self).__init__()

        # The DistilBERT model for the text data
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # A feed-forward neural network for the structured data
        self.ffnn = nn.Sequential(
            nn.Linear(num_structured_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        combined_dim = text_embedding_dim + 32

        # A layer to combine the outputs of the two models
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise SystemError('CUDA is not available. Please run the script on a system with CUDA installed.')

model = DualInputModel(num_structured_features=3, text_embedding_dim=768).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

epochs = 50
batch_size = 32
train_losses, val_losses = [], []



for epoch in range(epochs):
    for dataloader, is_training in [(train_dataloader, True), (val_dataloader, False)]:
        model.train(is_training)
        epoch_losses = []

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

            # Save MSE for this batch
            epoch_losses.append(loss.item())

        # Compute average MSE for this epoch and divide by batch size
        avg_loss = sum(epoch_losses) / len(epoch_losses) / len(batch)
        print(f"{'Train' if is_training else 'Validation'} loss per observation {avg_loss}")
        (train_losses if is_training else val_losses).append(avg_loss)




plt.plot(train_losses, label='Training Loss per observation')
plt.plot(val_losses, label='Validation Loss per observation')
plt.xlabel('Epoch')
plt.ylabel('Loss per observation')
plt.legend()

# Save the plot as an image file
plot_filename = 'plot.png'
plt.savefig(plot_filename)
plt.close()

from git import Repo, Actor

# Path to your local repository
# Note: Replace this path with the actual path to your local git repository
repo_path = '/10k/drafts'  
repo = Repo(repo_path)

# Create a new commit
index = repo.index
index.add([plot_filename])
author = Actor("ya-yatoure", "uctpbh1@ucl.ac.uk")  # Replace with your name and email
commit_message = 'Add plot'
index.commit(commit_message, author=author)

# Push the commit to the remote repository
origin = repo.remote('origin')
origin.push()
