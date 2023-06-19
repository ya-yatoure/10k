
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

# group by companies when test/train splitting so we dont have companies that appear in test and trainset
unique_companies = df['company_id'].unique()
train_companies, test_companies = train_test_split(unique_companies, test_size=0.2)

train_df = df[df['company_id'].isin(train_companies)]
test_df = df[df['company_id'].isin(test_companies)]


# Initialize  tokenizer and scaler
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
scaler = StandardScaler()

encodings = tokenizer(list(df['text']), truncation=True, padding=True)


# For Train and Test set:
# input IDs and attention masks as PyTorch tensors
# structured data to a PyTorch tensor
# abnormal returns variable as a PyTorch tensor

train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True)
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])
train_structured_data = scaler.fit_transform(train_df[['lev', 'logEMV', 'naics2']])
train_target = torch.tensor(train_df['ER_1'].values, dtype=torch.float)

test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True)
test_input_ids = torch.tensor(test_encodings['input_ids'])
test_attention_mask = torch.tensor(test_encodings['attention_mask'])
test_structured_data = scaler.transform(test_df[['lev', 'logEMV', 'naics2']])
test_target = torch.tensor(test_df['ER_1'].values, dtype=torch.float)


train_data = TensorDataset(train_input_ids, train_attention_mask, train_structured_data, train_target)
test_data = TensorDataset(test_input_ids, test_attention_mask, test_structured_data, test_target, torch.tensor(test_df['company_id'].values))  # including company_id for grouping later
train_dataloader = DataLoader(train_data, batch_size=16)
test_dataloader = DataLoader(test_data, batch_size=16)


# split train_data into training set and validation set
train_size = int(0.8 * len(train_data))  # 80% for training
val_size = len(train_data) - train_size  # the rest for validation
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

# DataLoaders for training and validation sets
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
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
        
        # layers after combination
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
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



# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DualInputModel(num_structured_features=3, text_embedding_dim=768).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

epochs = 10
batch_size = 16
train_losses, val_losses = [], []

for epoch in range(epochs):
    for dataloader, is_training in [(train_dataloader, True), (test_dataloader, False)]:
        
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
# Generate predictions after all epochs
model.eval()

# Dictionary to store company ids and their respective predictions
company_predictions = {}
actuals = []

with torch.no_grad():
    for batch in test_dataloader:
        test_inputs, test_masks, test_structured, targets, company_ids = (t.to(device) for t in batch)
        outputs = model(test_inputs, test_masks, test_structured)
        outputs = outputs.cpu().detach().numpy()

        # Group predictions by company id and take the average
        for comp_id, pred, actual in zip(company_ids.cpu().numpy(), outputs, targets.cpu().numpy()):
            if comp_id in company_predictions:
                company_predictions[comp_id].append(pred)
            else:
                company_predictions[comp_id] = [pred]
                actuals.append(actual)  # Only need one actual value per company

# Average the predictions for each company
avg_company_predictions = {comp_id: np.mean(preds) for comp_id, preds in company_predictions.items()}

# The 'avg_company_predictions' dictionary now holds the average prediction for each company
predictions = np.array(list(avg_company_predictions.values()))
actuals = np.array(actuals)

r2 = r2_score(actuals, predictions)
print(f"R^2 score: {r2}")

plt.plot(train_losses, label='Training Loss per observation')
plt.plot(val_losses, label='Validation Loss per observation')
plt.xlabel('Epoch')
plt.ylabel('Loss per observation')
plt.legend()
plt.show()

