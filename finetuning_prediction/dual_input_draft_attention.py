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
# Initialize  tokenizer and scaler
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
scaler = StandardScaler()

df = pd.read_csv("../Data/text_covars_to512_2019_sample_90mb.csv")

dataset_fraction = 0.2  # use 10% of the total data

# Use defined hyperparameter
df = df.sample(frac=DATASET_FRACTION)

# one hot encoding 'naics2'
df = pd.get_dummies(df, columns=['naics2'])

# group by companies when test/train splitting so we don't have companies that appear in both test and train sets
unique_companies = df['cik'].unique()
train_companies, test_companies = train_test_split(unique_companies, test_size=TRAIN_TEST_SPLIT_RATIO)

# Split the train_companies into train and validation
train_companies, val_companies = train_test_split(train_companies, test_size=0.2)

train_df = df[df['cik'].isin(train_companies)]
val_df = df[df['cik'].isin(val_companies)]
test_df = df[df['cik'].isin(test_companies)]
encodings = tokenizer(list(df['text']), truncation=True, padding=True)

# structured columns now include the one-hot-encoded 'naics2' columns as well
structured_columns = ['lev', 'logEMV'] + [col for col in df.columns if 'naics2' in col]

# For Train and Test set
train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True)
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])

# Only scale 'lev' and 'logEMV'
train_structured_data_to_scale = scaler.fit_transform(train_df[['lev', 'logEMV']])
train_structured_data_one_hot = train_df[[col for col in train_df.columns if 'naics2' in col]].values
train_structured_data = np.concatenate((train_structured_data_to_scale, train_structured_data_one_hot), axis=1)
train_structured_data = torch.tensor(train_structured_data, dtype=torch.float)

train_target = torch.tensor(train_df['ER_1'].values, dtype=torch.float)

test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True)
test_input_ids = torch.tensor(test_encodings['input_ids'])
test_attention_mask = torch.tensor(test_encodings['attention_mask'])

# Only scale 'lev' and 'logEMV'
test_structured_data_to_scale = scaler.transform(test_df[['lev', 'logEMV']])
test_structured_data_one_hot = test_df[[col for col in test_df.columns if 'naics2' in col]].values
test_structured_data = np.concatenate((test_structured_data_to_scale, test_structured_data_one_hot), axis=1)
test_structured_data = torch.tensor(test_structured_data, dtype=torch.float)

test_target = torch.tensor(test_df['ER_1'].values, dtype=torch.float)

train_data = TensorDataset(train_input_ids, train_attention_mask, train_structured_data, train_target)
test_data = TensorDataset(test_input_ids, test_attention_mask, test_structured_data, test_target)

test_cik = test_df['cik'].values.tolist()

val_encodings = tokenizer(list(val_df['text']), truncation=True, padding=True)
val_input_ids = torch.tensor(val_encodings['input_ids'])
val_attention_mask = torch.tensor(val_encodings['attention_mask'])

# Only scale 'lev' and 'logEMV'
val_structured_data_to_scale = scaler.transform(val_df[['lev', 'logEMV']])
val_structured_data_one_hot = val_df[[col for col in val_df.columns if 'naics2' in col]].values
val_structured_data = np.concatenate((val_structured_data_to_scale, val_structured_data_one_hot), axis=1)
val_structured_data = torch.tensor(val_structured_data, dtype=torch.float)

val_target = torch.tensor(val_df['ER_1'].values, dtype=torch.float)

val_data = TensorDataset(val_input_ids, val_attention_mask, val_structured_data, val_target)

# Use defined hyperparameters
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)



# define attention and dual input model classes here...

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs):
        energy = self.attention(encoder_outputs)
        attention = torch.softmax(energy, dim=1)
        context = torch.sum(attention * encoder_outputs, dim=1)
        return context


# DEFINE MODEL ARCHISTECTURE
class DualInputModel(nn.Module):
    def __init__(self, num_structured_features, context_vector_dim):
        super(DualInputModel, self).__init__()

        # DistilBERT model for the text data (unfrozen)
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # The global attention mechanism
        self.attention = Attention(context_vector_dim)

        # context vector plus structured feature NN output
        combined_dim = context_vector_dim + 8

        # A feed-forward neural network for the structured data
        self.ffnn = nn.Sequential(
            nn.Linear(num_structured_features, 24),
            nn.BatchNorm1d(24),  # Batch Normalization layer added here
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(24, 8),
            nn.BatchNorm1d(8)   # Batch Normalization layer added here
        )

        self.combined_layer = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.BatchNorm1d(64),  # Batch Normalization layer added here
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, structured_data):
        # Pass the text data through DistilBERT
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply the global attention mechanism
        context_vector = self.attention(distilbert_output.last_hidden_state)

        # Pass the structured data through the feed-forward neural network
        structured_embeddings = self.ffnn(structured_data)

        # Concatenate the context vector and structured embeddings
        combined = torch.cat((context_vector, structured_embeddings), dim=1)

        # Pass the combined data through the combined layer
        output = self.combined_layer(combined)

        return output

        


# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
model = DualInputModel(num_structured_features=len(structured_columns), context_vector_dim=768).to(device)


# Separately get the parameters of the DistilBERT model and the rest of your model
distilbert_params = model.distilbert.parameters()
other_params = [p for p in model.parameters() if p not in distilbert_params]

# Use a smaller learning rate for the DistilBERT parameters
optimizer = optim.Adam([
    {'params': distilbert_params, 'lr': 1e-5, 'weight_decay': 1e-5},
    {'params': other_params, 'lr': 1e-3, 'weight_decay': 1e-5}
])


loss_fn = nn.MSELoss()

epochs = 30
batch_size = 16
train_losses, val_losses = [], []

# initialize these for early stopping
best_val_loss = float('inf')
no_improve_epochs = 0

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
        
        # Early stopping
        if not is_training:  # we are in validation
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            if no_improve_epochs >= 5: 
                print("Early stopping as no improvement in validation loss for 5 consecutive epochs.")
                break
    if no_improve_epochs >= 5:
        break  # break out from epoch loop as well

# Generate predictions after all epochs
model.eval()

# Dictionary to store company ids and their respective predictions
company_predictions = {}
actuals = []

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        test_inputs, test_masks, test_structured, targets = (t.to(device) for t in batch)
        company_ids = test_cik[i * batch_size: (i + 1) * batch_size]  # Adjust indices based on batch size
        outputs = model(test_inputs, test_masks, test_structured)
        outputs = outputs.cpu().detach().numpy()

        # Group predictions by company id and take the average
        for comp_id, pred, actual in zip(company_ids, outputs, targets.cpu().numpy()):
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
