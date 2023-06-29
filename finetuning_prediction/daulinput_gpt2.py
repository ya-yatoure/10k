import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from transformers import GPT2Model, GPT2TokenizerFast
from torch import nn
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import DataParallel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Define the hyperparameters
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
LEARNING_RATE_GPT2 = 1e-5
WEIGHT_DECAY = 1e-5
DATASET_FRACTION = 0.001
TRAIN_TEST_SPLIT_RATIO = 0.2

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
scaler = StandardScaler()

df = pd.read_csv("../Data/text_covars_to1024_2019_sample.csv")

# Use defined hyperparameter
df = df.sample(frac=DATASET_FRACTION)

# one hot encoding 'naics2'
df = pd.get_dummies(df, columns=['naics2'])

# group by companies when test/train splitting so we don't have companies that appear in both test and train sets
unique_companies = df['cik'].unique()
train_companies, test_companies = train_test_split(unique_companies, test_size=TRAIN_TEST_SPLIT_RATIO)

train_df = df[df['cik'].isin(train_companies)]
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

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

# Use defined hyperparameters
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)


# DEFINE MODEL ARCHITECTURE
class GPT2WithStructuredData(nn.Module):
    def __init__(self, num_structured_features, context_vector_dim):
        super(GPT2WithStructuredData, self).__init__()

        # GPT2 model for the text data (unfrozen)
        self.gpt2 = GPT2Model.from_pretrained('gpt2')

        # context vector plus structured feature NN output
        combined_dim = context_vector_dim + 8

        # A feed-forward neural network for the structured data
        self.ffnn = nn.Sequential(
            nn.Linear(num_structured_features, 24),
            nn.BatchNorm1d(24),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(24, 8),
            nn.BatchNorm1d(8)
        )

        self.combined_layer = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, structured_data):
        # Pass the text data through GPT2
        gpt2_output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        
        # Instead of global attention mechanism, take the average of hidden state embeddings
        context_vector = gpt2_output.last_hidden_state.mean(dim=1)

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
model = GPT2WithStructuredData(num_structured_features=len(structured_columns), context_vector_dim=768).to(device)

print(model)
# Freeze the GPT2 weights
for param in model.gpt2.parameters():
    param.requires_grad = False

# Separately get the parameters of the GPT2 model and the rest of your model
gpt2_params = model.gpt2.parameters()
other_params = [p for p in model.parameters() if p not in gpt2_params]

# Use defined hyperparameters
optimizer = optim.Adam([
    {'params': gpt2_params, 'lr': LEARNING_RATE_GPT2, 'weight_decay': WEIGHT_DECAY},
    {'params': other_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
])

loss_fn = nn.MSELoss()

train_losses, val_losses = [], []

# initialize these for early stopping
best_val_loss = float('inf')
no_improve_epochs = 0

for epoch in range(EPOCHS):
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
            if no_improve_epochs >= 10: 
                print("Early stopping as no improvement in validation loss for 10 consecutive epochs.")
                break
    if no_improve_epochs >= 10:
        break  # break out from epoch loop as well

# Generate predictions after all epochs
model.eval()

# Dictionary to store company ids and their respective predictions
company_predictions = {}
actuals = []

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        test_inputs, test_masks, test_structured, targets = (t.to(device) for t in batch)
        company_ids = test_cik[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]  # Adjust indices based on batch size
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

