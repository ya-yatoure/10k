# Import necessary libraries
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW, DistilBertConfig

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch
import torch.optim as optim
import pandas as pd

# Set hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.4
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 5e-2
DATASET_FRACTION = 1.0

# Load data
df = pd.read_csv("../Data/text_covars_to512_2019HEADERS.csv")
df = df.sample(frac=DATASET_FRACTION)

# One-hot encode 'naics2' and 'day_type' columns
df = pd.get_dummies(df, columns=['naics2', 'day_type'])
day_type_columns = [col for col in df.columns if 'day_type' in col]


# print out the number of unique 'cik' values 
print(f"Number of unique companies: {len(df['cik'].unique())}")

# print the number of input sequences we have (ie rows)
print(f"Number of input sequences: {len(df)}")

# Define structured features and target
structured_features = ['logEMV', 'lev'] + [col for col in df.columns if 'naics2' in col]
target = 'ER_1'

# Initialize an empty DataFrame to store the processed data
df_processed = pd.DataFrame()

# Loop over each unique 'day_type'
for day_type in df['day_type'].unique():
    # Filter df by current 'day_type' and create a copy
    df_day_type = df[df['day_type'] == day_type].copy()
    
    # Fit a regression model using structured data for the current 'day_type'
    X = df_day_type[structured_features].values
    y = df_day_type[target].values

    reg_model = LinearRegression()
    reg_model.fit(X, y)

    # Calculate residuals
    df_day_type['residuals'] = y - reg_model.predict(X)

    # Append the processed data to df_processed
    df_processed = pd.concat([df_processed, df_day_type])

# Update df with df_processed
df = df_processed

print(df.head())
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

print(f"Null values in train text: {train_df['text'].isnull().sum()}")
print(f"Null values in val text: {val_df['text'].isnull().sum()}")
print(f"Null values in test text: {test_df['text'].isnull().sum()}")

# Replace null values with empty string
train_df['text'].fillna("", inplace=True)
val_df['text'].fillna("", inplace=True)
test_df['text'].fillna("", inplace=True)

# If there are non-string values, you might need to convert them to string
train_df['text'] = train_df['text'].apply(str)
val_df['text'] = val_df['text'].apply(str)
test_df['text'] = test_df['text'].apply(str)

# For Train, Validation and Test set
datasets = [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]
dataloaders = {}

for dataset, name in datasets:
    encodings = tokenizer(list(dataset['text']), truncation=True, padding=True)
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    
    # Include the one-hot encoded day type data
    day_type_data = torch.tensor(dataset[day_type_columns].values, dtype=torch.float)
    
    target = torch.tensor(dataset['residuals'].values, dtype=torch.float)
    
    data = TensorDataset(input_ids, attention_mask, day_type_data, target)
    dataloaders[name] = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

class DistilBertForSequenceRegression(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        num_day_type = len(day_type_columns)
        
        self.day_type_layer = nn.Sequential(
            nn.Linear(num_day_type, 50),  # Adjust the hidden layer size as needed
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pre_classifier = nn.Sequential(
            nn.Linear(config.dim + 50, config.dim),  # Adjust the input size
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.regressor = nn.Linear(config.dim, 1)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, day_type_data=None, labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]

        day_type_output = self.day_type_layer(day_type_data)
        pooled_output = torch.cat((pooled_output, day_type_output), dim=1)

        pooled_output = self.pre_classifier(pooled_output)
        predictions = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions.view(-1), labels.view(-1))
        return predictions if loss is None else (loss, predictions)


# Load pre-trained model
config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
config.seq_classif_dropout = 0.2
model = DistilBertForSequenceRegression(config)

# Freeze the distilbert parameters
for param in model.distilbert.parameters():
    param.requires_grad = False

# Define the optimizer
optimizer = optim.Adam([
    {'params': model.distilbert.parameters()},
    {'params': model.regressor.parameters()}], 
    lr=LEARNING_RATE)


# Training loop
# Check if CUDA is available and set device to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to device
model = model.to(device)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for batch in dataloaders['train']:
        input_ids, attention_mask, day_type_data, targets = [b.to(device) for b in batch]
        # Move data to device
        input_ids, attention_mask, targets = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs[0]  # access loss from the tuple
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(dataloaders['train'])
    print(f"Average training loss: {avg_train_loss}")

    # Validation loop
    model.eval()
    total_eval_loss = 0
    for batch in dataloaders['val']:
        # Move data to device
        input_ids, attention_mask, targets = [b.to(device) for b in batch]
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs[0]  # access loss from the tuple
        total_eval_loss += loss.item()

    avg_val_loss = total_eval_loss / len(dataloaders['val'])
    print(f"Validation Loss: {avg_val_loss}")

# Generate predictions on test set
model.eval()
preds = []
actuals = []

with torch.no_grad():
    for batch in dataloaders['test']:
        # Move data to device
        input_ids, attention_mask, targets = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        # outputs[1] gives the predictions from the model
        preds.extend(outputs[1].detach().cpu().numpy())
        actuals.extend(targets.detach().cpu().numpy())

# Compute R-squared
r_squared = r2_score(actuals, preds)

print(f"Out of Sample R-Squared: {r_squared}")
