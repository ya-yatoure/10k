# Import necessary libraries
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW, DistilBertConfig
from transformers import get_linear_schedule_with_warmup

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch
import torch.optim as optim
import pandas as pd

# Define hyperparameters
LEARNING_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 16
DATASET_FRACTION = 0.02
TRAIN_TEST_SPLIT_RATIO = 0.2
GRID_SEARCH = False  # Set this to True if you want to perform grid search

# Grid of hyperparameters
if GRID_SEARCH:
    hyperparams_grid = {
        'learning_rate': [0.01, 0.1],
        'epochs': [10, 20],
        'batch_size': [16, 32],
        'nn_structure': [(768, 1), (768, 30, 1), (768, 30, 30, 1)]  # tuples represent sizes of layers in the additional network
    }


# Load data n
df = pd.read_csv("../Data/text_covars_to512.csv")
df = df.sample(frac=DATASET_FRACTION)


# One-hot encode 'naics2'
df = pd.get_dummies(df, columns=['naics2'])
print(df.columns)


# Define structured features and target
structured_features = ['logEMV', 'lev'] + [col for col in df.columns if 'naics2' in col]
target = 'ER_1'

# Fit a regression model using structured data for the current 'day_type'
X = df[structured_features].values
y = df[target].values

reg_model = LinearRegression()
reg_model.fit(X, y)

# Print the R-squared for the regression model
print(f'Training set R-squared for regression model: {reg_model.score(X, y)}')

# Calculate residuals
df['residuals'] = y - reg_model.predict(X)


# Model definition
class DistilBertForSequenceRegression(DistilBertModel):
    def __init__(self, config, nn_structure):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        
        # Freeze the DistilBert parameters
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # Define additional layers on top of distilbert based on the nn_structure
        layers = []
        for i in range(len(nn_structure)):
            layers.append(nn.Linear(nn_structure[i][0], nn_structure[i][1]))
            if i != len(nn_structure) - 1:  # we don't want ReLU after the last layer
                layers.append(nn.ReLU())
        self.additional_nn = nn.Sequential(*layers)
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]  # Shape : [batch_size, seq_length, hidden_dim]
        pooled_output = hidden_state[:, 0]  # Shape : [batch_size, hidden_dim]. We are taking the [CLS] representation

        predictions = self.additional_nn(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions.view(-1), labels.view(-1))
        return predictions if loss is None else (loss, predictions)


# Train test split
unique_companies = df['cik'].unique()
train_companies, test_companies = train_test_split(unique_companies, test_size=TRAIN_TEST_SPLIT_RATIO)
train_companies, val_companies = train_test_split(train_companies, test_size=0.2)

# Create dataframes for each set
train_df = df[df['cik'].isin(train_companies)]
val_df = df[df['cik'].isin(val_companies)]
test_df = df[df['cik'].isin(test_companies)]

# More preprocessing
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df['text'].fillna("", inplace=True)
val_df['text'].fillna("", inplace=True)
test_df['text'].fillna("", inplace=True)


# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize and create dataloaders
datasets = [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]
dataloaders = {}
for dataset, name in datasets:
    encodings = tokenizer(list(dataset['text']), truncation=True, padding=True)
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    target = torch.tensor(dataset['residuals'].values, dtype=torch.float)
    data = TensorDataset(input_ids, attention_mask, target)
    dataloaders[name] = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Load pre-trained model
config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
NN_STRUCTURE = [(768, 30), (30, 1)]  # define structure for additional layers
model = DistilBertForSequenceRegression(config, NN_STRUCTURE)


# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

if GRID_SEARCH:
    best_r2 = -float("inf")
    best_params = None

    for params in ParameterGrid(hyperparams_grid):
        LEARNING_RATE = params['learning_rate']
        EPOCHS = params['epochs']
        BATCH_SIZE = params['batch_size']
        NN_STRUCTURE = params['nn_structure']
        
        model = DistilBertForSequenceRegression(config, NN_STRUCTURE).to(device)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

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

            if epoch % 2 == 0:  # Print losses every 2 epochs
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch in dataloaders['val']:
                    input_ids, attention_mask, targets = [b.to(device) for b in batch]
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_func(outputs.view(-1), targets.view(-1))
                    val_loss += loss.item()

            print(f"Epoch: {epoch+1}, Train Loss: {total_loss/len(dataloaders['train'])}, Validation Loss: {val_loss/len(dataloaders['val'])}")

        model.eval()
        predictions = []
        true_values = []
        with torch.no_grad():
            for batch in dataloaders['test']:
                input_ids, attention_mask, targets = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions.extend(outputs.detach().cpu().numpy())
                true_values.extend(targets.detach().cpu().numpy())
        
        r2 = r2_score(true_values, predictions)
        print(f'R2 score with learning rate {LEARNING_RATE} and epochs {EPOCHS}: {r2}')

        if r2 > best_r2:
            best_r2 = r2
            best_params = params

    print(f'Best parameters are {best_params} with R2 score: {best_r2}')

else:
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
        if epoch % 2 == 0:  # Print losses every 2 epochs
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in dataloaders['val']:
                input_ids, attention_mask, targets = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_func(outputs.view(-1), targets.view(-1))
                val_loss += loss.item()
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for batch in dataloaders['test']:
            input_ids, attention_mask, targets = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.detach().cpu().numpy())
            true_values.extend(targets.detach().cpu().numpy())

    r2 = r2_score(true_values, predictions)
    print(f'R2 score: {r2}')
