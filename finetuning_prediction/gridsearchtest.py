# Import necessary libraries
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW, DistilBertConfig
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# YOUR NEW CODE HERE
# Define the FFN
class FFN(nn.Module):
    def __init__(self, hidden_dim, dropout_p=0.1):
        super(FFN, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)  # Output dimension is 1 for regression task

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

# Define the model
class DistilBertForSequenceRegression(nn.Module):
    def __init__(self, config, ffn_hidden_dim=256, dropout_p=0.1):
        super(DistilBertForSequenceRegression, self).__init__()
        self.distilbert = DistilBertModel(config)
        self.ffn = FFN(hidden_dim=ffn_hidden_dim, dropout_p=dropout_p) # Modified line
        
    def forward(self, input_ids=None, attention_mask=None):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]  # get the hidden state from the DistilBERT model
        pooled_output = hidden_state[:, 0]  # take the [CLS] token representation
        logits = self.ffn(pooled_output)  # pass it to the FFN
        return logits
# END OF YOUR NEW CODE

# Define hyperparameters grid
hyperparams_grid = {'batch_size': [16, 32], 'learning_rate': [0.01, 0.05], 'epochs': [50, 100], 'dataset_fraction': [0.5]}

# Load data
df = pd.read_csv("../Data/text_covars_to512_2019HEADERS.csv")

# only days with fiscal policy stimuls
df = df[df['day_type'] == 'fiscal_policy_stimulus']

# Get unique companies
companies = df['cik'].unique()

# Shuffle companies
np.random.shuffle(companies)

# Split into train, validation and test
train, validate, test = np.split(companies, [int(.6*len(companies)), int(.8*len(companies))])

train_companies = train.tolist()
val_companies = validate.tolist()
test_companies = test.tolist()

# Loop over hyperparameters grid
for hyperparams in ParameterGrid(hyperparams_grid):
    # Assign hyperparameters
    BATCH_SIZE = hyperparams['batch_size']
    LEARNING_RATE = hyperparams['learning_rate']
    EPOCHS = hyperparams['epochs']
    DATASET_FRACTION = hyperparams['dataset_fraction']
    TRAIN_TEST_SPLIT_RATIO = 0.2

    # Sample the dataframe
    df_sample = df.sample(frac=DATASET_FRACTION)

    # keep only clumns where day_type == fiscal_policy_stimulus
    df = df[df['day_type'] == 'fiscal_policy_stimulus']

    # One-hot encode 'naics2'
    df = pd.get_dummies(df, columns=['naics2'])

    # Define structured features and target
    structured_features = ['logEMV', 'lev'] + [col for col in df.columns if 'naics2' in col]
    target = 'ER_1'

    # Fit a regression model using structured data for the current 'day_type'
    X = df[structured_features].values
    y = df[target].values

    reg_model = LinearRegression()
    reg_model.fit(X, y)

    # Calculate residuals
    df['residuals'] = y - reg_model.predict(X)

    print(df.head())

    # Print the r-squared score
    print(f"R-squared score: {r2_score(y, reg_model.predict(X))}")

    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Create dataframes for each set
    train_df = df[df['cik'].isin(train_companies)]
    val_df = df[df['cik'].isin(val_companies)]
    test_df = df[df['cik'].isin(test_companies)]

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

        target = torch.tensor(dataset['residuals'].values, dtype=torch.float)

        data = TensorDataset(input_ids, attention_mask, target)
        dataloaders[name] = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # YOUR NEW CODE HERE
    # Load pre-trained model
    config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceRegression(config)
    # END OF YOUR NEW CODE

    # Freeze the distilbert parameters
    for param in model.distilbert.parameters():
        param.requires_grad = False

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Modified line

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(EPOCHS):
        # (same as in your original script)

    # Generate predictions on test set
    model.eval()
    preds = []
    actuals = []

    with torch.no_grad():
        for batch in dataloaders['test']:
            input_ids, attention_mask, targets = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(outputs.detach().cpu().numpy())
            actuals.extend(targets.detach().cpu().numpy())

    # Compute R-squared
    r_squared = r2_score(actuals, preds)
    print(f"Out of Sample R-Squared: {r_squared} with BATCH_SIZE: {BATCH_SIZE}, LEARNING_RATE: {LEARNING_RATE}, EPOCHS: {EPOCHS}, DATASET_FRACTION: {DATASET_FRACTION}")
