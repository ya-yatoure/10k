# Import necessary libraries
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW, DistilBertConfig

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch
import torch.optim as optim
import pandas as pd

# Set hyperparameters
TRAIN_TEST_SPLIT_RATIO = 0.2
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-5
DATASET_FRACTION = 0.4

# Load data
df = pd.read_csv("../Data/text_covars_to512_2019_sample_90mb.csv")
df = df.sample(frac=DATASET_FRACTION)

# One hot encode 'naics2'
df = pd.get_dummies(df, columns=['naics2'])


# Define structured features and target
structured_features = ['logEMV', 'lev'] + [col for col in df.columns if 'naics2' in col]
target = 'ER_1'

# Fit a regression model using structured data
X = df[structured_features].values
y = df[target].values

reg_model = LinearRegression()
reg_model.fit(X, y)

# Calculate residuals
df['residuals'] = y - reg_model.predict(X)

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

# Define the model
class DistilBertForSequenceRegression(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, 1)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, 1)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        return logits if loss is None else (loss, logits)

# Load pre-trained model
config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
config.seq_classif_dropout = 0.2
model = DistilBertForSequenceRegression(config)

# Then continue your script from the optimizer definition

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

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
