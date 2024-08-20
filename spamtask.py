import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import nltk

# NLTK data download
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
file_path = 'spam1.csv'  # Ensure this file path is correct in your environment
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Keep only the necessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels (spam = 1, ham = 0)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Text preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

df['message'] = df['message'].apply(preprocess_text)

# Build vocabulary and encode messages as sequences of integers
vocab = Counter()
for message in df['message']:
    vocab.update(message)

vocab = {word: i+1 for i, (word, _) in enumerate(vocab.items())}
vocab_size = len(vocab) + 1

def encode_message(message):
    return [vocab.get(word, 0) for word in message]

df['message'] = df['message'].apply(encode_message)

# Padding sequences
max_len = max(df['message'].apply(len))
df['message'] = df['message'].apply(lambda x: x + [0]*(max_len - len(x)))

# Train-test split
X = torch.tensor(df['message'].tolist())
y = torch.tensor(df['label'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset class
class SpamDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data = SpamDataset(X_train, y_train)
test_data = SpamDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# CNN model
class SpamCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_filters, filter_sizes, output_size, dropout=0.5):
        super(SpamCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_size)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# Instantiate the model, define loss and optimizer
embed_size = 128
num_filters = 100
filter_sizes = [3, 4, 5]
output_size = 1
dropout = 0.5

model = SpamCNN(vocab_size, embed_size, num_filters, filter_sizes, output_size, dropout)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Streamlit interface
st.title("Spam Detection using CNN")

st.write("### Training Progress")
progress_bar = st.progress(0)
status_text = st.empty()

# Training loop
n_epochs = 5

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (messages, labels) in enumerate(train_loader):
        labels = labels.float().unsqueeze(1)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(messages)
        
        # Loss calculation
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Update progress bar and status text
    progress_bar.progress((epoch + 1) / n_epochs)
    status_text.text(f'Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_loader):.4f}')

st.write("### Enter a message to check if it's spam or ham:")

user_input = st.text_area("Message:")

if st.button("Predict"):
    if user_input:
        prediction = predict_message(model, user_input, vocab, max_len)
        st.write(f"The message is: **{prediction.upper()}**")
    else:
        st.write("Please enter a message.")

# Function to preprocess and encode new messages
def predict_message(model, message, vocab, max_len):
    model.eval()
    
    # Preprocess the message
    tokens = preprocess_text(message)
    encoded_message = [vocab.get(word, 0) for word in tokens]
    padded_message = encoded_message + [0] * (max_len - len(encoded_message))
    
    # Convert to tensor
    input_tensor = torch.tensor(padded_message).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()
        
    return "spam" if prediction > 0.5 else "ham"
