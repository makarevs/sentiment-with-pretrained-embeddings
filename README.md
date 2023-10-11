# sentiment-with-pretrained-embeddings
use the FastText pre-trained vectors for the English language and perform sentiment analysis on the large movie review dataset from IMDb

For this sample program, we'll use the FastText pre-trained vectors for the English language and perform sentiment analysis on the large movie review dataset from IMDb.

1. Install necessary libraries:

```bash
pip install fasttext torch
```
If that fails,
```bash
pip install fasttext-wheel
```
See https://pypi.org/project/fasttext-wheel/

2. Download the pre-trained FastText model:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
```

3. Download and extract the IMDb movie reviews dataset:

```bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xzf aclImdb_v1.tar.gz
```

4. Create the sample Python program (e.g., `fasttext_sentiment_analysis.py`):

```python
import fasttext
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load FastText pre-trained model
model = fasttext.load_model('cc.en.300.bin')

# IMDb dataset directory
dataset_dir = 'aclImdb'

def read_reviews(sentiment, dataset_type):
    reviews = []
    path = os.path.join(dataset_dir, dataset_type, sentiment)
    files = os.listdir(path)
    for file in files:
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            text = f.read()
            reviews.append(text)
    return reviews

# Read train and test datasets
train_pos_reviews = read_reviews('pos', 'train')
train_neg_reviews = read_reviews('neg', 'train')
test_pos_reviews = read_reviews('pos', 'test')
test_neg_reviews = read_reviews('neg', 'test')

# Merge datasets and create labels
train_reviews = train_pos_reviews + train_neg_reviews
train_labels = [1 for _ in train_pos_reviews] + [0 for _ in train_neg_reviews]
test_reviews = test_pos_reviews + test_neg_reviews
test_labels = [1 for _ in test_pos_reviews] + [0 for _ in test_neg_reviews]

def review_to_embedding(review, max_seq_len=100):
    words = review.split()
    words = words[:max_seq_len]
    embedding = torch.zeros(max_seq_len, 300)
    for i, word in enumerate(words):
        embedding[i] = torch.tensor(model.get_word_vector(word))
    return embedding

class ImdbDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels
    
    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return review_to_embedding(self.reviews[idx]), self.labels[idx]

# DataLoader settings
batch_size = 64

train_data = ImdbDataset(train_reviews, train_labels)
test_data = ImdbDataset(test_reviews, test_labels)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Define the model
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.lstm = nn.LSTM(300, 128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = SentimentClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

# Training
num_epochs = 5

print("Training started...\n")
for epoch in range(num_epochs):
    classifier.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # Evaluate after every epoch
    correct = 0
    total = 0
    classifier.eval()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = classifier(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
    print("\nTest Accuracy: {:.2f}%\n".format(acc))

print("Training completed.")
```

5. Run the sample program:

```bash
python fasttext_sentiment_analysis.py
```

The above sample program uses FastText pre-trained vectors to create embeddings for the movie reviews from the IMDb dataset. Then, it uses a simple LSTM-based model for sentiment analysis. The script trains the model for 5 epochs and evaluates its performance on the test set after every epoch.