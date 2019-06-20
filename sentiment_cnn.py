import torch
import spacy
import random
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torchtext import datasets
from funcs import train_cnn, evaluate_cnn, compute_epoch_time


# ------------------------------set the seed---------------------------------
SEED = 1
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# ------------------------------prepare data---------------------------------
# define the field for text and label
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

# load the IMDb dataset
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

# build the vocabulary
MAX_VOCAB_SIZE = 25000
TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

# create the iterators
BATCH_SIZE = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True)


# ------------------------------build the model------------------------------
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=num_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.linear = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)
        # text = [batch_size, sent_len]
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        # embedded = [batch_size, 1, sent_len. emb_dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch_size, num_filters, sent_len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch_size, num_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch_size, num_filters * len(filter_sizes)]
        return self.linear(cat)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
NUM_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PADDING_INDEX = TEXT.vocab.stoi[TEXT.pad_token]
UNKNOWN_INDEX = TEXT.vocab.stoi[TEXT.unk_token]

# create an instance of CNN class
model = CNN(INPUT_DIM,
            EMBEDDING_DIM,
            NUM_FILTERS,
            FILTER_SIZES,
            OUTPUT_DIM,
            DROPOUT,
            PADDING_INDEX)
# replace initial weights of embedding layer with pre-trained embeddings
model.embedding.weight.data.copy_(TEXT.vocab.vectors)
# set <unk> and <pad> token to zeros
model.embedding.weight.data[PADDING_INDEX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[UNKNOWN_INDEX] = torch.zeros(EMBEDDING_DIM)


# ------------------------------train the model------------------------------
print("start training")
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

NUM_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    train_loss, train_accuracy = train_cnn(model, train_iterator, optimizer, criterion)
    valid_loss, valid_accuracy = evaluate_cnn(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = compute_epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'parameters/sentiment_cnn.pt')

    print(f'Epoch: {epoch + 1} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'\t Evaluate Loss: {valid_loss:.3f} |  Evaluate Accuracy: {valid_accuracy * 100:.2f}%')


# ------------------------------test the model-------------------------------
model.load_state_dict(torch.load('parameters/sentiment_cnn.pt'))
torch.save(model, "models/model_cnn.pkl")
test_loss, test_accuracy = evaluate_cnn(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy*100:.2f}%')


# ------------------------------handle user input----------------------------
def predict_sentiment(_model, _sentence):
    _model.eval()
    words = [token.text for token in nlp.tokenizer(_sentence)]
    if len(words) < 5:
        words += ['<pad>'] * (5 - len(words))
    indices = [TEXT.vocab.stoi[word] for word in words]
    indices_tensor = torch.LongTensor(indices).unsqueeze(1)
    prediction = torch.sigmoid(_model(indices_tensor))
    return prediction.item()


nlp = spacy.load('en')
review = input("Please input the movie review (quit if input 'quit'): \n")
while review != 'quit':
    score = predict_sentiment(model, review)
    print(f'The sentiment score is {score:.3f}')
    review = input("Please input the movie review (quit if input 'quit'): \n")
