import torch
import spacy
import random
import time
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext import datasets
from funcs import train_rnn, evaluate_rnn, compute_epoch_time


# ------------------------------set the seed---------------------------------
SEED = 1
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# ------------------------------prepare data---------------------------------
# define the field for text and label
TEXT = data.Field(tokenize='spacy', include_lengths=True)
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
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=num_layers,
                           bidirectional=True,
                           dropout=dropout)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # apply embedding and dropout
        embedded = self.dropout(self.embedding(text))
        # pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        # apply rnn
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # concatenate the final forward and backward hidden layers and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # apply linear and return
        return self.linear(hidden)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
NUM_LAYERS = 2
DROPOUT = 0.5
PADDING_INDEX = TEXT.vocab.stoi[TEXT.pad_token]
UNKNOWN_INDEX = TEXT.vocab.stoi[TEXT.unk_token]

# create an instance of RNN class
model = RNN(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            NUM_LAYERS,
            DROPOUT,
            PADDING_INDEX)
# replace initial weights of embedding layer with pre-trained embeddings
model.embedding.weight.data.copy_(TEXT.vocab.vectors)
# set <unk> and <pad> token to zeros
model.embedding.weight.data[PADDING_INDEX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[UNKNOWN_INDEX] = torch.zeros(EMBEDDING_DIM)


# ------------------------------train the model------------------------------
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

NUM_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    train_loss, train_accuracy = train_rnn(model, train_iterator, optimizer, criterion)
    valid_loss, valid_accuracy = evaluate_rnn(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = compute_epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'parameters/sentiment_rnn.pt')

    print(f'Epoch: {epoch + 1} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'\t Evaluate Loss: {valid_loss:.3f} |  Evaluate Accuracy: {valid_accuracy * 100:.2f}%')


# ------------------------------test the model-------------------------------
model.load_state_dict(torch.load('parameters/sentiment_rnn.pt'))
torch.save(model, "models/model_rnn.pkl")
test_loss, test_accuracy = evaluate_rnn(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy*100:.2f}%')


# ------------------------------handle user input----------------------------
def predict_sentiment(_model, _sentence):
    _model.eval()
    words = [token.text for token in nlp.tokenizer(_sentence)]
    indices = [TEXT.vocab.stoi[word] for word in words]
    length = [len(indices)]
    indices_tensor = torch.LongTensor(indices).unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(_model(indices_tensor, length_tensor))
    return prediction.item()


nlp = spacy.load('en')
review = input("Please input the movie review (quit if input 'quit'): \n")
while review != 'quit':
    score = predict_sentiment(model, review)
    print(f'The sentiment score is {score:.3f}')
    review = input("Please input the movie review (quit if input 'quit'): \n")
