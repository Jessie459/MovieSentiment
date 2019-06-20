import torch


def compute_batch_accuracy(predictions, y):
    """Return accuracy per batch"""
    rounded_predictions = torch.round(torch.sigmoid(predictions))
    correct = (rounded_predictions == y)
    accuracy = correct.sum().float() / len(correct)
    return accuracy


def compute_epoch_time(start_time, end_time):
    """Compute how long an epoch takes"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_rnn(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_accuracy = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        accuracy = compute_batch_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def train_cnn(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_accuracy = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        accuracy = compute_batch_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def evaluate_rnn(model, iterator, criterion):
    epoch_loss = 0
    epoch_accuracy = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            accuracy = compute_batch_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def evaluate_cnn(model, iterator, criterion):
    epoch_loss = 0
    epoch_accuracy = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            accuracy = compute_batch_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)
