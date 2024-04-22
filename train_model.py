# Create a model and store the result as an artifact

from load_data import load_data
import torch
import torch.nn as nn
import torch.optim as optim


# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden.squeeze(0)
        output = self.fc(hidden)
        return output

# Train the model
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        text, labels = batch
        text = torch.LongTensor(text)
        labels = torch.LongTensor(labels)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Test the model
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text = torch.LongTensor(text)
            labels = torch.LongTensor(labels)
            output = model(text)
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return epoch_loss / len(iterator), accuracy


def train_model(model, train_loader, valid_loader, num_epochs, optimizer, model_path):
    best_accuracy = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        valid_loss, accuracy = evaluate(model, valid_loader, criterion)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Test Loss: {valid_loss:.3f}, Accuracy: {accuracy:.3f}')
        
        # Save the model if it achieves the best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
	# Load data
	train_csv_path = 'train.csv'
	test_csv_path = 'test.csv'
	train_loader, valid_loader, test_loader, vocab, tokenizer = load_data(train_csv_path, test_csv_path=test_csv_path, test_size=0.2)
	vocab_size = len(vocab)
	# Instantiate the model
	model = LSTMModel(vocab_size)
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	# Train the model and save weights
	N_EPOCHS = 5
	best_accuracy = 0
	model_path = './models/best_model1.pth'

	train_model(model, train_loader, valid_loader, optimizer, criterion, N_EPOCHS, model_path)

