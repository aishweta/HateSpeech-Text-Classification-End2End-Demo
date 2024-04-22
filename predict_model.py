# Make a prediction on some input data
import torch
from torchtext.data.utils import get_tokenizer
from load_data import load_data
from train_model import LSTMModel
import pickle

def read_vocab_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

# Example usage:
vocab = read_vocab_from_pickle('vocab.pkl')

def predict(input_data, pretrained_model_path):
    tokenizer = get_tokenizer("basic_english")
    
    # Load the pretrained model
    vocab_size = len(vocab)
    model = LSTMModel(vocab_size)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()  # Set the model to evaluation mode

    # Preprocess the input data (tokenization, numericalization, padding, etc.)
    max_length = 100  # Adjust as per your preprocessing steps
    tokens = tokenizer(input_data)[:max_length]
    ids = [vocab[token] for token in tokens if token in vocab]
    padded_ids = ids + [0] * (max_length - len(ids))  # Padding

    # Convert the preprocessed data to a PyTorch tensor
    input_tensor = torch.tensor([padded_ids])

    # Pass the input tensor through the model
    with torch.no_grad():
        output = model(input_tensor)

    # Interpret the model's output to get predictions
    _, predicted = torch.max(output, 1)
    prediction = predicted.item()

    # Define a dictionary mapping binary labels to actual labels
    label_mapping = {1: "hate", 0: "noHate"}

    # Convert the predicted label to the actual label
    actual_label = label_mapping[prediction]

    return actual_label

if __name__ == "__main__":
    input_data = "I can't stand that person, they're so annoying."
    pretrained_model_path = './models/best_model.pth'
    prediction = predict(input_data, pretrained_model_path)
    print("Prediction:", prediction)
