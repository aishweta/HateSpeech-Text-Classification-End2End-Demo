import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch


def get_vocab_and_tokenizer(dataframe):
    tokenizer = get_tokenizer("basic_english")

    # Build vocabulary from training data
    def yield_tokens(dataframe):
        for t in dataframe['text']:
            yield tokenizer(t)

    vocab = build_vocab_from_iterator(yield_tokens(dataframe), specials=["<unk>"])
    
    # Save the vocabulary to a file
    vocab_file_path = 'vocab.pkl'
    with open(vocab_file_path, 'wb') as file:
        pickle.dump(vocab, file)
    
    return tokenizer, vocab

# Custom Dataset class to handle text data
class TextDataset(Dataset):
    def __init__(self, dataframe, vocab, tokenizer, max_length):
        self.data = dataframe
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        tokens = self.tokenizer(text)[:self.max_length]
        ids = [self.vocab[token] for token in tokens if token in self.vocab]
        padded_ids = ids + [0] * (self.max_length - len(ids))  # Padding
        return padded_ids, label

# Function to collate batches and convert text to tensors
def collate_batch(batch):
    texts, labels = zip(*batch)
    text_tensors = [torch.tensor(text) for text in texts]  # Convert text to tensor
    label_tensor = torch.LongTensor(labels)  # Convert labels to tensor
    padded_text = pad_sequence(text_tensors, batch_first=True)  # Pad sequences to equal length
    return padded_text, label_tensor

def txt_to_csv(text_folder_path, annotation_csv_path, output_csv_path='train.csv'):
    annotations_df = pd.read_csv(annotation_csv_path)
    
    # Create an empty DataFrame
    combined_df = pd.DataFrame(columns=['file_id', 'text', 'label'])
    
    # Iterate over each .txt file in the folder
    for filename in os.listdir(text_folder_path):
        if filename.endswith('.txt'):
            file_id = filename.split('.')[0]  # Extract file_id from filename
            with open(os.path.join(text_folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()

            # Match file_id with annotations and retrieve label
            label = annotations_df.loc[annotations_df['file_id'] == file_id, 'label'].values[0]

            # Append data to the combined
            combined_df = combined_df.append({'file_id': file_id, 'text': text, 'label': label}, ignore_index=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_csv_path, index=False)

def load_data(train_csv_path, test_csv_path=None, vocab=None, tokenizer=None, max_length=100, test_size=0.2, random_state=42):
    # Load train data from CSV file
    train_df = pd.read_csv(train_csv_path)
    
    # If test CSV path is provided, load test data; otherwise, set it to None
    test_df = pd.read_csv(test_csv_path)
    
    # If vocab and tokenizer are not provided, build them from the training data
    if vocab is None or tokenizer is None:
        tokenizer, vocab = get_vocab_and_tokenizer(train_df)
    
    # Define the label mapping dictionary
    label_mapping = {"hate": 1, "noHate": 0}

    # Map the numeric labels to their corresponding string labels in train_df
    train_df['label'] = train_df['label'].map(label_mapping)
    test_df['label'] = test_df['label'].map(label_mapping)

    # Split the train data into train and validation sets
    train, valid = train_test_split(train_df, test_size=test_size, random_state=random_state)


    # Create DataLoader instances for train, validation, and test datasets
    train_dataset = TextDataset(train, vocab, tokenizer, max_length)
    valid_dataset = TextDataset(valid, vocab, tokenizer, max_length)
    test_dataset = TextDataset(test_df, vocab, tokenizer, max_length) if test_df is not None else None


    # Create DataLoader instances for train, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=64, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_batch) if test_dataset is not None else None

    return train_loader, valid_loader, test_loader, vocab, tokenizer
