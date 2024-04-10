import pandas as pd
import torch
from torch.utils.data import Dataset

def pad_or_truncate(tokenized_sequence, seq_len = 1024, 
                    pad_id = 0, eos_id = 2):
    if len(tokenized_sequence) <= seq_len:
        left = seq_len - len(tokenized_sequence)
        padding = [pad_id] * left
        tokenized_sequence += padding
    else:
        tokenized_sequence = tokenized_sequence[:seq_len]
        tokenized_sequence[-1] = eos_id
    return tokenized_sequence

class TextDataset(Dataset):
    def __init__(self, csv_file_path, tokenizer, max_length = 1024):
        self.txt_list = pd.read_csv(csv_file_path)['text']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.txt_list)
    def __getitem__(self, index): 
        encode_text = self.tokenizer.Encode(self.txt_list[index])
        X = pad_or_truncate([self.tokenizer.bos_id()] + encode_text, seq_len=self.max_length)
        y = pad_or_truncate(encode_text + [self.tokenizer.eos_id()], seq_len=self.max_length)
        return torch.tensor(X), torch.tensor(y)
        # return X, y

            
        