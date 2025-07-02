import torch
from torch.utils.data import DataLoader,Dataset

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data  # list of (input_text, output_text) tuples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, output_text = self.data[idx]
        
        # Tokenisera input och output och konvertera till tensorer
        input_tokens = torch.tensor(self.tokenizer.encode(input_text), dtype=torch.long)
        output_tokens = torch.tensor(self.tokenizer.encode(output_text), dtype=torch.long)
        
        return input_tokens, output_tokens