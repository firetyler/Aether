import torch
from torch.utils.data import Dataset
from ai_Model.utils.logger_common import get_logger

logger = get_logger("ChatDataset")

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        logger.info(f"ChatDataset initialized with {len(self.data)} records, max_len={self.max_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, out = self.data[idx]

        input_tokens = self.tokenizer.encode(inp)
        output_tokens = self.tokenizer.encode(out)

        input_tokens = input_tokens[:self.max_len]
        output_tokens = output_tokens[:self.max_len]

        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(output_tokens, dtype=torch.long)
        )