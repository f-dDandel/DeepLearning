from torch.utils.data import Dataset
from models.tokenizer_utils import load_tokenizer

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Tokenize the entire text
        self.tokens = self.tokenizer.encode(self.text).ids
        
    def __len__(self):
        return len(self.tokens) // self.max_length
    
    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length
        tokens = self.tokens[start_idx:end_idx]
        
        # Pad if necessary
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.token_to_id('[PAD]')] * (self.max_length - len(tokens))
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids