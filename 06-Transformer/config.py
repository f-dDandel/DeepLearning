import os
from pathlib import Path

# Общие настройки
class Config:
    def __init__(self):
        self.batch_size = 1
        self.max_length = 192
        self.d_model = 1024
        self.num_heads = 16
        self.d_ff = 4096
        self.num_layers = 8
        self.dropout = 0.1
        self.learning_rate = 1e-4
        self.num_epochs = 3
        self.save_dir = 'checkpoints'
        self.text_file = 'text_data.txt'
        self.tokenizer_path = 'tokenizer.json'
        self.vocab_size = 30000

config = Config()