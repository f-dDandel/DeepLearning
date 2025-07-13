from tqdm import tqdm
import torch
import torch.nn as nn
import os
from pathlib import Path
from model.transformer import GeneratorTransformer
from model.tokenizer_utils import train_tokenizer, load_tokenizer
from data.dataset import TextDataset
from torch.utils.data import DataLoader
from utils.helpers import safe_print, setup_encoding

def train_model():
    setup_encoding()
    # Configuration
    config = {
        'batch_size': 1,
        'max_length': 192,
        'd_model': 1024,
        'num_heads': 16,
        'd_ff': 4096,
        'num_layers': 8,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'num_epochs': 3,
        'save_dir': 'checkpoints',
        'text_file': 'text_data.txt',  # Path to your text file
        'tokenizer_path': 'tokenizer.json',
        'vocab_size': 30000
    }
    
    # Check if text file exists
    if not Path(config['text_file']).exists():
        raise FileNotFoundError(f"Text file not found: {config['text_file']}. Please create it with your training data.")
    
    # Create tokenizer if not exists
    if not Path(config['tokenizer_path']).exists():
        safe_print("Training tokenizer...")
        train_tokenizer([config['text_file']], config['vocab_size'], config['tokenizer_path'])
    
    tokenizer = load_tokenizer(config['tokenizer_path'])
    # Вывести все символы в словаре токенизатора
    safe_print("Special tokens:", tokenizer.get_vocab_size())
    for token in tokenizer.get_vocab():
        if not token.isascii():
            safe_print(f"Non-ASCII token: {token}")
    
    # Create dataset and dataloader
    dataset = TextDataset(config['text_file'], tokenizer, config['max_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    safe_print(f"Using device: {device}")
    
    model = GeneratorTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_len=config['max_length'],
        pad_token_id=tokenizer.token_to_id('[PAD]'),
        tokenizer=tokenizer
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(config['save_dir'], f"model_epoch_{epoch+1}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'tokenizer': tokenizer.to_str()
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    return model