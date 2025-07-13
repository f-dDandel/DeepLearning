import os
import torch
import argparse
from model.transformer import GeneratorTransformer
from model.tokenizer_utils import load_tokenizer
from utils.helpers import safe_print, setup_encoding
from config import config

def chat_interface(model_path: str = None):
    setup_encoding()
    
    if model_path is None:
        # Find the latest checkpoint
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"No checkpoints directory found at {checkpoint_dir}. Please train the model first.")
        
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if not checkpoints:
            raise FileNotFoundError(f"No model checkpoints found in {checkpoint_dir}. Please train the model first.")
        
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        model_path = os.path.join(checkpoint_dir, checkpoints[-1])
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # Load tokenizer
    if 'tokenizer' in checkpoint:
        temp_tokenizer_path = 'temp_tokenizer.json'
        with open(temp_tokenizer_path, 'w', encoding='utf-8') as f:
            f.write(checkpoint['tokenizer'])
        tokenizer = load_tokenizer(temp_tokenizer_path)
        os.remove(temp_tokenizer_path)
    else:
        tokenizer = load_tokenizer(config['tokenizer_path'])
    
    # Initialize model
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
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print("Chat with the model. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
                
            # Правильное формирование входного тензора
            input_ids = tokenizer.encode(user_input).ids
            if not input_ids:  # Если промпт пустой после токенизации
                print("Bot: Please provide a valid input")
                continue
                
            # Генерация ответа с обработкой ошибок
            try:
                response = model.generate_with_beam_search(
                    prompt=user_input,
                    max_length=100,
                    temperature=0.3,
                    num_beams=5,
                    top_k=40,
                    top_p=0.9,
                    eos_token_id=tokenizer.token_to_id('</s>')
                )
                print(f"Bot: {response}")
            except RuntimeError as e:
                print(f"Bot: Error generating response - {str(e)}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer Text Generator')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--chat', action='store_true', help='Start chat interface')
    parser.add_argument('--model', type=str, help='Path to model checkpoint', default=None)
    args = parser.parse_args()
    
    if args.train:
        from train import train_model
        print("Training mode...")
        train_model()
    elif args.chat:
        print("Chat mode...")
        chat_interface(args.model)
    else:
        print("Please specify either --train or --chat")