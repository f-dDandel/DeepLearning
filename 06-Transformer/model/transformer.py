import torch
import torch.nn as nn
import math
import re
from .tokenizer_utils import load_tokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Proper attention mask handling
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape mask for multi-head attention
        # Input mask shape: [batch_size, 1, seq_len, seq_len]
        # Output mask shape: [batch_size * num_heads, seq_len, seq_len]
        attn_mask = mask.repeat(1, self.num_heads, 1, 1)  # [batch_size, num_heads, seq_len, seq_len]
        attn_mask = attn_mask.view(-1, seq_len, seq_len)  # [batch_size * num_heads, seq_len, seq_len]
        
        # Self-attention with proper mask
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=False
        )
        
        # Residual connection and layer norm
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class GeneratorTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8, 
                 d_ff: int = 2048, num_layers: int = 6, dropout: float = 0.1, 
                 max_len: int = 512, pad_token_id: int = 0, tokenizer=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, vocab_size)
        
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        
        # Precompute causal mask
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))
        
    def create_mask(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        
        # Create padding mask
        pad_mask = (x == self.pad_token_id).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
        
        # Get causal mask for current sequence length
        causal_mask = self.causal_mask[:seq_len, :seq_len]  # [seq_len, seq_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Combine masks
        mask = causal_mask.masked_fill(pad_mask, float('-inf'))  # [batch_size, 1, seq_len, seq_len]
        
        return mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
        
        # Create proper mask
        mask = self.create_mask(x)
        
        # Embedding and positional encoding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        return self.projection(x)
    
    def generate_with_beam_search(
        self, 
        prompt: str,
        max_length: int = 100,
        num_beams: int = 5,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = None,
        repetition_penalty: float = 2.2
    ):
        self.eval()
        # Перед генерацией:
        prompt = re.sub(r"[^a-z0-9\s]", "", prompt.lower())  
        with torch.no_grad():
            # Токенизация промпта
            input_ids = self.tokenizer.encode(prompt).ids
            if not input_ids:
                return ""
                
            device = next(self.parameters()).device
            input_tensor = torch.tensor([input_ids], device=device)
            
            # Инициализация beam search
            beams = [{
                'tokens': input_tensor.clone(),
                'score': 0.0,
                'finished': False
            }]
            
            for _ in range(max_length):
                candidates = []
                
                for beam in beams:
                    if beam['finished']:
                        candidates.append(beam)
                        continue
                    
                    # Получаем предсказания
                    outputs = self(beam['tokens'])
                    next_token_logits = outputs[0, -1, :] / temperature
                    
                    # Применяем штраф за повторения
                    if repetition_penalty != 1.0:
                        for token_id in set(beam['tokens'][0].tolist()):
                            next_token_logits[token_id] /= repetition_penalty
                    
                    # Фильтрация (top-k + top-p)
                    if top_k > 0:
                        top_k_thresh = torch.topk(next_token_logits, min(top_k, len(next_token_logits))).values[-1]
                        next_token_logits[next_token_logits < top_k_thresh] = float('-inf')
                        
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Выбираем топ-N кандидатов для этого луча
                    probs = torch.softmax(next_token_logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs, num_beams)
                    
                    for i in range(num_beams):
                        new_tokens = torch.cat([
                            beam['tokens'], 
                            top_indices[i].unsqueeze(0).unsqueeze(0)
                        ], dim=1)
                        
                        new_score = beam['score'] + torch.log(top_probs[i])
                        
                        is_finished = (eos_token_id is not None and 
                                    top_indices[i].item() == eos_token_id)
                        
                        candidates.append({
                            'tokens': new_tokens,
                            'score': new_score,
                            'finished': is_finished
                        })
                
                # Сортируем всех кандидатов по score и выбираем топ-N
                candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
                beams = candidates[:num_beams]
                
                # Проверяем, все ли лучи завершены
                if all(beam['finished'] for beam in beams):
                    break
            
            # Выбираем лучший вариант
            best_beam = beams[0]
            generated_text = self.tokenizer.decode(
                best_beam['tokens'][0].tolist(),
                skip_special_tokens=True
            )
            
            return generated_text