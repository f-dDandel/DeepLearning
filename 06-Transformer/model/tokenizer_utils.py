from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os
from pathlib import Path

def train_tokenizer(text_files: list, vocab_size: int = 30000, save_path: str = "tokenizer.json"):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<s>", "</s>"],
        vocab_size=vocab_size
    )
    
    tokenizer.train(files=text_files, trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer

def load_tokenizer(tokenizer_path: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    return tokenizer