import json
import torch
import torch.nn as nn

def create_model(config):
    """Создает модель на основе конфигурации"""
    layers = []
    prev_size = config["input_size"]
    
    for layer_spec in config["layers"]:
        layer_type = layer_spec["type"]
        
        if layer_type == "linear":
            out_size = layer_spec["size"]
            layers.append(nn.Linear(prev_size, out_size))
            prev_size = out_size
        elif layer_type == "relu":
            layers.append(nn.ReLU())
        elif layer_type == "dropout":
            rate = layer_spec.get("rate", 0.5)
            layers.append(nn.Dropout(rate))
        elif layer_type == "batch_norm":
            layers.append(nn.BatchNorm1d(prev_size))
    
    layers.append(nn.Linear(prev_size, config["num_classes"]))
    return nn.Sequential(*layers)

def count_parameters(model):
    """Считает количество обучаемых параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    """Сохраняет модель на диск"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Загружает модель с диска"""
    model.load_state_dict(torch.load(path))
    return model