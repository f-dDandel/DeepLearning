import time
import json
import torch
from torch.utils.data import DataLoader

def get_model_config(depth, input_size, hidden_size=256, use_reg=False):
    """Генерирует конфигурацию модели заданной глубины"""
    layers = []
    
    for i in range(depth - 1):
        layers.append({"type": "linear", "size": hidden_size})
        if use_reg:
            layers.append({"type": "batch_norm"})
            layers.append({"type": "dropout", "rate": 0.2})
        layers.append({"type": "relu"})
    
    return {
        "input_size": input_size,
        "num_classes": 10,
        "layers": layers
    }

def run_training_epoch(model, data_loader, criterion, optimizer=None, device="cpu"):
    """Выполняет одну эпоху обучения или валидации"""
    model.train() if optimizer else model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        if optimizer:
            optimizer.zero_grad()
        
        outputs = model(inputs.view(inputs.size(0), -1))
        loss = criterion(outputs, labels)
        
        if optimizer:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(data_loader), correct / total

def evaluate_model(model, data_loader, device="cpu"):
    """Оценивает точность модели на данных"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.view(inputs.size(0), -1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total