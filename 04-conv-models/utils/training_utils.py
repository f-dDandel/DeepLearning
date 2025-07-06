import torch
from tqdm import tqdm
import time
import torch.nn as nn
import torch.optim as optim
from utils.visualization_utils import plot_gradient_flow

def log_gradient_stats(model):
    """Логирует статистику градиентов по всем слоям модели"""
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            grad_stats[name] = {
                'max': grad.abs().max().item(),
                'mean': grad.abs().mean().item(),
                'min': grad.abs().min().item()
            }
    return grad_stats

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu', model_name='model'):
    """Полная функция обучения с отслеживанием градиентов"""
    
    # Инициализация
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'gradients': []  # Для хранения статистики градиентов
    }
    
    # Цикл обучения
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Процесс обучения с прогресс-баром
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
            
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, target)
                
                # Backward pass и оптимизация
                optimizer.zero_grad()
                loss.backward()
                
                # Собираем статистику градиентов перед шагом оптимизации
                if epoch == epochs-1:  # Собираем только на последней эпохе
                    grad_stats = log_gradient_stats(model)
                    history['gradients'].append(grad_stats)
                
                optimizer.step()
                
                # Считаем метрики
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Обновляем прогресс-бар
                tepoch.set_postfix(loss=loss.item(), accuracy=100.*correct/total)
        
        # Сохраняем метрики обучения
        train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Валидация
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Визуализация градиентов
    if history['gradients']:
        plot_gradient_flow(history['gradients'], model_name, 
                         save_path=f"plots/gradient_flow_{model_name}.png")
    
    return history

def evaluate(model, loader, criterion, device):
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return loss / len(loader), correct / len(loader.dataset)