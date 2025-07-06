import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import matplotlib.pyplot as plt

def plot_gradient_flow(grad_history, model_name, save_path=None):
    """Визуализирует поток градиентов по слоям"""
    plt.figure(figsize=(12, 6))
    
    # Собираем данные по слоям
    layers = list(grad_history[0].keys())
    max_grads = {layer: [] for layer in layers}
    mean_grads = {layer: [] for layer in layers}
    
    for epoch_stats in grad_history:
        for layer, stats in epoch_stats.items():
            max_grads[layer].append(stats['max'])
            mean_grads[layer].append(stats['mean'])
    
    # Рисуем графики
    plt.subplot(1, 2, 1)
    for layer in layers:
        plt.plot(max_grads[layer], label=layer)
    plt.title('Max Gradient Magnitude')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for layer in layers:
        plt.plot(mean_grads[layer], label=layer)
    plt.title('Mean Gradient Magnitude')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient')
    plt.legend()
    
    plt.suptitle(f'Gradient Flow for {model_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_learning_curves(history, title="Learning Curves", save_path=None):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Accuracy')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(model, loader, device, save_path=None):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_maps(model, input_tensor, save_path=None):
    """Визуализирует feature maps первого сверточного слоя"""
    model.eval()
    
    # Получаем активации первого слоя
    activations = None
    def hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    handle = model.features[0].register_forward_hook(hook)
    with torch.no_grad():
        model(input_tensor)
    handle.remove()
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    for i in range(min(16, activations.shape[1])):  # Первые 16 каналов
        plt.subplot(4, 4, i+1)
        plt.imshow(activations[0, i].cpu(), cmap='viridis')
        plt.axis('off')
    plt.suptitle('First Layer Feature Maps')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def compare_models(histories, model_names, save_path=None):
    """Сравнивает несколько моделей на одном графике"""
    plt.figure(figsize=(12, 6))
    
    # Графики точности
    plt.subplot(1, 2, 1)
    for history, name in zip(histories, model_names):
        plt.plot(history['test_acc'], label=name)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Графики потерь
    plt.subplot(1, 2, 2)
    for history, name in zip(histories, model_names):
        plt.plot(history['test_loss'], label=name)
    plt.title('Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()