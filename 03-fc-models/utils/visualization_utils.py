import matplotlib.pyplot as plt
import os
import numpy as np

def plot_learning_curves(history, title="", save_path=None):
    """Рисует кривые обучения (loss и accuracy)"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train')
    plt.plot(history['test_losses'], label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train')
    plt.plot(history['test_accs'], label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    if title:
        plt.suptitle(title)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_depth_comparison(results, metric='test_acc', title="", save_path=None):
    """Сравнивает модели разной глубины по выбранной метрике"""
    depths = sorted(results.keys())
    values = [results[d][metric] for d in depths]
    
    plt.figure(figsize=(8, 5))
    plt.plot(depths, values, 'o-')
    plt.title(title or f'Model Comparison by Depth ({metric})')
    plt.xlabel('Model Depth')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_weight_distributions(weights_dict, title="", save_path=None, bins=50):
    """Визуализирует распределения весов для разных слоев модели"""
    plt.figure(figsize=(12, 6))
    
    for i, (layer_name, weights) in enumerate(weights_dict.items()):
        plt.subplot(1, len(weights_dict), i+1)
        plt.hist(weights, bins=bins, alpha=0.7)
        plt.title(layer_name.replace('.weight', ''))
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()