import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.fc_models import FCModel
from models.cnn_models import SimpleCNN, ResidualCNN
from utils.training_utils import train_model
from utils.visualization_utils import plot_learning_curves, plot_confusion_matrix, plot_gradient_flow
from utils.comparison_utils import compare_metrics, save_results

# 1.1 Сравните производительность на MNIST:
# - Полносвязная сеть (3-4 слоя)
# - Простая CNN (2-3 conv слоя)
# - CNN с Residual Block
# 
# Для каждого варианта:
# - Обучите модель с одинаковыми гиперпараметрами
# - Сравните точность на train и test множествах
# - Измерьте время обучения и инференса
# - Визуализируйте кривые обучения
# - Проанализируйте количество параметров
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_mnist_loaders(batch_size=64):
    """Загрузка данных MNIST"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_cifar_loaders(batch_size=64):
    """Загрузка данных CIFAR-10"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def run_mnist_comparison():
    """Сравнение моделей на MNIST"""
    print("\n=== MNIST Comparison ===")
    train_loader, test_loader = get_mnist_loaders()
    
    # Модели для сравнения
    models = {
        "FC": FCModel(input_size=28*28, hidden_sizes=[512, 256, 128], num_classes=10).to(device),
        "SimpleCNN": SimpleCNN(input_channels=1, num_classes=10).to(device),
        "ResidualCNN": ResidualCNN(input_channels=1, num_classes=10).to(device)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        start_time = time.time()
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=10,
            lr=0.001,
            device=device
        )
        
        train_time = time.time() - start_time
        results[name] = {
            "history": history,
            "train_time": train_time,
            "num_params": sum(p.numel() for p in model.parameters())
        }
        
        # Сохранение модели и результатов
        torch.save(model.state_dict(), f"results/mnist_comparison/{name}_model.pth")
        plot_learning_curves(history, title=f"{name} on MNIST", 
                            save_path=f"plots/mnist_{name}_learning.png")
    
    # Сравнение результатов
    compare_metrics(results, dataset="MNIST", save_path="results/mnist_comparison/metrics.txt")

# 1.2 Сравните производительность на CIFAR-10:
# - Полносвязная сеть (глубокая)
# - CNN с Residual блоками
# - CNN с регуляризацией и Residual блоками
# 
# Для каждого варианта:
# - Обучите модель с одинаковыми гиперпараметрами
# - Сравните точность и время обучения
# - Проанализируйте переобучение
# - Визуализируйте confusion matrix
# - Исследуйте градиенты (gradient flow)
def run_cifar_comparison():
    """Сравнение моделей на CIFAR-10"""
    print("\n=== CIFAR-10 Comparison ===")
    train_loader, test_loader = get_cifar_loaders()
    
    # Модели для сравнения
    models = {
        "FC_Deep": FCModel(input_size=32*32*3, hidden_sizes=[1024, 512, 256, 128], num_classes=10).to(device),
        "ResidualCNN": ResidualCNN(input_channels=3, num_classes=10).to(device),
        "ResidualCNN_Regularized": ResidualCNN(input_channels=3, num_classes=10, dropout_rate=0.3).to(device)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        start_time = time.time()
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=15,
            lr=0.001,
            device=device
        )
        # Визуализация градиентов
        if history['gradients']:
            plot_gradient_flow(history['gradients'], name,
                             save_path=f"plots/cifar_{name}_gradients.png")
        
        train_time = time.time() - start_time
        results[name] = {
            "history": history,
            "train_time": train_time,
            "num_params": sum(p.numel() for p in model.parameters())
        }
        
        # Сохранение и визуализация
        torch.save(model.state_dict(), f"results/cifar_comparison/{name}_model.pth")
        plot_learning_curves(history, title=f"{name} on CIFAR-10", 
                            save_path=f"plots/cifar_{name}_learning.png")
        
        # Confusion matrix
        plot_confusion_matrix(model, test_loader, device, 
                             save_path=f"plots/cifar_{name}_confusion.png")
    
    # Сравнение результатов
    compare_metrics(results, dataset="CIFAR-10", save_path="results/cifar_comparison/metrics.txt")

if __name__ == "__main__":
    run_mnist_comparison()
    run_cifar_comparison()