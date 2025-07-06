import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn_models import ConvSizeExperiment, DepthExperiment
from utils.training_utils import train_model
from utils.visualization_utils import plot_learning_curves, plot_feature_maps, compare_models, plot_gradient_flow
from utils.comparison_utils import compare_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# 2.1 Исследуйте влияние размера ядра свертки:
# - 3x3 ядра
# - 5x5 ядра
# - 7x7 ядра
# - Комбинация разных размеров (1x1 + 3x3)
# 
# Для каждого варианта:
# - Поддерживайте одинаковое количество параметров
# - Сравните точность и время обучения
# - Проанализируйте рецептивные поля
# - Визуализируйте активации первого слоя
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

def run_kernel_size_experiment():
    """2.1: Эксперимент с размером ядра свертки"""
    print("\n=== Kernel Size Experiment ===")
    train_loader, test_loader = get_cifar_loaders()
    
    # конфигурации для сравнения
    configs = {
        "3x3": {"kernels": [3, 3, 3]},
        "5x5": {"kernels": [5, 5, 5]},
        "7x7": {"kernels": [7, 7, 7]},
        "1x1+3x3": {"kernels": [1, 3, 3]}
    }
    
    results = {}
    for name, config in configs.items():
        print(f"\nTraining model with {name} kernels...")
        model = ConvSizeExperiment(
            input_channels=3,
            num_classes=10,
            kernels=config["kernels"]
        ).to(device)
        
        start_time = time.time()
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=15,
            lr=0.001,
            device=device,
            model_name=f"kernel_{name}"
        )
        
        # визуализация feature maps первого слоя
        plot_feature_maps(
            model, 
            test_loader.dataset[0][0].unsqueeze(0).to(device),
            save_path=f"plots/kernel_size/{name}_features.png"
        )
        
        results[name] = {
            "history": history,
            "train_time": time.time() - start_time,
            "num_params": sum(p.numel() for p in model.parameters()),
            "receptive_field": model.calculate_receptive_field()
        }
    
    # сохранение результатов
    compare_metrics(
        results, 
        dataset="Kernel Size", 
        save_path="results/architecture_analysis/kernel_size/metrics.txt"
    )

# 2.2 Исследуйте влияние глубины CNN:
# - Неглубокая CNN (2 conv слоя)
# - Средняя CNN (4 conv слоя)
# - Глубокая CNN (6+ conv слоев)
# - CNN с Residual связями
# 
# Для каждого варианта:
# - Сравните точность и время обучения
# - Проанализируйте vanishing/exploding gradients
# - Исследуйте эффективность Residual связей
# - Визуализируйте feature maps
def run_depth_experiment():
    """2.2: Эксперимент с глубиной сети"""
    print("\n=== Depth Experiment ===")
    train_loader, test_loader = get_cifar_loaders()
    
    # конфигурации для сравнения
    configs = {
        "Shallow (2)": {"num_blocks": 2},
        "Medium (4)": {"num_blocks": 4},
        "Deep (6)": {"num_blocks": 6},
        "Residual": {"num_blocks": 6, "residual": True}
    }
    
    # получаем 1 батч для визуализации
    sample_data, _ = next(iter(test_loader))
    sample_input = sample_data[0].unsqueeze(0).to(device)
    
    results = {}
    for name, config in configs.items():
        print(f"\nTraining {name} model...")
        model = DepthExperiment(
            input_channels=3,
            num_classes=10,
            num_blocks=config["num_blocks"],
            residual=config.get("residual", False)
        ).to(device)
        
        start_time = time.time()
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=15,
            lr=0.001,
            device=device,
            model_name=f"depth_{name}"
        )
        
        # визуализация
        plot_learning_curves(
            history,
            title=f"{name} Model Learning Curves",
            save_path=f"plots/depth/{name}_learning.png"
        )
        
        plot_feature_maps(
            model,
            sample_input,
            save_path=f"plots/depth/{name}_features.png"
        )
        
        if history.get('gradients'):
            plot_gradient_flow(
                history['gradients'],
                model_name=name,
                save_path=f"plots/depth/{name}_gradients.png"
            )
        
        results[name] = {
            "history": history,
            "train_time": time.time() - start_time,
            "num_params": sum(p.numel() for p in model.parameters())
        }
        
        torch.save(
            model.state_dict(),
            f"results/architecture_analysis/depth/{name}_model.pth"
        )
    
    # сохранение и сравнение результатов
    compare_metrics(
        results, 
        dataset="Depth Experiment", 
        save_path="results/architecture_analysis/depth/metrics.txt"
    )
    
    # сравнение всех моделей на одном графике
    compare_models(
        [r['history'] for r in results.values()],
        list(results.keys()),
        save_path="plots/depth/comparison.png"
    )

if __name__ == "__main__":
    run_kernel_size_experiment()
    run_depth_experiment()