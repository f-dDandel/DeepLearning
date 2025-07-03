import matplotlib.pyplot as plt
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from utils import model_utils, experiment_utils, visualization_utils, dataset_utils

# 1.1 Создайте и обучите модели с различным количеством слоев:
# - 1 слой (линейный классификатор)
# - 2 слоя (1 скрытый)
# - 3 слоя (2 скрытых)
# - 5 слоев (4 скрытых)
# - 7 слоев (6 скрытых)
# 
# Для каждого варианта:
# - Сравните точность на train и test
# - Визуализируйте кривые обучения
# - Проанализируйте время обучения


def run_depth_experiment(dataset='mnist', depths=[1, 2, 3, 5, 7], epochs=15):
    """ 1.1: Сравнение моделей разной глубины"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загрузка данных
    if dataset == 'mnist':
        train_loader, test_loader = dataset_utils.get_mnist_loaders(batch_size=128)
        input_size = 784
    else:
        train_loader, test_loader = dataset_utils.get_cifar_loaders(batch_size=128)
        input_size = 32*32*3
    
    results = {}
    
    for depth in depths:
        print(f"\nTraining {depth}-layer model on {dataset}...")
        
        # конфигурация модели
        config = experiment_utils.get_model_config(
            depth=depth,
            input_size=input_size,
            use_reg=True
        )
        
        # создание и обучение модели
        model = model_utils.create_model(config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        
        history = {
            'train_losses': [],
            'test_losses': [],
            'train_accs': [],
            'test_accs': []
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss, train_acc = experiment_utils.run_training_epoch(
                model, train_loader, criterion, optimizer, device
            )
            test_loss, test_acc = experiment_utils.run_training_epoch(
                model, test_loader, criterion, None, device
            )
            
            history['train_losses'].append(train_loss)
            history['test_losses'].append(test_loss)
            history['train_accs'].append(train_acc)
            history['test_accs'].append(test_acc)
        
        training_time = time.time() - start_time
        
        # сохранение результатов
        results[depth] = {
            'history': history,
            'training_time': training_time,
            'parameters': model_utils.count_parameters(model),
            'final_train_acc': history['train_accs'][-1],
            'final_test_acc': history['test_accs'][-1]
        }

        # сохранение модели
        model_path = f"results/depth_experiments/{dataset}_model_depth_{depth}_reg.pth"
        torch.save(model.state_dict(), model_path)
        
        # сохранение логов обучения
        log_path = f"results/depth_experiments/{dataset}_log_depth_{depth}_reg.json"
        with open(log_path, 'w') as f:
            json.dump({
                'config': config,
                'history': history,
                'metrics': {
                    'training_time': training_time,
                    'parameters': model_utils.count_parameters(model),
                    'final_train_acc': history['train_accs'][-1],
                    'final_test_acc': history['test_accs'][-1]
                }
            }, f, indent=4)
        
        # визуализация
        visualization_utils.plot_learning_curves(
            history,
            title=f"{dataset.upper()} - Depth {depth}",
            save_path=f"plots/{dataset}_reg_depth_{depth}.png"
        )

    # сохранение сводных результатов
    summary_path = f"results/depth_experiments/{dataset}_summary_reg.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # сравнительный анализ
    visualization_utils.plot_depth_comparison(
        results,
        metric='final_test_acc',
        title=f"{dataset.upper()} Accuracy by Model Depth",
        save_path=f"plots/{dataset}_reg_depth_comparison.png"
    )
    
    return results

# 1.2 Исследуйте влияние глубины на переобучение:
# - Постройте графики train/test accuracy по эпохам
# - Определите оптимальную глубину для каждого датасета
# - Добавьте Dropout и BatchNorm, сравните результаты
# - Проанализируйте, когда начинается переобучение


def analyze_overfitting(results, dataset):
    """ 1.2: Анализ переобучения"""
    depths = sorted(results.keys())
    gaps = []
    training_times = []
    
    print("\nOverfitting Analysis:")
    print(f"{'Depth':<10}{'Train Acc':<15}{'Test Acc':<15}{'Gap':<15}{'Time (s)':<15}")
    
    for depth in depths:
        train_acc = results[depth]['final_train_acc']
        test_acc = results[depth]['final_test_acc']
        time_taken = results[depth]['training_time']
        gap = train_acc - test_acc
        gaps.append(gap)
        training_times.append(time_taken)
        print(f"{depth:<10}{train_acc:.4f}{'':<10}{test_acc:.4f}{'':<10}{gap:.4f}{'':<10}{time_taken:.2f}")
    
    # визуализация переобучения
    plt.figure(figsize=(10, 5))
    plt.plot(depths, gaps, 'o-')
    plt.title(f'Overfitting Gap by Depth on {dataset.upper()}')
    plt.xlabel('Model Depth')
    plt.ylabel('Train-Test Accuracy Gap')
    plt.grid(True)
    plt.savefig(f"plots/{dataset}_reg_overfitting_gap.png")
    plt.close()
    
    # оптимальная глубина (минимальный gap)
    optimal_depth = depths[gaps.index(min(gaps))]
    print(f"\nOptimal depth for {dataset}: {optimal_depth} layers (smallest gap)")

    # сохранение анализа переобучения
    analysis_path = f"results/depth_experiments/{dataset}_overfitting_analysis_reg.json"
    with open(analysis_path, 'w') as f:
        json.dump({
            'depths': depths,
            'gaps': gaps,
            'training_times': training_times,
            'optimal_depth': optimal_depth
        }, f, indent=4)

if __name__ == "__main__":

    #  1.1 + 1.2 для MNIST
    print("RUNNING MNIST EXPERIMENTS")
    mnist_results = run_depth_experiment(dataset='mnist')
    analyze_overfitting(mnist_results, 'mnist')
    
    #  1.1 + 1.2 для CIFAR
    print("RUNNING CIFAR EXPERIMENTS")
    cifar_results = run_depth_experiment(dataset='cifar')
    analyze_overfitting(cifar_results, 'cifar')