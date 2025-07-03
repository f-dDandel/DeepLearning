import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.dataset_utils import get_mnist_loaders, get_cifar_loaders
from utils.model_utils import create_model, count_parameters
from utils.visualization_utils import plot_learning_curves
# 2.1 Создайте модели с различной шириной слоев:
# - Узкие слои: [64, 32, 16]
# - Средние слои: [256, 128, 64]
# - Широкие слои: [1024, 512, 256]
# - Очень широкие слои: [2048, 1024, 512]
# 
# Для каждого варианта:
# - Поддерживайте одинаковую глубину (3 слоя)
# - Сравните точность и время обучения
# - Проанализируйте количество параметров
def create_width_configs():
    """Конфигурации для экспериментов с шириной"""
    return {
        "narrow": [64, 32, 16],
        "medium": [256, 128, 64], 
        "wide": [1024, 512, 256],
        "xwide": [2048, 1024, 512]
    }

def run_width_experiment(dataset='mnist', epochs=15):
    """ 2.1: Сравнение моделей разной ширины"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # загрузка данных
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_loaders(batch_size=256) 
        input_size = 784
    else:
        train_loader, test_loader = get_cifar_loaders(batch_size=256)
        input_size = 32*32*3
    
    width_configs = create_width_configs()
    results = {}

    for name, widths in width_configs.items():
        print(f"\nTraining {name} model ({widths}) on {dataset}...")
        
        # создаем конфигурацию модели
        layers = []
        prev_size = input_size
        for width in widths:
            layers.extend([
                {"type": "linear", "size": width},
                {"type": "relu"}
            ])
            prev_size = width
        
        config = {
            "input_size": input_size,
            "num_classes": 10,
            "layers": layers
        }
        
        # создаем и обучаем модель
        model = create_model(config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        
        history = {
            'train_losses': [], 'test_losses': [],
            'train_accs': [], 'test_accs': []
        }
        
        start_time = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            # обучение
            model.train()
            train_loss, train_acc = 0, 0
            for inputs, labels in train_loader:
                inputs = inputs.view(inputs.size(0), -1).to(device)  # Предварительно преобразуем и переносим
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                train_acc += (predicted == labels).sum().item()
                train_loss += loss.item()
            #тесты
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            history['train_losses'].append(train_loss/len(train_loader))
            history['test_losses'].append(test_loss)
            history['train_accs'].append(train_acc/len(train_loader.dataset))
            history['test_accs'].append(test_acc)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{epochs} - Train Acc: {history['train_accs'][-1]:.4f}, "
                  f"Test Acc: {history['test_accs'][-1]:.4f}, Time: {epoch_time:.2f}s")
        
        training_time = time.time() - start_time
        
        # сохраняем результаты
        results[name] = {
            'widths': widths,
            'history': history,
            'time': training_time,
            'params': count_parameters(model),
            'final_train_acc': history['train_accs'][-1],
            'final_test_acc': history['test_accs'][-1]
        }
        
        # вывод информации о модели
        print(f"\nResults for {name} model:")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Number of parameters: {results[name]['params']}")
        print(f"Final train accuracy: {results[name]['final_train_acc']:.4f}")
        print(f"Final test accuracy: {results[name]['final_test_acc']:.4f}")
        
        # сохраняем графики и модель
        plot_learning_curves(
            history, 
            title=f"{dataset} - {name} ({widths})",
            save_path=f"plots/width_experiments/{dataset}_{name}.png"
        )
        torch.save(model.state_dict(), f"results/width_experiments/{dataset}_{name}.pth")
    
    # Сохраняем сводные результаты
    with open(f"results/width_experiments/{dataset}_width_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def evaluate(model, test_loader, criterion, device):
    """Оценка модели на тестовых данных"""
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    return test_loss/len(test_loader), correct/len(test_loader.dataset)

# 2.2 Найдите оптимальную архитектуру:
# - Используйте поиск по сетке для поиска лучшей комбинации
# - Попробуйте различные схемы изменения ширины (расширение, сужение, постоянная ширина)
# - Визуализируйте результаты в виде тепловой карты
def optimize_architecture(dataset='mnist', epochs=10):
    """2.2: Оптимизация архитектуры"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # зпгрузка данных
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_loaders(batch_size=256) 
        input_size = 784
    else:
        train_loader, test_loader = get_cifar_loaders(batch_size=256)
        input_size = 32*32*3
    
    # параметры для поиска
    first_layer_sizes = [128, 256, 512]
    second_layer_sizes = [64, 128, 256]
    third_layer_sizes = [32, 64, 128]
    
    results = []
    total_combinations = len(first_layer_sizes) * len(second_layer_sizes) * len(third_layer_sizes)
    current_combination = 0
    
    for fl in first_layer_sizes:
        for sl in second_layer_sizes:
            for tl in third_layer_sizes:
                current_combination += 1
                widths = [fl, sl, tl]
                print(f"\nTesting architecture {widths} ({current_combination}/{total_combinations})...")
                
                # создаем модель
                layers = [
                    {"type": "linear", "size": fl}, {"type": "relu"},
                    {"type": "linear", "size": sl}, {"type": "relu"},
                    {"type": "linear", "size": tl}, {"type": "relu"}
                ]
                
                config = {
                    "input_size": input_size,
                    "num_classes": 10,
                    "layers": layers
                }
                
                model = create_model(config).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters())
                
                # краткое обучение (для поиска)
                start_time = time.time()
                
                # начальная оценка
                _, init_acc = evaluate(model, test_loader, criterion, device)
                
                for epoch in range(epochs):
                    epoch_start = time.time()
                    model.train()
                    train_loss, train_acc = 0, 0
                    for inputs, labels in train_loader:
                        inputs = inputs.view(inputs.size(0), -1).to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        train_acc += (predicted == labels).sum().item()
                        train_loss += loss.item()
                    
                    # оценка после каждой эпохи
                    _, test_acc = evaluate(model, test_loader, criterion, device)
                    epoch_time = time.time() - epoch_start
                    print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc/len(train_loader.dataset):.4f}, "
                          f"Test Acc: {test_acc:.4f}, Time: {epoch_time:.2f}s")
                
                _, final_acc = evaluate(model, test_loader, criterion, device)
                time_taken = time.time() - start_time
                
                results.append({
                    'widths': widths,
                    'init_test_acc': init_acc,
                    'final_test_acc': final_acc,
                    'time': time_taken,
                    'params': count_parameters(model)
                })
                
                # информация о текущей конфигурации
                print(f"\nResults for {widths}:")
                print(f"Training time: {time_taken:.2f} seconds")
                print(f"Number of parameters: {results[-1]['params']}")
                print(f"Initial test accuracy: {init_acc:.4f}")
                print(f"Final test accuracy: {final_acc:.4f}")
                print(f"Improvement: {final_acc - init_acc:.4f}")
    
    # сохраняем результаты поиска
    with open(f"results/width_experiments/{dataset}_grid_search.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # визуализация тепловой карты
    plot_heatmaps(results, first_layer_sizes, second_layer_sizes, third_layer_sizes, dataset)
    
    return results

def plot_heatmaps(results, fl_sizes, sl_sizes, tl_sizes, dataset):
    """Визуализация результатов поиска по сетке с несколькими тепловыми картами"""
    acc_matrix = np.zeros((len(fl_sizes), len(sl_sizes), len(tl_sizes)))
    param_matrix = np.zeros_like(acc_matrix)
    
    for res in results:
        i = fl_sizes.index(res['widths'][0])
        j = sl_sizes.index(res['widths'][1])
        k = tl_sizes.index(res['widths'][2])
        acc_matrix[i,j,k] = res['final_test_acc']
        param_matrix[i,j,k] = res['params'] / 1e6  # млн
    
    # тепловые карты для каждого третьего слоя
    for k, tl in enumerate(tl_sizes):
        plt.figure(figsize=(10, 8))
        plt.imshow(acc_matrix[:,:,k], cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Test Accuracy')
        plt.xticks(np.arange(len(sl_sizes)), labels=sl_sizes)
        plt.yticks(np.arange(len(fl_sizes)), labels=fl_sizes)
        plt.xlabel('Second Layer Size')
        plt.ylabel('First Layer Size')
        plt.title(f'{dataset.upper()} Architecture Optimization\nThird Layer Size: {tl}')
        
        #  аннотации с точностью
        for i in range(len(fl_sizes)):
            for j in range(len(sl_sizes)):
                plt.text(j, i, f"{acc_matrix[i,j,k]:.2f}",
                         ha="center", va="center", color="w")
                
        
        plt.savefig(f'plots/width_experiments/{dataset}_heatmap_tl{tl}.png')
        plt.close()
    
    # тепловая карта со средними значениями

    #в предыдущем варианте кода создавалась общая heatmap, усредненная по третьему слою, которая сохранилась в артефактах (cifar/mnist_heatmap_general)
    
    plt.figure(figsize=(10, 8))
    mean_acc = acc_matrix.mean(axis=2)
    plt.imshow(mean_acc, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Average Test Accuracy')
    plt.xticks(np.arange(len(sl_sizes)), labels=sl_sizes)
    plt.yticks(np.arange(len(fl_sizes)), labels=fl_sizes)
    plt.xlabel('Second Layer Size')
    plt.ylabel('First Layer Size')
    plt.title(f'{dataset.upper()} Architecture Optimization\nAverage Across Third Layer Sizes')
    
    # аннотации с точностью
    for i in range(len(fl_sizes)):
        for j in range(len(sl_sizes)):
            plt.text(j, i, f"{mean_acc[i,j]:.2f}",
                     ha="center", va="center", color="w")
    
    plt.savefig(f'plots/width_experiments/{dataset}_heatmap_mean.png')
    plt.close()

if __name__ == "__main__":
    
    # 2.1
    print("Running Width Experiments...")
    start_time = time.time()
    mnist_width_results = run_width_experiment(dataset='mnist')
    cifar_width_results = run_width_experiment(dataset='cifar')
    print(f"\nWidth experiments completed in {(time.time() - start_time)/60:.2f} minutes")
    
    # 2.2
    print("\nRunning Architecture Optimization...")
    start_time = time.time()
    mnist_opt_results = optimize_architecture(dataset='mnist')
    cifar_opt_results = optimize_architecture(dataset='cifar')
    print(f"\nArchitecture optimization completed in {(time.time() - start_time)/60:.2f} minutes")