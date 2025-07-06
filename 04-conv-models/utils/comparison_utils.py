import json
import os

def compare_metrics(results, dataset="MNIST", save_path=None):
    output = f"\n=== {dataset} Model Comparison ===\n"
    output += "{:<15} {:<10} {:<10} {:<10} {:<10}\n".format(
        "Model", "Train Acc", "Test Acc", "Params", "Time (s)")
    
    for name, res in results.items():
        output += "{:<15} {:<10.4f} {:<10.4f} {:<10} {:<10.2f}\n".format(
            name,
            res['history']['train_acc'][-1],
            res['history']['test_acc'][-1],
            res['num_params'],
            res['train_time']
        )
    
    print(output)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(output)

def save_results(results, file_path):
    """Сохраняет результаты в JSON файл"""
    # Преобразуем данные для сериализации
    serializable_results = {}
    for model_name, data in results.items():
        serializable_results[model_name] = {
            'train_acc': data['history']['train_acc'],
            'test_acc': data['history']['test_acc'],
            'train_loss': data['history']['train_loss'],
            'test_loss': data['history']['test_loss'],
            'num_params': data['num_params'],
            'train_time': data['train_time']
        }
    
    # Создаем папку, если ее нет
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Сохраняем в файл
    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)